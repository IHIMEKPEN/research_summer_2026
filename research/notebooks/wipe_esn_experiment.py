"""Independent state-only ESN experiment used by ``main.ipynb``.

Benchmark stabilization (Aug 2026):
  - Terminate on NaN / non-finite QACC.
  - Cloth jumps > 5 cm never inflate path/coverage (see ``wipe_task_metrics``).
  - Every reported loss term is bounded to ``[0, 1]``.
  - Component losses are recorded separately.
  - ``L_teacher`` is only nonzero for a real frozen-UnifoLM cache; held-out /
    teacher-weight-0 runs report ``teacher_source="none"``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

import mujoco
import numpy as np

G1_DOF = 29
STATE_DIM = 58  # q (29) + qdot (29), not q + teacher
DT = 0.010
TEACHER_PERIOD = 0.570
ARM_SLICE = slice(15, 29)
MIN_WIPE_PATH_M = 0.768
TARGET_CONTACT_RATIO = 0.90
TARGET_TABLE_AREA_M2 = 0.36 * 0.56
TARGET_COVERAGE_RATIO = 0.90
MAX_CLOTH_JUMP_M = 0.05
# Reference scales that map raw signals into [0, 1] shortfalls / penalties.
SMOOTH_REF_MSE = 0.01  # (0.1 rad)^2 mean per-step command change
TEACHER_REF_MSE = 1.0  # rad^2 mean joint MSE at anchors
# Physical path sanity: multi-pass wipes can exceed table diagonal, but
# NaN/teleport runs previously reported tens of meters. Cap rejects those.
MAX_PLAUSIBLE_WIPE_PATH_M = 8.0

CommandFn = Callable[[float, np.ndarray, np.ndarray], np.ndarray]


def joint_state(row) -> np.ndarray:
    body = np.asarray(row["observation.body"], dtype=np.float32).reshape(-1)
    left = np.asarray(row["observation.left_arm"], dtype=np.float32).reshape(-1)
    right = np.asarray(row["observation.right_arm"], dtype=np.float32).reshape(-1)
    q = np.concatenate([body[:15], left, right])
    if q.size != G1_DOF:
        raise ValueError(f"Expected 29 joints, got {q.size}")
    return q


def pack_episodes(dataset, episode_ids: Iterable[int]) -> dict[int, dict[str, np.ndarray]]:
    wanted = set(map(int, episode_ids))
    rows: dict[int, list] = {ep: [] for ep in wanted}
    for row in dataset:
        ep = int(row["episode_index"])
        if ep in wanted:
            rows[ep].append(row)
    packed = {}
    for ep in sorted(wanted):
        ep_rows = sorted(rows[ep], key=lambda x: int(x["frame_index"]))
        if not ep_rows:
            raise ValueError(f"Episode {ep} was not found")
        packed[ep] = {
            "t": np.asarray([r["timestamp"] for r in ep_rows], dtype=np.float64),
            "q": np.stack([joint_state(r) for r in ep_rows]),
            "gl": np.asarray([r["observation.left_gripper"] for r in ep_rows], dtype=np.float32),
            "gr": np.asarray([r["observation.right_gripper"] for r in ep_rows], dtype=np.float32),
        }
    return packed


def native_states(ep: dict[str, np.ndarray]) -> np.ndarray:
    q, t = ep["q"], ep["t"]
    qd = np.zeros_like(q)
    dt = np.maximum(np.diff(t), 1e-4)
    qd[1:] = np.diff(q, axis=0) / dt[:, None]
    return np.concatenate([q, qd], axis=1).astype(np.float32)


def control_grid(ep: dict[str, np.ndarray]):
    rel = ep["t"] - ep["t"][0]
    grid = np.arange(0.0, rel[-1] + 1e-9, DT)
    q = np.stack([np.interp(grid, rel, ep["q"][:, j]) for j in range(G1_DOF)], axis=1)
    qd = np.zeros_like(q)
    qd[1:] = np.diff(q, axis=0) / DT
    next_native = np.searchsorted(rel, grid + 1e-9, side="right").clip(0, len(rel) - 1)
    target = ep["q"][next_native]
    return np.concatenate([q, qd], axis=1).astype(np.float32), target.astype(np.float32)


def teacher_cache(ep: dict[str, np.ndarray], duration_s: float | None = None):
    end = float(ep["t"][-1] - ep["t"][0]) if duration_s is None else duration_s
    times = np.arange(0.0, end + 1e-9, TEACHER_PERIOD)
    rel = ep["t"] - ep["t"][0]
    idx = np.searchsorted(rel, times, side="left").clip(0, len(rel) - 1)
    return times.astype(np.float32), ep["q"][idx].astype(np.float32)


def _bound01(x: float) -> float:
    if not np.isfinite(x):
        return 1.0
    return float(min(max(x, 0.0), 1.0))


@dataclass
class ESN:
    n: int = 116
    leak: float = 0.3
    rho: float = 0.95
    seed: int = 0
    Win: np.ndarray = field(init=False)
    W: np.ndarray = field(init=False)
    Wout: np.ndarray = field(init=False)
    mean: np.ndarray = field(init=False)
    scale: np.ndarray = field(init=False)
    h: np.ndarray = field(init=False)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.Win = rng.uniform(-0.15, 0.15, (self.n, STATE_DIM)).astype(np.float32)
        W = rng.uniform(-1, 1, (self.n, self.n)).astype(np.float32)
        W[rng.random(W.shape) < 0.90] = 0
        radius = float(np.max(np.abs(np.linalg.eigvals(W.astype(np.float64)))))
        self.W = (W * self.rho / max(radius, 1e-6)).astype(np.float32)
        self.Wout = np.zeros((G1_DOF, self.n + STATE_DIM + 1), dtype=np.float32)
        self.mean = np.zeros(STATE_DIM, dtype=np.float32)
        self.scale = np.ones(STATE_DIM, dtype=np.float32)
        self.h = np.zeros(self.n, dtype=np.float32)

    def reset(self):
        self.h.fill(0)

    def features(self, state: np.ndarray) -> np.ndarray:
        z = (np.asarray(state, dtype=np.float32) - self.mean) / self.scale
        self.h = (1 - self.leak) * self.h + self.leak * np.tanh(self.Win @ z + self.W @ self.h)
        return np.concatenate([self.h, z, np.ones(1, dtype=np.float32)])

    def act(self, state: np.ndarray, q_reference: np.ndarray | None = None) -> np.ndarray:
        out = self.Wout @ self.features(state)
        if q_reference is not None:
            out = np.clip(out, q_reference - 0.35, q_reference + 0.35)
        return out.astype(np.float32)


def fit_bc_initializer(esn: ESN, episodes: dict[int, dict[str, np.ndarray]], ridge: float = 1.0):
    all_states = np.vstack([control_grid(ep)[0][:-1] for ep in episodes.values()])
    esn.mean = all_states.mean(axis=0).astype(np.float32)
    esn.scale = np.maximum(all_states.std(axis=0), 1e-3).astype(np.float32)
    phis, targets = [], []
    for ep in episodes.values():
        states, target = control_grid(ep)
        esn.reset()
        for i in range(len(states) - 1):
            phi = esn.features(states[i])
            if i >= 10:
                phis.append(phi)
                targets.append(target[i + 1])
    P = np.asarray(phis, dtype=np.float64)
    Y = np.asarray(targets, dtype=np.float64)
    I = np.eye(P.shape[1])
    esn.Wout = np.linalg.solve(P.T @ P + ridge * I, P.T @ Y).T.astype(np.float32)
    return {"samples": len(P), "demo_hz": 30.0, "controller_hz": 100.0,
            "target_semantics": "next recorded demo configuration"}


def _interp_signal(ep, key: str, t: float):
    rel = ep["t"] - ep["t"][0]
    return float(np.interp(t, rel, ep[key]))


def _make_env(mjcf_path: Path, init_q: np.ndarray):
    from src.g1_dex1 import Dex1Binding, GRIPPER_OPEN_TYPICAL
    from src.mujoco_wipe_scene import WipeClothController, make_wipe_scene_env_model

    model = make_wipe_scene_env_model(mjcf_path, interactive_cloth=True)
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)
    bind = Dex1Binding.from_model(model, body_dof=G1_DOF)
    mujoco.mj_resetData(model, data)
    data.qpos[bind.body_qpos_adr] = init_q
    bind.set_gripper_qpos(data, GRIPPER_OPEN_TYPICAL, GRIPPER_OPEN_TYPICAL, zero_vel=True)
    initial_base = data.qpos[:7].copy()
    mujoco.mj_forward(model, data)
    cloth = WipeClothController(model, data)
    return model, data, bind, cloth, initial_base


# Free-joint DOFs 0..5 often trip mjWARN_BADQACC when we re-pin the floating
# base each substep; stiff PD also produces large-but-finite actuated qacc.
# Terminate only on non-finite state — cloth teleports are handled by jump
# rejection in wipe_task_metrics, not by qacc magnitude heuristics.
def _sim_unstable(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    del model  # reserved for future joint-range checks
    return not (
        np.isfinite(data.qpos).all()
        and np.isfinite(data.qvel).all()
        and np.isfinite(data.qacc).all()
        and np.isfinite(data.ctrl).all()
    )


def rollout_policy(
    command_fn: CommandFn,
    ep: dict[str, np.ndarray],
    mjcf_path: Path,
    *,
    max_s: float | None = None,
    teacher_joint_targets: np.ndarray | None = None,
    teacher_weight: float = 0.0,
    capture_anchors: bool = False,
    press_table: bool = False,
    policy_name: str = "policy",
    contact_controller=None,
):
    """Closed-loop MuJoCo wipe rollout with stabilized loss accounting.

    ``contact_controller``, when provided, must expose ``reset(q0)`` and
    ``project(q_intent, q_current) -> q_cmd`` (see ``wipe_contact_controller``).
    """
    from src.g1_dex1 import dex1_width_to_slide
    from src.mujoco_wipe_scene import (
        GRIPPER_GRASP_THRESHOLD, TABLE_BODY_POS,
        TABLE_TOP_HALF_EXTENTS, TABLE_TOP_Z,
        WipeClothController,
    )
    from src.wipe_task_metrics import WipeTaskMetricsRecorder

    duration = float(ep["t"][-1] - ep["t"][0])
    if max_s is not None:
        duration = min(duration, float(max_s))
    model, data, bind, cloth, initial_base = _make_env(mjcf_path, ep["q"][0])
    if press_table:
        cloth.press_to_table = True
    if contact_controller is not None:
        contact_controller.reset(ep["q"][0].astype(np.float32))
        if hasattr(contact_controller, "model"):
            contact_controller.model = model

    rel = ep["t"] - ep["t"][0]
    closed = np.flatnonzero(ep["gr"] < GRIPPER_GRASP_THRESHOLD)
    if closed.size:
        probe = int(closed[0])
        data.qpos[bind.body_qpos_adr] = ep["q"][probe]
        mujoco.mj_forward(model, data)
        attach, _ = cloth._hand_target_pose()
        cloth.set_rest_pose(WipeClothController.rest_pose_from_hand_attach(attach))
    data.qpos[bind.body_qpos_adr] = ep["q"][0]
    data.qpos[:7] = initial_base
    mujoco.mj_forward(model, data)
    cloth.reset()

    use_real_teacher = teacher_joint_targets is not None and float(teacher_weight) > 0.0
    teacher_t, proxy_q = teacher_cache(ep, duration)
    if float(teacher_weight) == 0.0:
        teacher_q = proxy_q
        teacher_source = "none"
    elif teacher_joint_targets is None:
        teacher_q = proxy_q
        teacher_source = "demonstration_proxy"
    else:
        teacher_q = np.asarray(teacher_joint_targets, dtype=np.float32)
        teacher_source = "frozen_unifolm_cache"
        if len(teacher_q) != len(teacher_t):
            raise ValueError(f"Teacher cache has {len(teacher_q)} targets for {len(teacher_t)} anchors")

    captures = {"time_s": [], "q": [], "qd": [], "ee_proprio": [], "rgb": []}
    renderer = None
    camera = None
    if capture_anchors:
        import os
        os.environ.setdefault("MUJOCO_GL", "egl")
        from src.vla_ee_bridge import joints_to_ee_proprio
        renderer = mujoco.Renderer(model, height=224, width=224)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance, camera.azimuth, camera.elevation = 2.15, 118.0, -22.0
        camera.lookat[:] = np.array([0.345, -0.2, 0.88])

    next_anchor = 0
    teacher_errors, smooth_errors, limit_penalties = [], [], []
    recorder = WipeTaskMetricsRecorder(
        control_hz=1 / DT, table_top_z=TABLE_TOP_Z,
        max_cloth_jump_m=MAX_CLOTH_JUMP_M,
        min_wipe_path_m=MIN_WIPE_PATH_M,
        min_table_contact_ratio=TARGET_CONTACT_RATIO,
        min_wipe_coverage_m2=TARGET_TABLE_AREA_M2 * TARGET_COVERAGE_RATIO,
        target_xy_bounds=(
            float(TABLE_BODY_POS[0] - TABLE_TOP_HALF_EXTENTS[0]),
            float(TABLE_BODY_POS[0] + TABLE_TOP_HALF_EXTENTS[0]),
            float(TABLE_BODY_POS[1] - TABLE_TOP_HALF_EXTENTS[1]),
            float(TABLE_BODY_POS[1] + TABLE_TOP_HALF_EXTENTS[1]),
        ),
    )
    previous_cmd = ep["q"][0].copy()
    steps = int(duration / DT)
    terminated_unstable = False
    terminate_step = None
    for k in range(steps):
        sim_t = k * DT
        q = np.asarray(data.qpos[bind.body_qpos_adr], dtype=np.float32).copy()
        qd = np.asarray(data.qvel[bind.body_dof_adr], dtype=np.float32).copy()
        if not (np.isfinite(q).all() and np.isfinite(qd).all()):
            terminated_unstable = True
            terminate_step = k
            break
        intent = np.asarray(command_fn(sim_t, q, qd), dtype=np.float32).reshape(-1)
        if intent.size != G1_DOF or not np.isfinite(intent).all():
            terminated_unstable = True
            terminate_step = k
            break
        if contact_controller is not None:
            cmd = np.asarray(contact_controller.project(intent, q), dtype=np.float32).reshape(-1)
        else:
            cmd = intent
        if cmd.size != G1_DOF or not np.isfinite(cmd).all():
            terminated_unstable = True
            terminate_step = k
            break
        smooth_errors.append(float(np.mean((cmd - previous_cmd) ** 2)))
        previous_cmd = cmd.copy()

        if next_anchor < len(teacher_t) and sim_t + DT / 2 >= teacher_t[next_anchor]:
            if use_real_teacher:
                teacher_errors.append(float(np.mean((cmd - teacher_q[next_anchor]) ** 2)))
            if renderer is not None:
                renderer.update_scene(data, camera=camera)
                captures["time_s"].append(sim_t)
                captures["q"].append(q.copy())
                captures["qd"].append(qd.copy())
                captures["ee_proprio"].append(joints_to_ee_proprio(model, data, q))
                captures["rgb"].append(renderer.render().copy())
            next_anchor += 1

        tau_body = 160.0 * (cmd - q) - 10.0 * qd
        data.ctrl.fill(0)
        for j, aid in enumerate(bind.body_actuator_ids):
            data.ctrl[int(aid)] = np.clip(tau_body[j], -80.0, 80.0)
        gl, gr = _interp_signal(ep, "gl", sim_t), _interp_signal(ep, "gr", sim_t)
        for side, width in (("left", gl), ("right", gr)):
            target = dex1_width_to_slide(width)
            for aid, qadr, dadr in zip(bind.finger_actuator_ids[side], bind.finger_qpos_adr[side], bind.finger_dof_adr[side]):
                data.ctrl[int(aid)] = 300.0 * (target - data.qpos[qadr]) - 6.0 * data.qvel[dadr]
        for _ in range(round(DT / model.opt.timestep)):
            mujoco.mj_step(model, data)
            data.qpos[:7] = initial_base
            data.qvel[:6] = 0
            if _sim_unstable(model, data):
                terminated_unstable = True
                break
        if terminated_unstable:
            terminate_step = k
            break
        cloth.update(gr, gl)
        mujoco.mj_forward(model, data)
        if _sim_unstable(model, data):
            terminated_unstable = True
            terminate_step = k
            break
        hand_pos, _ = cloth._hand_target_pose()
        cloth_pos = cloth.cloth_position()
        if not (np.isfinite(hand_pos).all() and np.isfinite(cloth_pos).all()):
            terminated_unstable = True
            terminate_step = k
            break
        recorder.record_step(
            joint_err_sq_mean=float(np.mean((q - cmd) ** 2)),
            cloth_pos=cloth_pos, right_hand_pos=hand_pos,
            right_gripper=gr, cloth_ctrl=cloth,
        )
        joint_ids = model.actuator_trnid[bind.body_actuator_ids, 0].astype(np.int32)
        limited = np.zeros(G1_DOF, dtype=bool)
        has_limit = model.jnt_limited[joint_ids].astype(bool)
        limited[has_limit] = ((cmd[has_limit] < model.jnt_range[joint_ids[has_limit], 0])
                              | (cmd[has_limit] > model.jnt_range[joint_ids[has_limit], 1]))
        limit_penalties.append(float(np.mean(limited)))

    metrics = recorder.finalize()
    if renderer is not None:
        renderer.close()

    # Bounded component losses in [0, 1].
    L_grasp = 0.0 if metrics.grasp_success else 1.0
    L_path = _bound01(1.0 - metrics.wipe_path_length_m / MIN_WIPE_PATH_M)
    L_contact = _bound01(1.0 - metrics.table_contact_ratio / TARGET_CONTACT_RATIO)
    L_coverage = _bound01(
        1.0 - metrics.wipe_coverage_m2 / (TARGET_TABLE_AREA_M2 * TARGET_COVERAGE_RATIO)
    )
    L_smooth = _bound01(float(np.mean(smooth_errors)) / SMOOTH_REF_MSE) if smooth_errors else 0.0
    L_limits = _bound01(float(np.mean(limit_penalties))) if limit_penalties else 0.0
    L_jump = 1.0 if metrics.max_cloth_jump_m > MAX_CLOTH_JUMP_M else 0.0
    if terminated_unstable:
        # Hard failure: saturate every integrity term.
        L_grasp = L_path = L_contact = L_coverage = L_smooth = L_limits = L_jump = 1.0

    L_task = _bound01(
        (L_grasp + L_path + L_contact + L_coverage + L_smooth + L_limits + L_jump) / 7.0
    )
    if use_real_teacher and teacher_errors:
        L_teacher = _bound01(float(np.mean(teacher_errors)) / TEACHER_REF_MSE)
    else:
        L_teacher = 0.0

    metrics.task_success = bool(
        (not terminated_unstable)
        and metrics.task_success
        and metrics.max_cloth_jump_m <= MAX_CLOTH_JUMP_M
        and L_limits == 0.0
        and metrics.wipe_path_length_m <= MAX_PLAUSIBLE_WIPE_PATH_M
    )
    result = {
        "L_task": L_task,
        "L_grasp": L_grasp,
        "L_path": L_path,
        "L_contact": L_contact,
        "L_coverage": L_coverage,
        "L_smooth": L_smooth,
        "L_limits": L_limits,
        "L_jump": L_jump,
        "L_teacher": L_teacher,
        "L_total": _bound01(L_task + float(teacher_weight) * L_teacher),
        "teacher_weight": float(teacher_weight),
        "teacher_source": teacher_source,
        "anchors": len(teacher_errors) if use_real_teacher else 0,
        "joint_limit_violation": bool(L_limits > 0.0),
        "terminated_unstable": terminated_unstable,
        "terminate_step": terminate_step,
        "policy_name": policy_name,
        "plausible_path": bool(metrics.wipe_path_length_m <= MAX_PLAUSIBLE_WIPE_PATH_M),
        **metrics.to_dict(),
    }
    if capture_anchors:
        result["captures"] = {k: np.asarray(v) for k, v in captures.items()}
    return result


def rollout(
    esn: ESN,
    ep: dict[str, np.ndarray],
    mjcf_path: Path,
    *,
    max_s: float | None = None,
    teacher_joint_targets: np.ndarray | None = None,
    teacher_weight: float = 0.0,
    capture_anchors: bool = False,
    press_table: bool = False,
):
    """Closed-loop: MuJoCo state -> ESN -> PD -> MuJoCo next state."""
    esn.reset()

    def command_fn(_t: float, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        return esn.act(np.concatenate([q, qd]), q_reference=q)

    return rollout_policy(
        command_fn, ep, mjcf_path,
        max_s=max_s,
        teacher_joint_targets=teacher_joint_targets,
        teacher_weight=teacher_weight,
        capture_anchors=capture_anchors,
        press_table=press_table,
        policy_name="esn",
    )


def spsa_finetune(esn: ESN, ep: dict[str, np.ndarray], mjcf_path: Path, *, iterations=2, seed=1):
    rng = np.random.default_rng(seed)
    history = []
    accepted_metrics = rollout(esn, ep, mjcf_path)
    for it in range(iterations):
        c = 0.01 / ((it + 1) ** 0.101)
        a = 0.002 / ((it + 1) ** 0.602)
        delta = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), esn.Wout.shape)
        base = esn.Wout.copy()
        esn.Wout = base + c * delta
        plus = rollout(esn, ep, mjcf_path)
        esn.Wout = base - c * delta
        minus = rollout(esn, ep, mjcf_path)
        grad_scale = (plus["L_total"] - minus["L_total"]) / (2 * c)
        esn.Wout = (base - a * grad_scale * delta).astype(np.float32)
        candidate = rollout(esn, ep, mjcf_path)
        accepted = candidate["L_total"] < accepted_metrics["L_total"]
        if accepted:
            accepted_metrics = candidate
        else:
            esn.Wout = base
        history.append({"iteration": it + 1, "accepted": accepted, **accepted_metrics})
    return history
