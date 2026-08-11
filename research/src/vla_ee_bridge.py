"""
Bridge UnifoLM-VLA 23D end-effector actions ↔ 29D G1 joint targets for the ESN.

UnifoLM-VLA-Base (g1_wipe_table) predicts EE poses in R6 layout:
  [L_xyz(3), L_rot6d(6), R_xyz(3), R_rot6d(6), waist5(5)]

The CUDA ESN was trained on 29D ``observation.body`` joint targets from
``unitreerobotics/G1_Dex1_Wipe_Table``. This module converts VLA EE outputs
into joint-space targets and builds the 23D proprio vector VLA expects.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

import mujoco

from src.step1_profile_unifolm_vla0 import G1_DOF
from src.step2_esn_cuda_ridge import extract_joint_state_29d

EE_STATE_DIM = 23
QPOS_ACTUATED_START = 7
LEFT_ARM_SLICE = slice(15, 22)
RIGHT_ARM_SLICE = slice(22, 29)
WAIST_SLICE = slice(12, 15)
LEG_SLICE = slice(0, 12)

LEFT_HAND_BODY = "left_wrist_yaw_link"
RIGHT_HAND_BODY = "right_wrist_yaw_link"

# Live UnifoLM EE xyz can leave the wipe workspace; clamp before IK.
WIPE_EE_XYZ_LO = np.array([0.05, -0.60, 0.72], dtype=np.float64)
WIPE_EE_XYZ_HI = np.array([0.70, 0.20, 1.20], dtype=np.float64)
# Per-tick joint rate limit at ~100 Hz (crawl toward targets without flailing).
MAX_DQ_LEG = 0.015
MAX_DQ_WAIST = 0.04
MAX_DQ_ARM = 0.06

ARM_JOINT_NAMES = (
    (
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ),
    (
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ),
)


def rotmat_to_rot6d(rot: np.ndarray) -> np.ndarray:
    """First two columns of a rotation matrix flattened (6D rotation rep)."""
    return rot.reshape(3, 3)[:, :2].reshape(6).astype(np.float32)


def build_wipe_table_model(robot_mjcf: Path) -> mujoco.MjModel:
    """Compile G1 + wipe table (static cloth geom) aligned with ``mujoco_wipe_scene``."""
    # Keep IK / Step-3 static scene consistent with interactive Step-4 geometry.
    from src.mujoco_wipe_scene import (
        CLOTH_HALF_EXTENTS,
        CLOTH_TABLE_POS,
        TABLE_BODY_POS,
        TABLE_TOP_HALF_EXTENTS,
    )

    robot_mjcf = robot_mjcf.resolve()
    hx, hy, hz = CLOTH_HALF_EXTENTS
    tx, ty, tz = TABLE_BODY_POS
    thx, thy, thz = TABLE_TOP_HALF_EXTENTS
    leg_half = max(0.05, float(tz) * 0.5)
    leg_z = -leg_half
    scene_xml = f"""
<mujoco model="g1_wipe_table_scene">
  <include file="{robot_mjcf.name}"/>
  <worldbody>
    <body name="wipe_table" pos="{tx} {ty} {tz}">
      <geom name="table_top" type="box" size="{thx} {thy} {thz}" rgba="0.55 0.38 0.22 1"/>
      <geom name="table_leg_fl" type="cylinder" pos="{0.75*thx} {0.75*thy} {leg_z}" size="0.02 {leg_half}" rgba="0.35 0.35 0.35 1"/>
      <geom name="table_leg_fr" type="cylinder" pos="{0.75*thx} {-0.75*thy} {leg_z}" size="0.02 {leg_half}" rgba="0.35 0.35 0.35 1"/>
      <geom name="table_leg_bl" type="cylinder" pos="{-0.75*thx} {0.75*thy} {leg_z}" size="0.02 {leg_half}" rgba="0.35 0.35 0.35 1"/>
      <geom name="table_leg_br" type="cylinder" pos="{-0.75*thx} {-0.75*thy} {leg_z}" size="0.02 {leg_half}" rgba="0.35 0.35 0.35 1"/>
    </body>
    <geom name="wipe_cloth" type="box" pos="{CLOTH_TABLE_POS[0]} {CLOTH_TABLE_POS[1]} {CLOTH_TABLE_POS[2]}"
          size="{hx} {hy} {hz}" rgba="0.92 0.88 0.55 1"/>
  </worldbody>
</mujoco>
"""
    scene_path = robot_mjcf.parent / "_g1_wipe_table_scene_runtime.xml"
    try:
        scene_path.write_text(scene_xml, encoding="utf-8")
    except OSError:
        scene_path = Path("/tmp") / "_g1_wipe_table_scene_runtime.xml"
        scene_xml = scene_xml.replace(
            f'<include file="{robot_mjcf.name}"/>',
            f'<include file="{robot_mjcf}"/>',
        )
        scene_path.write_text(scene_xml, encoding="utf-8")
    try:
        return mujoco.MjModel.from_xml_path(str(scene_path))
    finally:
        scene_path.unlink(missing_ok=True)


def resolve_robot_mjcf(user_path: Optional[str] = None) -> Path:
    """Resolve the base G1 robot MJCF (without scene furniture)."""
    if user_path:
        path = Path(user_path).expanduser().resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(f"MJCF not found: {path}")
    for candidate in (
        Path(os.environ.get("G1_MJCF", "")),
        Path.home() / "unitree_mujoco" / "unitree_robots" / "g1" / "g1_29dof.xml",
        Path("/home/aihimekpen/research/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.xml"),
    ):
        if str(candidate) and candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("G1 robot MJCF not found.")


def load_wipe_table_init_joints(
    episode_index: int = 0,
    dataset_id: str = "unitreerobotics/G1_Dex1_Wipe_Table",
) -> np.ndarray:
    """Return the 29D ``observation.body`` pose from a UnifoLM G1 LeRobot dataset."""
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split="train", streaming=True)
    for row in ds:
        if int(row["episode_index"]) == episode_index:
            return extract_joint_state_29d(row).numpy().astype(np.float32)
    raise ValueError(f"Episode {episode_index} not found in {dataset_id}")


def _body_id(model: mujoco.MjModel, name: str) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise ValueError(f"Body not found in MJCF: {name}")
    return bid


def _joint_dof_indices(model: mujoco.MjModel, joint_names: Sequence[str]) -> np.ndarray:
    dofs = []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Joint not found in MJCF: {name}")
        dofs.append(int(model.jnt_dofadr[jid]))
    return np.asarray(dofs, dtype=np.int32)


def set_joint_positions_in_data(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joints_29d: np.ndarray,
) -> None:
    data.qpos[QPOS_ACTUATED_START : QPOS_ACTUATED_START + G1_DOF] = np.asarray(
        joints_29d, dtype=np.float64
    ).reshape(-1)
    mujoco.mj_forward(model, data)


def joints_to_ee_proprio(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joints_29d: np.ndarray,
) -> np.ndarray:
    """Build the 23D proprio vector expected by UnifoLM-VLA (g1_wipe_table)."""
    set_joint_positions_in_data(model, data, joints_29d)
    left_id = _body_id(model, LEFT_HAND_BODY)
    right_id = _body_id(model, RIGHT_HAND_BODY)

    left_xyz = data.xpos[left_id].astype(np.float32)
    right_xyz = data.xpos[right_id].astype(np.float32)
    left_rot6d = rotmat_to_rot6d(data.xmat[left_id])
    right_rot6d = rotmat_to_rot6d(data.xmat[right_id])
    waist = np.asarray(joints_29d, dtype=np.float32)[WAIST_SLICE]
    waist5 = np.pad(waist, (0, max(0, 5 - waist.size)))[:5]

    return np.concatenate([left_xyz, left_rot6d, right_xyz, right_rot6d, waist5], axis=0)


def _ik_arm_to_position(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joints_29d: np.ndarray,
    *,
    arm_slice: slice,
    hand_body: str,
    joint_names: Sequence[str],
    target_xyz: np.ndarray,
    steps: int = 24,
    step_scale: float = 0.35,
) -> np.ndarray:
    """Jacobian IK: nudge arm joints so the hand body reaches ``target_xyz``."""
    target = np.asarray(joints_29d, dtype=np.float64).copy()
    dof_ids = _joint_dof_indices(model, joint_names)
    hand_id = _body_id(model, hand_body)
    target_xyz = np.asarray(target_xyz, dtype=np.float64).reshape(3)
    n_arm = arm_slice.stop - arm_slice.start

    for _ in range(steps):
        set_joint_positions_in_data(model, data, target)
        err = target_xyz - data.xpos[hand_id]
        if float(np.linalg.norm(err)) < 1e-3:
            break

        jacp = np.zeros((3, model.nv), dtype=np.float64)
        jacr = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jac(model, data, jacp, jacr, data.xpos[hand_id], hand_id)
        jac_arm = jacp[:, dof_ids]
        dq = jac_arm.T @ np.linalg.solve(jac_arm @ jac_arm.T + 1e-4 * np.eye(3), err)
        target[arm_slice] += step_scale * dq[:n_arm]

    return target.astype(np.float32)


def clip_ee_action_to_wipe_workspace(ee_action_23d: np.ndarray) -> np.ndarray:
    """Clamp left/right EE xyz into the wipe table workspace before IK."""
    ee = np.asarray(ee_action_23d, dtype=np.float32).reshape(-1).copy()
    if ee.size < EE_STATE_DIM:
        ee = np.pad(ee, (0, EE_STATE_DIM - ee.size))
    ee = ee[:EE_STATE_DIM]
    ee[0:3] = np.clip(ee[0:3], WIPE_EE_XYZ_LO, WIPE_EE_XYZ_HI)
    ee[9:12] = np.clip(ee[9:12], WIPE_EE_XYZ_LO, WIPE_EE_XYZ_HI)
    return ee


def clip_joints_to_model_limits(
    model: mujoco.MjModel,
    joints_29d: np.ndarray,
) -> np.ndarray:
    """Clip actuated joints to MuJoCo ``jnt_range`` when limited."""
    q = np.asarray(joints_29d, dtype=np.float32).copy().reshape(-1)
    if q.size != G1_DOF:
        raise ValueError(f"Expected {G1_DOF} joints, got {q.size}")
    actuated_i = 0
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        if actuated_i >= G1_DOF:
            break
        if model.jnt_limited[j]:
            lo, hi = model.jnt_range[j]
            q[actuated_i] = float(np.clip(q[actuated_i], lo, hi))
        actuated_i += 1
    return q


def stabilize_joint_command(
    cmd_29d: np.ndarray,
    *,
    current_29d: np.ndarray,
    freeze_legs_to: Optional[np.ndarray] = None,
    max_dq_leg: float = MAX_DQ_LEG,
    max_dq_waist: float = MAX_DQ_WAIST,
    max_dq_arm: float = MAX_DQ_ARM,
    model: Optional[mujoco.MjModel] = None,
    rate_from: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Keep live wipe commands from exploding: freeze legs, rate-limit, joint limits.

    Rate-limit against ``rate_from`` (previous command) when provided so the
    controller can still crawl toward a distant VLA/ESN target. Limiting only
    vs measured proprio freezes motion when the ESN stays near the robot state.
    """
    cmd = np.asarray(cmd_29d, dtype=np.float32).reshape(-1).copy()
    cur = np.asarray(current_29d, dtype=np.float32).reshape(-1)
    ref = np.asarray(rate_from if rate_from is not None else cur, dtype=np.float32).reshape(-1)
    if freeze_legs_to is not None:
        legs = np.asarray(freeze_legs_to, dtype=np.float32).reshape(-1)
        cmd[LEG_SLICE] = legs[LEG_SLICE]
    else:
        cmd[LEG_SLICE] = cur[LEG_SLICE]

    def _rate(slc: slice, max_dq: float) -> None:
        d = cmd[slc] - ref[slc]
        cmd[slc] = ref[slc] + np.clip(d, -max_dq, max_dq)

    _rate(LEG_SLICE, max_dq_leg)
    _rate(WAIST_SLICE, max_dq_waist)
    _rate(LEFT_ARM_SLICE, max_dq_arm)
    _rate(RIGHT_ARM_SLICE, max_dq_arm)
    if model is not None:
        cmd = clip_joints_to_model_limits(model, cmd)
    return cmd.astype(np.float32)


def ee_action_to_joint_target(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ee_action_23d: np.ndarray,
    current_joints_29d: np.ndarray,
) -> np.ndarray:
    """
    Convert a denormalized 23D VLA EE action into a 29D joint target for the ESN.

    Leg joints are held near the current state; waist and arms track the VLA EE pose.
    """
    ee = clip_ee_action_to_wipe_workspace(ee_action_23d)

    current = np.asarray(current_joints_29d, dtype=np.float32)
    target = current.copy()
    target[LEG_SLICE] = current[LEG_SLICE]
    target[WAIST_SLICE] = np.clip(ee[18:21], -0.8, 0.8)

    target = _ik_arm_to_position(
        model,
        data,
        target,
        arm_slice=LEFT_ARM_SLICE,
        hand_body=LEFT_HAND_BODY,
        joint_names=ARM_JOINT_NAMES[0],
        target_xyz=ee[0:3],
    )
    target = _ik_arm_to_position(
        model,
        data,
        target,
        arm_slice=RIGHT_ARM_SLICE,
        hand_body=RIGHT_HAND_BODY,
        joint_names=ARM_JOINT_NAMES[1],
        target_xyz=ee[9:12],
    )
    target = clip_joints_to_model_limits(model, target)
    target[LEG_SLICE] = current[LEG_SLICE]
    return target.astype(np.float32)


def press_right_cloth_to_table(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joints_29d: np.ndarray,
    *,
    table_top_z: float,
    cloth_half_thickness: float = 0.004,
    hand_offset: Optional[np.ndarray] = None,
    contact_gap_m: float = 0.015,
    blend: float = 0.45,
    ik_steps: int = 18,
) -> np.ndarray:
    """
    Lower the right arm so the cloth attach frame sits on the table (wipe press).

    Live UnifoLM often holds the hand too high after grasp; this geometric prior
    keeps cloth underside within the table-contact band while preserving XY.
    """
    from src.mujoco_wipe_scene import CLOTH_HAND_OFFSET

    offset = np.asarray(
        hand_offset if hand_offset is not None else CLOTH_HAND_OFFSET,
        dtype=np.float64,
    ).reshape(3)
    q = np.asarray(joints_29d, dtype=np.float32).copy()
    set_joint_positions_in_data(model, data, q)
    hand_id = _body_id(model, RIGHT_HAND_BODY)
    rot = data.xmat[hand_id].reshape(3, 3)
    hand_pos = data.xpos[hand_id].copy()
    attach = hand_pos + rot @ offset

    # Cloth center height that puts the underside ~contact_gap_m above the table.
    target_cloth_z = float(table_top_z) + float(cloth_half_thickness) + float(contact_gap_m)
    target_cloth = np.array([attach[0], attach[1], target_cloth_z], dtype=np.float64)
    target_hand = target_cloth - rot @ offset

    pressed = _ik_arm_to_position(
        model,
        data,
        q,
        arm_slice=RIGHT_ARM_SLICE,
        hand_body=RIGHT_HAND_BODY,
        joint_names=ARM_JOINT_NAMES[1],
        target_xyz=target_hand,
        steps=ik_steps,
        step_scale=0.40,
    )
    alpha = float(np.clip(blend, 0.0, 1.0))
    out = q.copy()
    out[RIGHT_ARM_SLICE] = (1.0 - alpha) * q[RIGHT_ARM_SLICE] + alpha * pressed[RIGHT_ARM_SLICE]
    return out.astype(np.float32)
