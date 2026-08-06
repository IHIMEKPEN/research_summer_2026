"""
============================================================
Step 3 — Live UnifoLM Wipe Success (Layer F)
============================================================

Dual-process live UnifoLM + bridge (ESN/ZOH/…) with interactive mocap cloth
and the same wipe-task metrics as Step 4.

Unlike Step 4 (dataset-oracle demo tokens), VLA targets come from **live**
UnifoLM-VLA-Base. Dex1 gripper is not in the UnifoLM EE action — cloth grasp
uses proximity-gated synthetic gripper (see WipeClothController).

Step 3 dual-thread reports remain timing-only; this module is the task-success
companion.

Usage (from research/):
  export HF_HOME=/raid/data/aihimekpen/hf_cache
  python3 -m src.step3_live_wipe_eval --bridge esn --duration_s 30 --record_video
  python3 -m src.step3_live_wipe_eval --bridges esn,zoh --duration_s 20
  python3 -m src.step3_live_wipe_eval --mock --duration_s 5   # plumbing smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.mujoco_wipe_scene import RIGHT_HAND_BODY, TABLE_TOP_Z
from src.paths import models_path, results_path
from src.step3_control_baselines import online_linear_command, online_pid_command
from src.step3_dual_thread_mujoco import (
    ALL_BRIDGES,
    DEFAULT_INSTRUCTION,
    DEFAULT_UNNORM_KEY,
    DEFAULT_VLA_HZ,
    G1_DOF,
    MAX_DURATION_S,
    TARGET_HZ,
    VIDEO_FPS,
    VLA_IMAGE_SIZE,
    VLA_LOAD_TIMEOUT_S,
    BridgeMode,
    SharedMemoryRegisters,
    _finalize_step_stats,
    _vla_perception_worker,
    read_init_joints,
    read_vla_token,
    resolve_esn_checkpoint,
    resolve_mjcf_path,
    save_video_mp4,
    write_init_joints,
    write_observation,
    write_vla_token,
)
from src.step2_esn_cuda_ridge import load_checkpoint
from src.step4_mujoco_evaluation import G1WipeTableEvalEnv
from src.vla_ee_bridge import load_wipe_table_init_joints
from src.wipe_task_metrics import WipeTaskMetrics, WipeTaskMetricsRecorder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = results_path("step3_live_wipe")


@dataclass
class LiveWipeStats:
    """Timing + wipe-task metrics for one live UnifoLM trial."""

    bridge: str = "esn"
    mock_vla: bool = False
    steps: int = 0
    duration_s: float = 0.0
    mean_step_ms: float = 0.0
    max_step_ms: float = 0.0
    max_step_ms_steady: float = 0.0
    p99_step_ms: float = 0.0
    achieved_control_hz: float = 0.0
    vla_ticks: int = 0
    grasp_frames: int = 0
    video_path: Optional[str] = None
    task_metrics: Optional[WipeTaskMetrics] = None
    gripper_mode: str = "proximity_synthetic"
    init_episode: int = 160
    instruction: str = DEFAULT_INSTRUCTION

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.task_metrics is not None:
            d["task_metrics"] = self.task_metrics.to_dict()
        return d


def _live_wipe_control_worker(
    registers: SharedMemoryRegisters,
    result_queue: mp.Queue,
    *,
    mjcf_path: str,
    esn_checkpoint: str,
    duration_s: float,
    control_hz: float,
    device_str: str,
    mock: bool,
    record_video: bool,
    video_path: str,
    video_fps: float,
    image_shape: Tuple[int, int, int],
    init_episode: int,
    bridge: str,
    vla_hz: float,
    instruction: str,
) -> None:
    """Process B: 100 Hz MuJoCo + interactive cloth + wipe metrics."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if bridge not in ALL_BRIDGES:
        result_queue.put(RuntimeError(f"Unknown bridge mode: {bridge}"))
        return

    stats = LiveWipeStats(
        bridge=bridge,
        mock_vla=mock,
        duration_s=duration_s,
        init_episode=init_episode,
        instruction=instruction,
    )
    step_times: list[float] = []
    video_frames: List[np.ndarray] = []
    grasp_frames = 0

    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        result_queue.put(RuntimeError("CUDA required for live wipe control process."))
        return

    init_joints = read_init_joints(registers)
    env = G1WipeTableEvalEnv(
        mjcf_path=Path(mjcf_path),
        control_hz=control_hz,
        enable_video_renderer=record_video,
        init_joints_29d=init_joints,
        wipe_table_scene=False,
        pd_kp=120.0,
        pd_kd=8.0,
    )
    # Cloth rest: default workspace pose (aligned table). Optionally nudge from
    # dataset init episode later; live VLA must reach it from camera.
    if env.cloth is not None:
        env.cloth.reset()

    esn = None
    if bridge == "esn":
        esn = load_checkpoint(esn_checkpoint, device=device)
        esn.eval()

    init_q = env.get_joint_positions()
    write_vla_token(registers, init_q)

    dt = 1.0 / control_hz
    hold_ticks = max(1, int(round(control_hz / max(vla_hz, 1e-6))))
    joint_gpu = torch.zeros(G1_DOF, device=device, dtype=torch.float32)
    vla_gpu = torch.zeros(G1_DOF, device=device, dtype=torch.float32)

    prev_token = init_q.astype(np.float32).copy()
    curr_token = init_q.astype(np.float32).copy()
    pid_q = init_q.astype(np.float32).copy()
    last_seq = 0
    ticks_since_update = 0
    metrics_rec = WipeTaskMetricsRecorder(
        control_hz=control_hz,
        table_top_z=TABLE_TOP_Z,
    )

    def _publish_observation() -> Tuple[np.ndarray, np.ndarray]:
        image = env.render_rgb()
        joint_pos = env.get_joint_positions()
        ee_proprio = env.get_ee_proprio()
        write_observation(registers, image, joint_pos, ee_proprio, image_shape)
        return image, joint_pos

    wait_timeout_s = VLA_LOAD_TIMEOUT_S if not mock else 30.0
    if registers.vla_ready.value == 0:
        logger.info("Publishing observations while VLA loads (timeout %.0fs) ...", wait_timeout_s)
        bootstrap_deadline = time.perf_counter() + wait_timeout_s
        while registers.vla_ready.value == 0 and time.perf_counter() < bootstrap_deadline:
            _publish_observation()
            time.sleep(dt)

    if registers.vla_ready.value:
        logger.info("VLA ready — live wipe loop %.1fs (bridge=%s).", duration_s, bridge)
    else:
        logger.warning("VLA not ready after %.0fs — seeded token only.", wait_timeout_s)

    t_end = time.perf_counter() + duration_s
    next_tick = time.perf_counter()
    try:
        while time.perf_counter() < t_end and not registers.stop.is_set():
            tick_start = time.perf_counter()
            _, joint_pos = _publish_observation()

            vla_token, seq = read_vla_token(registers)
            if seq != last_seq:
                prev_token = curr_token.copy()
                curr_token = np.asarray(vla_token, dtype=np.float32).reshape(G1_DOF).copy()
                last_seq = int(seq)
                ticks_since_update = 0
            else:
                ticks_since_update += 1

            if bridge == "esn":
                assert esn is not None
                joint_gpu.copy_(torch.from_numpy(joint_pos).to(device=device, dtype=torch.float32))
                vla_gpu.copy_(torch.from_numpy(curr_token).to(device=device, dtype=torch.float32))
                esn.update_vla_target(vla_gpu)
                cmd_np = esn.step_proprio(joint_gpu).detach().cpu().numpy()
            elif bridge == "zoh":
                cmd_np = curr_token
            elif bridge == "linear":
                cmd_np = online_linear_command(
                    prev_token=prev_token,
                    curr_token=curr_token,
                    ticks_since_update=ticks_since_update,
                    hold_ticks=hold_ticks,
                )
            else:
                pid_q = online_pid_command(q=pid_q, target=curr_token, dt=dt)
                cmd_np = pid_q

            env.apply_unified_control(cmd_np)
            env.step_physics()

            right_g = 4.5
            left_g = 4.5
            if env.cloth is not None:
                right_g = env.cloth.synthetic_gripper_from_proximity()
                if env.update_cloth(right_g, left_g):
                    grasp_frames += 1
                hand_attach, _ = env.cloth._hand_target_pose()
                right_id = env.model.body(RIGHT_HAND_BODY).id
                right_hand_pos = np.asarray(env.data.xpos[right_id], dtype=np.float64)
                err_sq = float(np.mean((env.get_joint_positions() - curr_token) ** 2))
                metrics_rec.record_step(
                    joint_err_sq_mean=err_sq,
                    cloth_pos=env.cloth.cloth_position(),
                    right_hand_pos=right_hand_pos,
                    right_gripper=right_g,
                    cloth_ctrl=env.cloth,
                    right_ee_target=hand_attach,
                )

            if record_video:
                video_frames.append(env.render_video_frame())

            step_times.append((time.perf_counter() - tick_start) * 1000.0)
            stats.steps += 1

            next_tick += dt
            sleep_s = next_tick - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.perf_counter()
    finally:
        registers.stop.set()
        env.close()

    if step_times:
        # Reuse dual-thread finalizer via a tiny shim object.
        from src.step3_dual_thread_mujoco import ControlLoopStats

        tmp = ControlLoopStats(bridge=bridge)
        _finalize_step_stats(step_times, tmp)
        stats.mean_step_ms = tmp.mean_step_ms
        stats.max_step_ms = tmp.max_step_ms
        stats.max_step_ms_steady = tmp.max_step_ms_steady
        stats.p99_step_ms = tmp.p99_step_ms
        stats.achieved_control_hz = tmp.esn_hz

    stats.vla_ticks = int(registers.vla_ticks.value)
    stats.grasp_frames = grasp_frames
    stats.task_metrics = metrics_rec.finalize()

    if record_video and video_frames:
        out_path = Path(video_path)
        save_video_mp4(video_frames, out_path, source_hz=control_hz, target_fps=video_fps)
        stats.video_path = str(out_path) if out_path.is_file() else (
            str(out_path.with_suffix(".gif")) if out_path.with_suffix(".gif").is_file() else None
        )

    result_queue.put(stats)


@dataclass
class LiveWipeConfig:
    mjcf_path: Path
    esn_checkpoint: str
    mock: bool = False
    duration_s: float = 30.0
    control_hz: float = TARGET_HZ
    vla_hz: float = DEFAULT_VLA_HZ
    instruction: str = DEFAULT_INSTRUCTION
    device: str = "cuda"
    record_video: bool = True
    video_path: Optional[Path] = None
    video_fps: float = VIDEO_FPS
    unnorm_key: str = DEFAULT_UNNORM_KEY
    init_episode: int = 160
    bridge: BridgeMode = "esn"


class LiveWipeController:
    """Live UnifoLM dual-process wipe eval with cloth metrics."""

    def __init__(self, config: LiveWipeConfig):
        self.config = config
        self.registers = SharedMemoryRegisters.create()

    def run(self) -> LiveWipeStats:
        cfg = self.config
        if cfg.duration_s > MAX_DURATION_S:
            raise ValueError(f"duration_s={cfg.duration_s} exceeds MAX_DURATION_S={MAX_DURATION_S}")
        if not torch.cuda.is_available() and cfg.device.startswith("cuda"):
            raise RuntimeError("CUDA required.")

        ctx = mp.get_context("spawn")
        result_queue: mp.Queue = ctx.Queue()
        image_shape = (*VLA_IMAGE_SIZE, 3)
        video_path = cfg.video_path or (
            RESULTS_DIR
            / f"live_wipe_{cfg.bridge}_{'mock' if cfg.mock else 'live'}.mp4"
        )
        video_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            init_pose = load_wipe_table_init_joints(cfg.init_episode)
            write_init_joints(self.registers, init_pose)
            logger.info("Seeded init pose from wipe episode %d.", cfg.init_episode)
        except Exception as exc:
            logger.warning("Could not load dataset init pose: %s", exc)

        vla_proc = ctx.Process(
            target=_vla_perception_worker,
            name="VLA_Perception_Process",
            args=(self.registers,),
            kwargs={
                "mock": cfg.mock,
                "instruction": cfg.instruction,
                "vla_hz": cfg.vla_hz,
                "device_str": cfg.device,
                "image_shape": image_shape,
                "mjcf_path": str(cfg.mjcf_path.resolve()),
                "unnorm_key": cfg.unnorm_key,
                "use_wipe_table_scene": True,
            },
            daemon=False,
        )
        ctrl_proc = ctx.Process(
            target=_live_wipe_control_worker,
            name="LiveWipe_Control_Process",
            args=(self.registers, result_queue),
            kwargs={
                "mjcf_path": str(cfg.mjcf_path.resolve()),
                "esn_checkpoint": cfg.esn_checkpoint,
                "duration_s": cfg.duration_s,
                "control_hz": cfg.control_hz,
                "device_str": cfg.device,
                "mock": cfg.mock,
                "record_video": cfg.record_video,
                "video_path": str(video_path),
                "video_fps": cfg.video_fps,
                "image_shape": image_shape,
                "init_episode": cfg.init_episode,
                "bridge": cfg.bridge,
                "vla_hz": cfg.vla_hz,
                "instruction": cfg.instruction,
            },
            daemon=False,
        )

        logger.info(
            "Live wipe | bridge=%s | mock=%s | duration=%.1fs | video=%s",
            cfg.bridge,
            cfg.mock,
            cfg.duration_s,
            cfg.record_video,
        )
        vla_proc.start()
        ctrl_proc.start()
        queue_timeout_s = cfg.duration_s + VLA_LOAD_TIMEOUT_S + 120.0
        try:
            stats = result_queue.get(timeout=queue_timeout_s)
        except Exception as exc:
            raise RuntimeError(
                f"Control process did not return stats (exit={ctrl_proc.exitcode}): {exc}"
            ) from exc
        finally:
            self.registers.stop.set()
            if ctrl_proc.is_alive():
                ctrl_proc.join(timeout=10.0)
            if vla_proc.is_alive():
                vla_proc.join(timeout=10.0)

        if not isinstance(stats, LiveWipeStats):
            raise RuntimeError(f"Control process failed: {stats}")
        return stats


def print_live_wipe_summary(stats: LiveWipeStats, *, report_path: Path) -> None:
    tm = stats.task_metrics
    print("\n" + "=" * 60)
    print("  Step 3 — Live UnifoLM Wipe Success (Layer F)")
    print("=" * 60)
    print(f"  Bridge / VLA      : {stats.bridge} / {'mock' if stats.mock_vla else 'live UnifoLM'}")
    print(f"  Gripper mode      : {stats.gripper_mode}")
    print(f"  Steps / duration  : {stats.steps:,} / {stats.duration_s:.1f}s")
    print(f"  Mean step / Hz    : {stats.mean_step_ms:.3f} ms / {stats.achieved_control_hz:.1f} Hz")
    print(f"  Steady max / P99  : {stats.max_step_ms_steady:.2f} / {stats.p99_step_ms:.2f} ms")
    print(f"  VLA ticks         : {stats.vla_ticks}")
    print(f"  Cloth grasped     : {stats.grasp_frames}/{stats.steps} frames")
    if tm is not None:
        print("  --- Wipe task (live) ---")
        print(f"  Grasp success     : {tm.grasp_success}  (prox={tm.grasp_proximity_error_m:.4f} m)")
        print(f"  Wipe path (XY)    : {tm.wipe_path_length_m:.3f} m")
        print(f"  Table contact     : {tm.table_contact_ratio:.1%}")
        print(f"  Wipe coverage     : {tm.wipe_coverage_m2:.4f} m²")
        print(f"  Task success      : {tm.task_success}")
    print(f"  Report JSON       : {report_path}")
    if stats.video_path:
        print(f"  Video             : {stats.video_path}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live UnifoLM wipe success (Layer F)")
    parser.add_argument("--mjcf", type=str, default=None)
    parser.add_argument("--esn_checkpoint", type=str, default=str(models_path("esn_cuda_ridge")))
    parser.add_argument("--bridge", choices=list(ALL_BRIDGES), default=None)
    parser.add_argument(
        "--modes",
        type=str,
        default=None,
        help="Comma list of control modes e.g. esn,zoh (alias: avoid --bridges; "
        "UnifoLM argv scanner treats 'bridge' as BridgeData 7-D)",
    )
    parser.add_argument(
        "--bridges",
        type=str,
        default=None,
        help=argparse.SUPPRESS,  # deprecated alias — prefer --modes
    )
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--duration_s", type=float, default=30.0)
    parser.add_argument("--control_hz", type=float, default=TARGET_HZ)
    parser.add_argument("--vla_hz", type=float, default=DEFAULT_VLA_HZ)
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION)
    parser.add_argument("--init_episode", type=int, default=160)
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--no_video", action="store_true")
    parser.add_argument("--video_fps", type=float, default=VIDEO_FPS)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--unnorm_key", type=str, default=DEFAULT_UNNORM_KEY)
    args = parser.parse_args()

    if args.duration_s > MAX_DURATION_S:
        raise ValueError(f"--duration_s must be <= {MAX_DURATION_S}")

    if args.modes:
        bridges = [b.strip() for b in args.modes.split(",") if b.strip()]
    elif args.bridges:
        bridges = [b.strip() for b in args.bridges.split(",") if b.strip()]
    elif args.bridge:
        bridges = [args.bridge]
    else:
        bridges = ["esn"]
    for b in bridges:
        if b not in ALL_BRIDGES:
            raise ValueError(f"Unknown bridge {b!r}")

    want_video = bool(args.record_video) and not args.no_video
    mjcf = resolve_mjcf_path(args.mjcf)
    ckpt = resolve_esn_checkpoint(args.esn_checkpoint)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    reports: List[Dict[str, Any]] = []
    for i, bridge in enumerate(bridges):
        record_video = want_video and i == 0  # one video (first bridge)
        video_path = RESULTS_DIR / f"live_wipe_{bridge}_{'mock' if args.mock else 'live'}.mp4"
        cfg = LiveWipeConfig(
            mjcf_path=mjcf,
            esn_checkpoint=str(ckpt),
            mock=args.mock,
            duration_s=args.duration_s,
            control_hz=args.control_hz,
            vla_hz=args.vla_hz,
            instruction=args.instruction,
            device=args.device,
            record_video=record_video,
            video_path=video_path if record_video else None,
            video_fps=args.video_fps,
            unnorm_key=args.unnorm_key,
            init_episode=args.init_episode,
            bridge=bridge,  # type: ignore[arg-type]
        )
        stats = LiveWipeController(cfg).run()
        tag = "mock" if args.mock else "live"
        report_path = RESULTS_DIR / f"live_wipe_report_{bridge}_{tag}.json"
        report = stats.to_dict()
        report["esn_checkpoint"] = str(ckpt)
        report_path.write_text(json.dumps(report, indent=2))
        reports.append(report)
        print_live_wipe_summary(stats, report_path=report_path)

    summary = {
        "n_trials": len(reports),
        "mock_vla": args.mock,
        "duration_s": args.duration_s,
        "bridges": bridges,
        "gripper_mode": "proximity_synthetic",
        "note": (
            "Live UnifoLM wipe success. Grasp uses proximity-synthetic Dex1 proxy "
            "(UnifoLM EE actions have no gripper channel). Distinct from Step-4 oracle."
        ),
        "reports": reports,
    }
    if reports:
        summary["grasp_success_rate"] = float(
            np.mean([1.0 if (r.get("task_metrics") or {}).get("grasp_success") else 0.0 for r in reports])
        )
        summary["task_success_rate"] = float(
            np.mean([1.0 if (r.get("task_metrics") or {}).get("task_success") else 0.0 for r in reports])
        )
        summary["table_contact_ratio_mean"] = float(
            np.mean([float((r.get("task_metrics") or {}).get("table_contact_ratio", 0.0)) for r in reports])
        )
    sum_path = RESULTS_DIR / f"live_wipe_summary_{'mock' if args.mock else 'live'}.json"
    sum_path.write_text(json.dumps(summary, indent=2))
    (RESULTS_DIR / "live_wipe_report.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSummary: {sum_path}")
    print(
        f"grasp={summary.get('grasp_success_rate', float('nan')):.1%} | "
        f"contact={summary.get('table_contact_ratio_mean', float('nan')):.1%} | "
        f"task_success={summary.get('task_success_rate', float('nan')):.1%}"
    )


if __name__ == "__main__":
    main()
