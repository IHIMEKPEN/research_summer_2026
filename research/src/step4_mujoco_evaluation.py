"""
============================================================
Step 4 — MuJoCo Wipe-Table Evaluation (Dataset Oracle + Video)
Phase 4: Full visual benchmark of ESN + scene fidelity
Research Plan: VLA + ESN for Real-Time Humanoid Control
Author: Osemudiamen Andrew Ihimekpen | PVAMU CREDIT Center
============================================================

Replays ``G1_Dex1_Wipe_Table`` through the Step 2 CUDA ESN with:
  - GT proprio + 2 Hz held VLA tokens (same as ESN training)
  - Mocap cloth grasped when Dex1 right gripper closes
  - Kinematic or PD control + 60 FPS benchmark video

Step 3 remains **dual-process only** (VLA @ 2 Hz + ESN @ 100 Hz).

Usage (from research/):
  python3 -m src.step4_mujoco_evaluation --episode 0
  python3 -m src.step4_mujoco_evaluation --control_mode pd --duration_s 12
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import numpy as np
import torch

from src.mujoco_wipe_scene import (
    RIGHT_HAND_BODY,
    WipeClothController,
    make_wipe_scene_env_model,
)from src.wipe_task_metrics import WipeTaskMetrics, WipeTaskMetricsRecorder
from src.paths import models_path, results_path
from src.step2_esn_cuda_ridge import (
    CONTROL_HZ,
    DATASET_ID,
    load_checkpoint,
    load_episode_gripper_trajectory_numpy,
    load_episode_trajectory_numpy,
)
from src.step3_dual_thread_mujoco import (
    G1_DOF,
    G1MuJoCoEnv,
    VIDEO_FPS,
    _warmup_esn_reservoir,
    load_esn_checkpoint_metadata,
    resolve_esn_checkpoint,
    resolve_mjcf_path,
    save_video_mp4,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = results_path("step4_mujoco_evaluation")
ControlMode = Literal["kinematic", "pd"]
MAX_DURATION_S = 60.0


def _loop_index(step: int, traj_steps: int) -> int:
    return int(step % traj_steps)


@dataclass
class MuJoCoEvalStats:
    steps: int = 0
    esn_hz: float = 0.0
    mean_step_ms: float = 0.0
    max_step_ms: float = 0.0
    max_step_ms_steady: float = 0.0
    p99_step_ms: float = 0.0
    tracking_mse: float = 0.0
    tracking_rmse: float = 0.0
    trajectory_steps: int = 0
    control_mode: str = "kinematic"
    init_episode: int = 0
    grasp_frames: int = 0
    episode_loops: int = 1
    video_path: Optional[str] = None
    task_metrics: Optional[WipeTaskMetrics] = None


@dataclass
class MuJoCoEvalConfig:
    mjcf_path: Path
    esn_checkpoint: str
    init_episode: int = 0
    duration_s: float = MAX_DURATION_S
    control_hz: float = CONTROL_HZ
    vla_hz: float = 2.0
    control_mode: ControlMode = "kinematic"
    use_gt_proprio_for_esn: bool = True
    record_video: bool = True
    video_path: Optional[Path] = None
    video_fps: float = VIDEO_FPS
    device: str = "cuda"
    loop_episode: bool = True


class G1WipeTableEvalEnv(G1MuJoCoEnv):
    """G1 + wipe table + mocap cloth (Step 4 visual benchmark)."""

    cloth: Optional[WipeClothController] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        import mujoco

        robot_path = self.mjcf_path if self.mjcf_path.is_file() else None
        self.model = make_wipe_scene_env_model(robot_path, interactive_cloth=True)
        self.data = mujoco.MjData(self.model)
        if self.model.nu != G1_DOF:
            raise ValueError(f"Expected {G1_DOF} actuators, got {self.model.nu}")

        h, w = self.image_size
        self.renderer = mujoco.Renderer(self.model, height=h, width=w)
        if self.enable_video_renderer:
            vh, vw = self.video_size
            self.video_renderer = mujoco.Renderer(self.model, height=vh, width=vw)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = 2.15
        self.camera.azimuth = 118.0
        self.camera.elevation = -22.0
        self.camera.lookat[:] = np.array([0.42, 0.0, 0.78])

        self.cloth = WipeClothController(self.model, self.data)
        self.reset()

    def reset(self) -> None:
        G1MuJoCoEnv.reset(self)
        if self.cloth is not None:
            self.cloth.reset()

    def update_cloth(self, right_gripper: float, left_gripper: float) -> bool:
        if self.cloth is None:
            return False
        grasped = self.cloth.update(right_gripper, left_gripper)
        import mujoco

        mujoco.mj_forward(self.model, self.data)
        return grasped


class MuJoCoWipeEvaluator:
    """Dataset-oracle replay with cloth grasp + optional video export."""

    def __init__(self, config: MuJoCoEvalConfig):
        self.config = config

    def run(self) -> MuJoCoEvalStats:
        cfg = self.config
        if cfg.duration_s > MAX_DURATION_S:
            raise ValueError(f"duration_s={cfg.duration_s} exceeds MAX_DURATION_S={MAX_DURATION_S}")
        if not torch.cuda.is_available() and cfg.device.startswith("cuda"):
            raise RuntimeError("CUDA required.")

        device = torch.device(cfg.device)
        logger.info(
            "Loading episode %d from %s @ %.0f Hz (VLA hold %.1f Hz)",
            cfg.init_episode,
            DATASET_ID,
            cfg.control_hz,
            cfg.vla_hz,
        )
        ground_truth, vla_targets = load_episode_trajectory_numpy(
            cfg.init_episode, control_hz=cfg.control_hz, vla_hz=cfg.vla_hz
        )
        left_gripper, right_gripper = load_episode_gripper_trajectory_numpy(
            cfg.init_episode, control_hz=cfg.control_hz
        )
        traj_steps = int(ground_truth.shape[0])
        episode_duration_s = traj_steps / cfg.control_hz
        sim_steps = int(round(cfg.duration_s * cfg.control_hz))
        if sim_steps > traj_steps and not cfg.loop_episode:
            logger.warning(
                "duration_s=%.1fs exceeds episode length %.2fs — capping to episode.",
                cfg.duration_s,
                episode_duration_s,
            )
            sim_steps = traj_steps
        episode_loops = max(1, int(np.ceil(sim_steps / traj_steps)))
        logger.info(
            "Episode %d: %d steps (%.2fs). Target %.1fs → %d sim steps (%d loop%s).",
            cfg.init_episode,
            traj_steps,
            episode_duration_s,
            cfg.duration_s,
            sim_steps,
            episode_loops,
            "s" if episode_loops != 1 else "",
        )

        env = G1WipeTableEvalEnv(
            mjcf_path=cfg.mjcf_path,
            control_hz=cfg.control_hz,
            enable_video_renderer=cfg.record_video,
            init_joints_29d=ground_truth[0],
            wipe_table_scene=False,
            pd_kp=220.0 if cfg.control_mode == "pd" else 120.0,
            pd_kd=14.0 if cfg.control_mode == "pd" else 8.0,
        )
        esn = load_checkpoint(cfg.esn_checkpoint, device=device)
        esn.eval()
        _warmup_esn_reservoir(esn, ground_truth, vla_targets, device)

        stats = MuJoCoEvalStats(
            control_mode=cfg.control_mode,
            init_episode=cfg.init_episode,
            trajectory_steps=sim_steps,
            episode_loops=episode_loops,
        )
        step_times: list[float] = []
        video_frames: List[np.ndarray] = []
        tracking_sq_err: list[float] = []
        grasp_frames = 0
        metrics_rec = WipeTaskMetricsRecorder(control_hz=cfg.control_hz)

        joint_gpu = torch.zeros(G1_DOF, device=device, dtype=torch.float32)
        vla_gpu = torch.zeros(G1_DOF, device=device, dtype=torch.float32)
        dt = 1.0 / cfg.control_hz
        video_path = cfg.video_path or (RESULTS_DIR / "table_wipe_benchmark.mp4")
        video_path.parent.mkdir(parents=True, exist_ok=True)

        next_tick = time.perf_counter()
        for step in range(sim_steps):
            tick_start = time.perf_counter()
            t = _loop_index(step, traj_steps)

            if cfg.loop_episode and step > 0 and t == 0:
                esn.reset_state()
                _warmup_esn_reservoir(esn, ground_truth, vla_targets, device)
                env.step_kinematic(ground_truth[0])
                if env.cloth is not None:
                    env.cloth.reset()
                    import mujoco
                    mujoco.mj_forward(env.model, env.data)

            proprio_in = ground_truth[t] if cfg.use_gt_proprio_for_esn else env.get_joint_positions()
            joint_gpu.copy_(torch.from_numpy(proprio_in).to(device=device, dtype=torch.float32))
            vla_gpu.copy_(torch.from_numpy(vla_targets[t]).to(device=device, dtype=torch.float32))
            esn.update_vla_target(vla_gpu)
            cmd_np = esn.step_proprio(joint_gpu).detach().cpu().numpy()

            if cfg.control_mode == "kinematic":
                env.step_kinematic(cmd_np)
            else:
                env.apply_unified_control(cmd_np)
                env.step_physics()

            if env.update_cloth(right_gripper[t], left_gripper[t]):
                grasp_frames += 1

            joint_after = env.get_joint_positions()
            err_sq = float(np.mean((joint_after - ground_truth[t]) ** 2))
            tracking_sq_err.append(err_sq)

            if env.cloth is not None:
                hand_attach_pose, _ = env.cloth._hand_target_pose()
                right_id = env.model.body(RIGHT_HAND_BODY).id
                right_hand_pos = np.asarray(env.data.xpos[right_id], dtype=np.float64)
                metrics_rec.record_step(
                    joint_err_sq_mean=err_sq,
                    cloth_pos=env.cloth.cloth_position(),
                    right_hand_pos=right_hand_pos,
                    right_gripper=right_gripper[t],
                    cloth_ctrl=env.cloth,
                    right_ee_target=hand_attach_pose,
                )

            if cfg.record_video:
                video_frames.append(env.render_video_frame())

            step_times.append((time.perf_counter() - tick_start) * 1000.0)
            stats.steps += 1

            next_tick += dt
            sleep_s = next_tick - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.perf_counter()

        env.close()

        if step_times:
            arr = np.asarray(step_times, dtype=np.float64)
            stats.mean_step_ms = float(arr.mean())
            stats.max_step_ms = float(arr.max())
            steady = arr[5:] if arr.size > 5 else arr
            stats.max_step_ms_steady = float(steady.max()) if steady.size else stats.max_step_ms
            stats.p99_step_ms = float(np.percentile(arr, 99))
            stats.esn_hz = 1000.0 / stats.mean_step_ms if stats.mean_step_ms > 0 else 0.0

        if tracking_sq_err:
            stats.tracking_mse = float(np.mean(tracking_sq_err))
            stats.tracking_rmse = float(stats.tracking_mse ** 0.5)

        stats.grasp_frames = grasp_frames
        stats.task_metrics = metrics_rec.finalize()

        if cfg.record_video and video_frames:
            save_video_mp4(video_frames, video_path, source_hz=cfg.control_hz, target_fps=cfg.video_fps)
            stats.video_path = str(video_path)

        tm = stats.task_metrics
        logger.info(
            "Eval complete | RMSE=%.5f rad | cloth grasped %d/%d (%.1f%%) | "
            "max cloth jump=%.4f m | wipe path=%.3f m",
            stats.tracking_rmse,
            grasp_frames,
            sim_steps,
            100.0 * grasp_frames / max(sim_steps, 1),
            tm.max_cloth_jump_m if tm else 0.0,
            tm.wipe_path_length_m if tm else 0.0,
        )
        return stats


def print_eval_summary(stats: MuJoCoEvalStats, *, report_path: Path) -> None:
    print("\n" + "=" * 60)
    print("  Step 4 — MuJoCo Wipe-Table Evaluation (Dataset Oracle)")
    print("=" * 60)
    print(f"  Episode           : {stats.init_episode}")
    print(f"  Control mode      : {stats.control_mode}")
    print(f"  Steps             : {stats.steps:,}  ({stats.episode_loops} episode loop(s))")
    print(f"  Tracking RMSE     : {stats.tracking_rmse:.5f} rad  (MSE={stats.tracking_mse:.2e})")
    print(f"  Cloth grasped     : {stats.grasp_frames}/{stats.trajectory_steps} frames")
    if stats.task_metrics is not None:
        tm = stats.task_metrics
        print("  --- Wipe task benchmarks ---")
        print(f"  Max cloth jump    : {tm.max_cloth_jump_m:.4f} m  (mean {tm.mean_cloth_jump_m:.4f} m)")
        print(f"  Grasp proximity   : {tm.grasp_proximity_error_m:.4f} m  (success={tm.grasp_success})")
        print(f"  False attach      : {tm.false_attach_frames} frames")
        print(f"  Wipe path (XY)    : {tm.wipe_path_length_m:.3f} m")
        print(f"  Table contact     : {tm.table_contact_ratio:.1%} of grasp frames")
    print(f"  Mean step latency : {stats.mean_step_ms:.3f} ms  ({stats.esn_hz:.1f} Hz)")
    print(f"  Report JSON       : {report_path}")
    if stats.video_path:
        print(f"  Benchmark video   : {stats.video_path}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 4 MuJoCo wipe-table evaluation")
    parser.add_argument("--mjcf", type=str, default=None)
    parser.add_argument("--esn_checkpoint", type=str, default=str(models_path("esn_cuda_ridge")))
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--duration_s", type=float, default=MAX_DURATION_S,
                        help=f"Video/sim duration in seconds (max {MAX_DURATION_S:.0f}s)")
    parser.add_argument("--no_loop", action="store_true",
                        help="Do not loop episode when duration exceeds episode length")
    parser.add_argument("--control_hz", type=float, default=CONTROL_HZ)
    parser.add_argument("--vla_hz", type=float, default=2.0)
    parser.add_argument("--control_mode", choices=("kinematic", "pd"), default="kinematic")
    parser.add_argument("--no_video", action="store_true")
    parser.add_argument("--video_fps", type=float, default=VIDEO_FPS)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    mjcf = resolve_mjcf_path(args.mjcf)
    ckpt = resolve_esn_checkpoint(args.esn_checkpoint)
    meta = load_esn_checkpoint_metadata(ckpt)

    config = MuJoCoEvalConfig(
        mjcf_path=mjcf,
        esn_checkpoint=str(ckpt),
        init_episode=args.episode,
        duration_s=args.duration_s,
        control_hz=args.control_hz,
        vla_hz=args.vla_hz,
        control_mode=args.control_mode,
        record_video=not args.no_video,
        video_fps=args.video_fps,
        device=args.device,
        loop_episode=not args.no_loop,
    )
    stats = MuJoCoWipeEvaluator(config).run()

    report = {
        "step": 4,
        "task": "g1_wipe_table",
        "dataset_id": DATASET_ID,
        "init_episode": args.episode,
        "control_mode": args.control_mode,
        "esn_checkpoint": str(ckpt),
        "esn_step2_mse": meta.get("metrics", {}).get("mse"),
        "tracking_mse": stats.tracking_mse,
        "tracking_rmse": stats.tracking_rmse,
        "grasp_frames": stats.grasp_frames,
        "task_metrics": stats.task_metrics.to_dict() if stats.task_metrics else None,
        "trajectory_steps": stats.trajectory_steps,
        "steps": stats.steps,
        "mean_step_ms": stats.mean_step_ms,
        "video_path": stats.video_path,
    }
    report_path = RESULTS_DIR / "mujoco_eval_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print_eval_summary(stats, report_path=report_path)


if __name__ == "__main__":
    main()
