"""
============================================================
Step 3 — Unified Dynamical System Integration
Phase 3: Dual-process MuJoCo control (VLA @ ~2 Hz + ESN @ 100 Hz)
Research Plan: VLA + ESN for Real-Time Humanoid Control
Author: Osemudiamen Andrew Ihimekpen | PVAMU CREDIT Center
============================================================

Process A (VLA perception): UnifoLM-VLA-Base CUDA graph inference @ ~2 Hz
Process B (MuJoCo control):   Physics @ 100 Hz + CUDA sparse ESN readout

A lock-free ``multiprocessing.Array`` register shares the 29-DoF VLA target
token between processes, bypassing the Python GIL that caused 180 ms latency
spikes under the prior threading design.

For dataset-oracle replay, benchmark video, and cloth grasp visuals see Step 4:
  python3 -m src.step4_mujoco_evaluation

Usage (from research/):
  # Live UnifoLM (default) — required for timing / closed-loop claims
  python3 -m src.step3_dual_thread_mujoco --duration_s 10 --bridge esn
  python3 -m src.step3_dual_thread_mujoco --duration_s 10 --bridge zoh
  python3 -m src.step3_dual_thread_mujoco --duration_s 10 --bridge linear

  # Mock VLA — timing smoke test only (not a paper result)
  python3 -m src.step3_dual_thread_mujoco --mock --duration_s 5 --bridge esn
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import time
from ctypes import c_float, c_int, c_uint8
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch

# Headless EGL rendering on lab servers without DISPLAY.
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco

from src.paths import models_path, results_path
from src.step1_profile_unifolm_vla0 import G1_DOF, TARGET_HZ, UnifoLMVLAWrapper
from src.step2_esn_cuda_ridge import (
    BEST_CHECKPOINT_BASENAME,
    CONTROL_HZ,
    DATASET_ID,
    load_checkpoint,
)
from src.step3_control_baselines import online_linear_command, online_pid_command
from src.vla_ee_bridge import (
    EE_STATE_DIM,
    build_wipe_table_model,
    ee_action_to_joint_target,
    joints_to_ee_proprio,
    load_wipe_table_init_joints,
    resolve_robot_mjcf,
)

BridgeMode = Literal["esn", "zoh", "linear", "pid"]
ALL_BRIDGES: tuple[BridgeMode, ...] = ("esn", "zoh", "linear", "pid")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = results_path("step3_dual_thread")
DEFAULT_MJCF_CANDIDATES = (
    Path(os.environ.get("G1_MJCF", "")),
    Path.home() / "unitree_mujoco" / "unitree_robots" / "g1" / "g1_29dof.xml",
    Path("/home/aihimekpen/research/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.xml"),
)
DEFAULT_INSTRUCTION = "Wipe the table with the cloth."
DEFAULT_VLA_HZ = 2.0
DEFAULT_UNNORM_KEY = "g1_wipe_table"
VLA_ACTION_CHUNK = 25
VLA_EE_ACTION_DIM = EE_STATE_DIM
VLA_LOAD_TIMEOUT_S = 300.0  # max wait for UnifoLM-VLA-Base to load in child process
QPOS_ACTUATED_START = 7  # skip free-floating base (7-DoF) in qpos
VLA_IMAGE_SIZE = (224, 224)
VIDEO_SIZE = (480, 640)  # height × width (must fit MJCF offscreen framebuffer)
VIDEO_FPS = 60.0
MAX_DURATION_S = 30.0
MAX_STEP_MS_THRESHOLD = 10.0
WARMUP_STEPS = 5  # exclude CUDA / MuJoCo first-tick init from steady-state max


# ── Lock-free shared-memory registers ─────────────────────────
@dataclass
class SharedMemoryRegisters:
    """Inter-process registers backed by ``multiprocessing.Array`` (GIL-free)."""

    vla_token: Any
    vla_sequence: Any
    obs_image: Any
    obs_joints: Any
    obs_ee_proprio: Any
    obs_sequence: Any
    stop: Any
    vla_ticks: Any
    vla_ready: Any
    init_joints: Any

    @classmethod
    def create(cls, image_shape: Tuple[int, int, int] = (*VLA_IMAGE_SIZE, 3)) -> SharedMemoryRegisters:
        h, w, c = image_shape
        ctx = mp.get_context("spawn")
        return cls(
            vla_token=ctx.Array(c_float, G1_DOF, lock=False),
            vla_sequence=ctx.Value(c_int, 0, lock=False),
            obs_image=ctx.Array(c_uint8, h * w * c, lock=False),
            obs_joints=ctx.Array(c_float, G1_DOF, lock=False),
            obs_ee_proprio=ctx.Array(c_float, EE_STATE_DIM, lock=False),
            obs_sequence=ctx.Value(c_int, 0, lock=False),
            stop=ctx.Event(),
            vla_ticks=ctx.Value(c_int, 0, lock=False),
            vla_ready=ctx.Value(c_int, 0, lock=False),
            init_joints=ctx.Array(c_float, G1_DOF, lock=False),
        )

def _shared_buffer(shared: Any) -> Any:
    """Return the raw ctypes buffer for a ``multiprocessing.Array`` (parent or child)."""
    return shared.get_obj() if hasattr(shared, "get_obj") else shared


def write_init_joints(registers: SharedMemoryRegisters, joints_29d: np.ndarray) -> None:
    np.frombuffer(_shared_buffer(registers.init_joints), dtype=np.float32, count=G1_DOF)[:] = (
        np.asarray(joints_29d, dtype=np.float32).reshape(-1)
    )


def read_init_joints(registers: SharedMemoryRegisters) -> Optional[np.ndarray]:
    buf = np.frombuffer(_shared_buffer(registers.init_joints), dtype=np.float32, count=G1_DOF)
    if not np.any(buf):
        return None
    return buf.copy()


def write_vla_token(registers: SharedMemoryRegisters, token: np.ndarray) -> None:
    flat = np.asarray(token, dtype=np.float32).reshape(-1)
    if flat.size != G1_DOF:
        raise ValueError(f"Expected {G1_DOF}-D token, got {flat.size}")
    buf = _shared_buffer(registers.vla_token)
    np.frombuffer(buf, dtype=np.float32, count=G1_DOF)[:] = flat
    registers.vla_sequence.value += 1


def read_vla_token(registers: SharedMemoryRegisters) -> Tuple[np.ndarray, int]:
    seq_before = registers.vla_sequence.value
    buf = _shared_buffer(registers.vla_token)
    token = np.frombuffer(buf, dtype=np.float32, count=G1_DOF).copy()
    seq_after = registers.vla_sequence.value
    if seq_before != seq_after:
        token = np.frombuffer(buf, dtype=np.float32, count=G1_DOF).copy()
        seq_after = registers.vla_sequence.value
    return token, seq_after


def write_observation(
    registers: SharedMemoryRegisters,
    image: np.ndarray,
    joint_pos: np.ndarray,
    ee_proprio: np.ndarray,
    image_shape: Tuple[int, int, int],
) -> None:
    h, w, c = image_shape
    flat = np.ascontiguousarray(image, dtype=np.uint8).reshape(-1)
    buf = _shared_buffer(registers.obs_image)
    np.frombuffer(buf, dtype=np.uint8, count=h * w * c)[:] = flat
    np.frombuffer(_shared_buffer(registers.obs_joints), dtype=np.float32, count=G1_DOF)[:] = (
        np.asarray(joint_pos, dtype=np.float32).reshape(-1)
    )
    np.frombuffer(
        _shared_buffer(registers.obs_ee_proprio), dtype=np.float32, count=EE_STATE_DIM
    )[:] = np.asarray(ee_proprio, dtype=np.float32).reshape(-1)[:EE_STATE_DIM]
    registers.obs_sequence.value += 1


def read_observation(
    registers: SharedMemoryRegisters,
    image_shape: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    seq = registers.obs_sequence.value
    if seq <= 0:
        return (
            np.zeros(image_shape, dtype=np.uint8),
            np.zeros(G1_DOF, dtype=np.float32),
            np.zeros(EE_STATE_DIM, dtype=np.float32),
            False,
        )
    h, w, c = image_shape
    ibuf = _shared_buffer(registers.obs_image)
    image = np.frombuffer(ibuf, dtype=np.uint8, count=h * w * c).reshape(image_shape).copy()
    joints = np.frombuffer(_shared_buffer(registers.obs_joints), dtype=np.float32, count=G1_DOF).copy()
    ee = np.frombuffer(
        _shared_buffer(registers.obs_ee_proprio), dtype=np.float32, count=EE_STATE_DIM
    ).copy()
    return image, joints, ee, True


# ── MuJoCo G1 environment ─────────────────────────────────────
@dataclass
class G1MuJoCoEnv:
    """Unitree G1 MJCF wrapper with unified 29-DoF PD position tracking."""

    mjcf_path: Path
    control_hz: float = CONTROL_HZ
    image_size: Tuple[int, int] = VLA_IMAGE_SIZE
    video_size: Tuple[int, int] = VIDEO_SIZE
    enable_video_renderer: bool = True
    init_joints_29d: Optional[np.ndarray] = None
    pd_kp: float = 120.0
    pd_kd: float = 8.0
    wipe_table_scene: bool = False
    model: mujoco.MjModel = field(init=False, repr=False)
    data: mujoco.MjData = field(init=False, repr=False)
    renderer: mujoco.Renderer = field(init=False, repr=False)
    video_renderer: Optional[mujoco.Renderer] = field(init=False, repr=False, default=None)
    camera: mujoco.MjvCamera = field(init=False, repr=False)
    _initial_qpos: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.wipe_table_scene:
            robot_path = self.mjcf_path if self.mjcf_path.is_file() else resolve_robot_mjcf()
            self.model = build_wipe_table_model(robot_path)
        else:
            if not self.mjcf_path.is_file():
                raise FileNotFoundError(f"MJCF not found: {self.mjcf_path}")
            self.model = mujoco.MjModel.from_xml_path(str(self.mjcf_path))
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
        if self.wipe_table_scene or "wipe_table" in self.mjcf_path.name:
            self.camera.distance = 2.15
            self.camera.azimuth = 118.0
            self.camera.elevation = -22.0
            self.camera.lookat[:] = np.array([0.42, 0.0, 0.78])
        else:
            self.camera.distance = 2.8
            self.camera.azimuth = 135.0
            self.camera.elevation = -18.0
            self.camera.lookat[:] = np.array([0.0, 0.0, 0.85])

        self.reset()

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        q = self.data.qpos.copy()
        if self.init_joints_29d is not None:
            q[QPOS_ACTUATED_START : QPOS_ACTUATED_START + G1_DOF] = np.asarray(
                self.init_joints_29d, dtype=np.float64
            ).reshape(-1)
        elif self.model.nq >= QPOS_ACTUATED_START + G1_DOF:
            q[QPOS_ACTUATED_START + 3] = -0.25
            q[QPOS_ACTUATED_START + 9] = -0.25
            q[QPOS_ACTUATED_START + 0] = -0.15
            q[QPOS_ACTUATED_START + 6] = -0.15
        self.data.qpos[:] = q
        self._initial_qpos = self.data.qpos.copy()
        mujoco.mj_forward(self.model, self.data)

    def get_ee_proprio(self) -> np.ndarray:
        return joints_to_ee_proprio(
            self.model, self.data, self.get_joint_positions()
        )

    def get_joint_positions(self) -> np.ndarray:
        return self.data.qpos[QPOS_ACTUATED_START : QPOS_ACTUATED_START + G1_DOF].copy()

    def render_rgb(self) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render().copy()

    def render_video_frame(self) -> np.ndarray:
        if self.video_renderer is None:
            return self.render_rgb()
        self.video_renderer.update_scene(self.data, camera=self.camera)
        return self.video_renderer.render().copy()

    def apply_unified_control(self, ctrl_29d: np.ndarray) -> None:
        """PD tracking: ESN readout is a desired 29-DoF joint configuration."""
        q_des = np.asarray(ctrl_29d, dtype=np.float64).reshape(-1)
        if q_des.size != self.model.nu:
            raise ValueError(f"ctrl must be {self.model.nu}-D, got {q_des.size}")
        q = self.data.qpos[QPOS_ACTUATED_START : QPOS_ACTUATED_START + G1_DOF]
        qd = self.data.qvel[6 : 6 + G1_DOF]
        tau = self.pd_kp * (q_des - q) - self.pd_kd * qd
        for i in range(self.model.nu):
            jid = self.model.actuator_trnid[i, 0]
            if self.model.jnt_actfrclimited[jid]:
                lo, hi = self.model.jnt_actfrcrange[jid]
                tau[i] = np.clip(tau[i], lo, hi)
        self.data.ctrl[:] = tau

    def set_actuated_joints(self, joints_29d: np.ndarray) -> None:
        """Kinematic set of actuated joints (matches Step 2 open-loop replay)."""
        q = np.asarray(joints_29d, dtype=np.float64).reshape(-1)
        if q.size != G1_DOF:
            raise ValueError(f"Expected {G1_DOF} joints, got {q.size}")
        self.data.qpos[QPOS_ACTUATED_START : QPOS_ACTUATED_START + G1_DOF] = q
        self.data.qvel[6 : 6 + G1_DOF] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def step_kinematic(self, joints_29d: np.ndarray) -> None:
        """Advance one tick by directly applying joint targets (no PD torque)."""
        self.set_actuated_joints(joints_29d)
        self.data.qpos[:QPOS_ACTUATED_START] = self._initial_qpos[:QPOS_ACTUATED_START]
        self.data.qvel[:6] = 0.0

    def step_physics(self) -> None:
        mujoco.mj_step(self.model, self.data)
        self.data.qpos[:QPOS_ACTUATED_START] = self._initial_qpos[:QPOS_ACTUATED_START]
        self.data.qvel[:6] = 0.0

    def close(self) -> None:
        self.renderer.close()
        if self.video_renderer is not None:
            self.video_renderer.close()


def resolve_mjcf_path(user_path: Optional[str]) -> Path:
    return resolve_robot_mjcf(user_path)


def resolve_esn_checkpoint(user_path: Optional[str] = None) -> Path:
    """
    Resolve the Step 2 ESN ridge checkpoint (``esn_cuda_ridge_best.pth``).

    Trained on ``unitreerobotics/G1_Dex1_Wipe_Table`` episode 0 — same task as Phase 3.
    """
    if user_path:
        path = Path(user_path).expanduser().resolve()
        if path.is_dir():
            for name in (f"{BEST_CHECKPOINT_BASENAME}.pth", f"{BEST_CHECKPOINT_BASENAME}.pt"):
                candidate = path / name
                if candidate.is_file():
                    return candidate
            raise FileNotFoundError(f"No Step 2 best checkpoint in directory: {path}")
        if path.is_file():
            return path
        raise FileNotFoundError(f"ESN checkpoint not found: {path}")

    ckpt_dir = models_path("esn_cuda_ridge")
    for name in (f"{BEST_CHECKPOINT_BASENAME}.pth", f"{BEST_CHECKPOINT_BASENAME}.pt"):
        candidate = ckpt_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Step 2 ESN checkpoint not found under {ckpt_dir}. "
        "Run step2_esn_cuda_ridge.ipynb first."
    )


def load_esn_checkpoint_metadata(checkpoint: Path) -> Dict[str, Any]:
    """Read Step 2 training metrics from ``config.json`` next to the checkpoint."""
    meta_path = checkpoint.parent / "config.json"
    if not meta_path.is_file():
        return {}
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def save_video_mp4(
    frames: List[np.ndarray],
    path: Path,
    *,
    source_hz: float,
    target_fps: float = VIDEO_FPS,
) -> None:
    """Compile RGB frames into an H.264-compatible MP4 at ``target_fps``.

    Falls back to OpenCV, then Pillow GIF if imageio/ffmpeg is unavailable.
    """
    if not frames:
        logger.warning("No frames captured — skipping video export.")
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_out = max(1, int(round(len(frames) * target_fps / source_hz)))
    indices = np.linspace(0, len(frames) - 1, n_out, dtype=int)
    selected = [np.asarray(frames[i], dtype=np.uint8) for i in indices]

    h, w = selected[0].shape[:2]
    try:
        import imageio.v2 as imageio

        imageio.mimsave(
            str(path),
            selected,
            fps=target_fps,
            codec="libx264",
            pixelformat="yuv420p",
        )
        logger.info("Video saved via imageio: %s (%d frames @ %.0f fps)", path, len(selected), target_fps)
        return
    except Exception as exc:
        logger.warning("imageio H.264 export unavailable: %s", exc)

    try:
        import cv2
    except ModuleNotFoundError:
        cv2 = None  # type: ignore[assignment]

    if cv2 is not None:
        writer = None
        codec_used = "mp4v"
        for fourcc_str in ("avc1", "H264", "mp4v"):
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            candidate = cv2.VideoWriter(str(path), fourcc, target_fps, (w, h))
            if candidate.isOpened():
                writer = candidate
                codec_used = fourcc_str
                break
            candidate.release()
        if writer is not None:
            for frame in selected:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(bgr)
            writer.release()
            logger.info(
                "Video saved via OpenCV (%s): %s (%d frames @ %.0f fps)",
                codec_used,
                path,
                len(selected),
                target_fps,
            )
            return
        logger.warning("cv2.VideoWriter failed to open: %s", path)

    # Last resort: animated GIF (plays in Jupyter without OpenCV/ffmpeg).
    try:
        from PIL import Image

        gif_path = path.with_suffix(".gif")
        imgs = [Image.fromarray(frame) for frame in selected]
        duration_ms = max(1, int(round(1000.0 / max(target_fps, 1e-6))))
        imgs[0].save(
            gif_path,
            save_all=True,
            append_images=imgs[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
        )
        logger.warning(
            "Saved GIF fallback (install opencv-python-headless imageio imageio-ffmpeg for MP4): %s",
            gif_path,
        )
        return
    except Exception as exc:
        raise RuntimeError(
            "Video export needs imageio[+ffmpeg], opencv-python-headless, or Pillow. "
            "Install with: pip install opencv-python-headless imageio imageio-ffmpeg"
        ) from exc


# ── Process A: VLA perception worker ──────────────────────────
def _vla_perception_worker(
    registers: SharedMemoryRegisters,
    *,
    mock: bool,
    instruction: str,
    vla_hz: float,
    device_str: str,
    image_shape: Tuple[int, int, int],
    mjcf_path: str,
    unnorm_key: str = DEFAULT_UNNORM_KEY,
    use_wipe_table_scene: bool = True,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA required for VLA perception process.")

    if use_wipe_table_scene:
        ik_model = build_wipe_table_model(resolve_robot_mjcf(mjcf_path))
    else:
        ik_model = mujoco.MjModel.from_xml_path(mjcf_path)
    ik_data = mujoco.MjData(ik_model)

    vla_model = UnifoLMVLAWrapper(
        model_id="__mock__" if mock else "unitreerobotics/UnifoLM-VLA-Base",
        allow_mock_fallback=mock,
        use_fp16=True,
        use_cuda_graph=not mock,
        unnorm_key=unnorm_key,
    )

    if not mock:
        deadline = time.perf_counter() + VLA_LOAD_TIMEOUT_S
        while not registers.stop.is_set():
            image, joint_pos, ee_proprio, ready = read_observation(registers, image_shape)
            if ready:
                vla_model.warmup_cuda_graph(image, instruction, ee_proprio)
                break
            if time.perf_counter() > deadline:
                logger.warning("VLA warmup timed out — proceeding without CUDA graph warmup.")
                break
            time.sleep(0.01)
    else:
        time.sleep(0.05)

    registers.vla_ready.value = 1
    logger.info(
        "VLA perception process ready (mock=%s, unnorm_key=%s).",
        mock,
        unnorm_key,
    )

    period_s = 1.0 / vla_hz
    chunk_period_s = 1.0 / max(vla_hz * VLA_ACTION_CHUNK, 1e-6)
    use_nvtx = device.type == "cuda" and hasattr(torch.cuda, "nvtx")
    stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None

    logger.info(
        "VLA perception process started @ %.1f Hz (action chunk %d @ %.1f Hz)",
        vla_hz,
        VLA_ACTION_CHUNK,
        vla_hz * VLA_ACTION_CHUNK,
    )
    while not registers.stop.is_set():
        t0 = time.perf_counter()
        image, joint_pos, ee_proprio, ready = read_observation(registers, image_shape)
        if not ready:
            time.sleep(0.001)
            continue

        if use_nvtx:
            torch.cuda.nvtx.range_push("VLA_Perception_Process")
        try:
            if stream is not None:
                with torch.cuda.stream(stream):
                    action_gpu, _, _ = vla_model.infer_gpu(
                        image,
                        instruction,
                        joint_state=ee_proprio,
                    )
                stream.synchronize()
            else:
                action_gpu, _, _ = vla_model.infer_gpu(
                    image,
                    instruction,
                    joint_state=ee_proprio,
                )

            if mock:
                token = action_gpu.detach().float().cpu().numpy().reshape(-1)[:G1_DOF]
                write_vla_token(registers, token)
                registers.vla_ticks.value += 1
            else:
                ee_chunk = _reshape_vla_action_chunk(action_gpu)
                for ee_action in ee_chunk:
                    if registers.stop.is_set():
                        break
                    _, joint_pos, _, ready = read_observation(registers, image_shape)
                    token = ee_action_to_joint_target(
                        ik_model, ik_data, ee_action, joint_pos
                    )
                    write_vla_token(registers, token)
                    registers.vla_ticks.value += 1
                    sleep_s = chunk_period_s - (time.perf_counter() - t0)
                    t_step = time.perf_counter()
                    if sleep_s > 0:
                        time.sleep(sleep_s)
                    t0 = t_step
        finally:
            if use_nvtx:
                torch.cuda.nvtx.range_pop()

        if mock:
            elapsed = time.perf_counter() - t0
            sleep_s = period_s - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

    logger.info("VLA perception process stopped (%d ticks).", registers.vla_ticks.value)


# ── Process B: 100 Hz ESN + physics + video ───────────────────
@dataclass
class ControlLoopStats:
    steps: int = 0
    esn_hz: float = 0.0
    vla_ticks: int = 0
    vla_seq_final: int = 0
    mean_step_ms: float = 0.0
    max_step_ms: float = 0.0
    max_step_ms_steady: float = 0.0
    p99_step_ms: float = 0.0
    video_path: Optional[str] = None
    gil_bypass_ok: bool = False
    bridge: str = "esn"


def _warmup_esn_reservoir(
    esn: torch.nn.Module,
    ground_truth: np.ndarray,
    vla_targets: np.ndarray,
    device: torch.device,
) -> None:
    """Drive the reservoir on GT data (matches Step 2 washout before readout)."""
    washout = int(getattr(esn.cfg, "washout", 50))
    esn.reset_state()
    n = min(washout, ground_truth.shape[0])
    with torch.no_grad():
        for t in range(n):
            joint_gpu = torch.from_numpy(ground_truth[t]).to(device=device, dtype=torch.float32)
            vla_gpu = torch.from_numpy(vla_targets[t]).to(device=device, dtype=torch.float32)
            esn.update_vla_target(vla_gpu)
            esn.step_proprio(joint_gpu)
    logger.info("ESN reservoir warmed up on %d dataset ticks (washout=%d).", n, washout)


def _reshape_vla_action_chunk(action_gpu: torch.Tensor) -> np.ndarray:
    """Return (chunk_len, EE_DIM) denormalized EE actions from the VLA head."""
    flat = action_gpu.detach().float().cpu().numpy().reshape(-1)
    chunk_len = VLA_ACTION_CHUNK
    need = chunk_len * VLA_EE_ACTION_DIM
    if flat.size < need:
        reps = int(np.ceil(need / max(flat.size, 1)))
        flat = np.tile(flat, reps)[:need]
    return flat[:need].reshape(chunk_len, VLA_EE_ACTION_DIM)


def _finalize_step_stats(step_times: list[float], stats: ControlLoopStats) -> None:
    if not step_times:
        return
    arr = np.asarray(step_times, dtype=np.float64)
    stats.mean_step_ms = float(arr.mean())
    stats.max_step_ms = float(arr.max())
    steady = arr[WARMUP_STEPS:] if arr.size > WARMUP_STEPS else arr
    stats.max_step_ms_steady = float(steady.max()) if steady.size else stats.max_step_ms
    stats.p99_step_ms = float(np.percentile(arr, 99))
    stats.esn_hz = 1000.0 / stats.mean_step_ms if stats.mean_step_ms > 0 else 0.0
    stats.gil_bypass_ok = stats.max_step_ms_steady < MAX_STEP_MS_THRESHOLD


def _mujoco_control_worker(
    registers: SharedMemoryRegisters,
    result_queue: mp.Queue,
    *,
    mjcf_path: str,
    esn_checkpoint: str,
    duration_s: float,
    control_hz: float,
    device_str: str,
    mock: bool,
    profile: bool,
    profile_steps: int,
    record_video: bool,
    video_path: str,
    video_fps: float,
    image_shape: Tuple[int, int, int],
    init_episode: int = 0,
    use_wipe_table_scene: bool = True,
    bridge: str = "esn",
    vla_hz: float = DEFAULT_VLA_HZ,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if bridge not in ALL_BRIDGES:
        result_queue.put(RuntimeError(f"Unknown bridge mode: {bridge}"))
        return

    stats = ControlLoopStats(bridge=bridge)
    step_times: list[float] = []
    video_frames: List[np.ndarray] = []

    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA required for MuJoCo control process.")

    wipe_scene = use_wipe_table_scene
    init_joints = read_init_joints(registers) if wipe_scene else None

    env = G1MuJoCoEnv(
        mjcf_path=Path(mjcf_path),
        control_hz=control_hz,
        enable_video_renderer=record_video,
        init_joints_29d=init_joints,
        wipe_table_scene=wipe_scene,
    )
    esn = None
    if bridge == "esn":
        esn = load_checkpoint(esn_checkpoint, device=device)
        esn.eval()

    init_joints = env.get_joint_positions()
    write_vla_token(registers, init_joints)

    dt = 1.0 / control_hz
    hold_ticks = max(1, int(round(control_hz / max(vla_hz, 1e-6))))
    joint_gpu = torch.zeros(G1_DOF, device=device, dtype=torch.float32)
    vla_gpu = torch.zeros(G1_DOF, device=device, dtype=torch.float32)
    use_nvtx = device.type == "cuda" and hasattr(torch.cuda, "nvtx")

    prev_token = init_joints.astype(np.float32).copy()
    curr_token = init_joints.astype(np.float32).copy()
    pid_q = init_joints.astype(np.float32).copy()
    last_seq = 0
    ticks_since_update = 0

    def _publish_observation() -> Tuple[np.ndarray, np.ndarray]:
        image = env.render_rgb()
        joint_pos = env.get_joint_positions()
        ee_proprio = env.get_ee_proprio()
        write_observation(registers, image, joint_pos, ee_proprio, image_shape)
        return image, joint_pos

    wait_timeout_s = VLA_LOAD_TIMEOUT_S if not mock else 30.0
    if registers.vla_ready.value == 0:
        logger.info(
            "Publishing observations while VLA loads (timeout %.0fs) ...",
            wait_timeout_s,
        )
        bootstrap_deadline = time.perf_counter() + wait_timeout_s
        while registers.vla_ready.value == 0 and time.perf_counter() < bootstrap_deadline:
            _publish_observation()
            time.sleep(dt)

    if registers.vla_ready.value:
        logger.info(
            "VLA ready — starting %.1fs timed control loop (bridge=%s).",
            duration_s,
            bridge,
        )
    else:
        logger.warning(
            "VLA not ready after %.0fs — running with seeded token only.",
            wait_timeout_s,
        )

    profiler = None
    if profile and device.type == "cuda":
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            with_stack=False,
        )
        profiler.__enter__()

    t_end = time.perf_counter() + duration_s
    next_tick = time.perf_counter()

    logger.info("MuJoCo control process started @ %.0f Hz (bridge=%s)", control_hz, bridge)
    try:
        while time.perf_counter() < t_end and not registers.stop.is_set():
            tick_start = time.perf_counter()

            if use_nvtx:
                torch.cuda.nvtx.range_push("Control_Process")

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
                joint_gpu.copy_(
                    torch.from_numpy(joint_pos).to(device=device, dtype=torch.float32)
                )
                vla_gpu.copy_(
                    torch.from_numpy(curr_token).to(device=device, dtype=torch.float32)
                )
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
            else:  # pid
                pid_q = online_pid_command(q=pid_q, target=curr_token, dt=dt)
                cmd_np = pid_q

            env.apply_unified_control(cmd_np)
            env.step_physics()

            if record_video:
                video_frames.append(env.render_video_frame())

            if use_nvtx:
                torch.cuda.nvtx.range_pop()

            step_ms = (time.perf_counter() - tick_start) * 1000.0
            step_times.append(step_ms)
            stats.steps += 1

            if profiler is not None and mock and stats.steps >= profile_steps:
                break

            next_tick += dt
            sleep_s = next_tick - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.perf_counter()

    finally:
        registers.stop.set()
        if profiler is not None:
            profiler.__exit__(None, None, None)
            trace_path = RESULTS_DIR / "step3_dual_thread_trace.json"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            profiler.export_chrome_trace(str(trace_path))
            logger.info("PyTorch profiler trace saved: %s", trace_path)
        env.close()

    if step_times:
        _finalize_step_stats(step_times, stats)

    stats.vla_ticks = int(registers.vla_ticks.value)
    _, stats.vla_seq_final = read_vla_token(registers)

    if record_video and video_frames:
        out_path = Path(video_path)
        save_video_mp4(
            video_frames,
            out_path,
            source_hz=control_hz,
            target_fps=video_fps,
        )
        stats.video_path = str(out_path)

    result_queue.put(stats)



def run_esn_physics_loop(
    env: G1MuJoCoEnv,
    esn: torch.nn.Module,
    registers: SharedMemoryRegisters,
    *,
    duration_s: float,
    control_hz: float,
    device: torch.device,
    profile: bool = False,
    profile_steps: int = 500,
    record_video: bool = True,
    video_path: Optional[Path] = None,
    video_fps: float = VIDEO_FPS,
) -> ControlLoopStats:
    """
    In-process 100 Hz loop (testing helper).

    Production runs use ``DualProcessController`` so VLA inference stays GIL-free.
    """
    image_shape = (*env.image_size, 3)
    stats = ControlLoopStats()
    step_times: list[float] = []
    video_frames: List[np.ndarray] = []

    joint_gpu = torch.zeros(G1_DOF, device=device, dtype=torch.float32)
    vla_gpu = torch.zeros(G1_DOF, device=device, dtype=torch.float32)
    dt = 1.0 / control_hz
    out_video = video_path or (RESULTS_DIR / "table_wipe_benchmark.mp4")

    t_end = time.perf_counter() + duration_s
    next_tick = time.perf_counter()

    while time.perf_counter() < t_end:
        tick_start = time.perf_counter()
        image = env.render_rgb()
        joint_pos = env.get_joint_positions()
        ee_proprio = env.get_ee_proprio()
        write_observation(registers, image, joint_pos, ee_proprio, image_shape)

        vla_token, _ = read_vla_token(registers)
        joint_gpu.copy_(torch.from_numpy(joint_pos).to(device=device, dtype=torch.float32))
        vla_gpu.copy_(torch.from_numpy(vla_token).to(device=device, dtype=torch.float32))

        esn.update_vla_target(vla_gpu)
        cmd_gpu = esn.step_proprio(joint_gpu)
        env.apply_unified_control(cmd_gpu.detach().cpu().numpy())
        env.step_physics()

        if record_video:
            video_frames.append(env.render_video_frame())

        step_ms = (time.perf_counter() - tick_start) * 1000.0
        step_times.append(step_ms)
        stats.steps += 1

        next_tick += dt
        sleep_s = next_tick - time.perf_counter()
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            next_tick = time.perf_counter()

        if profile and stats.steps >= profile_steps:
            break

    if step_times:
        _finalize_step_stats(step_times, stats)

    if record_video and video_frames:
        save_video_mp4(video_frames, out_video, source_hz=control_hz, target_fps=video_fps)
        stats.video_path = str(out_video)

    return stats


@dataclass
class DualProcessConfig:
    mjcf_path: Path
    esn_checkpoint: str
    mock: bool = False
    duration_s: float = 10.0
    control_hz: float = TARGET_HZ
    vla_hz: float = DEFAULT_VLA_HZ
    instruction: str = DEFAULT_INSTRUCTION
    device: str = "cuda"
    profile: bool = False
    profile_steps: int = 500
    record_video: bool = False
    video_path: Optional[Path] = None
    video_fps: float = VIDEO_FPS
    unnorm_key: str = DEFAULT_UNNORM_KEY
    init_episode: int = 0
    init_dataset_id: str = DATASET_ID
    use_wipe_table_scene: bool = True
    bridge: BridgeMode = "esn"
    task_id: str = "wipe_table"


class DualProcessController:
    """Orchestrates Process A (VLA) and Process B (MuJoCo + ESN)."""

    def __init__(self, config: DualProcessConfig):
        self.config = config
        self.registers = SharedMemoryRegisters.create()
        self._vla_process: Optional[mp.Process] = None
        self._control_process: Optional[mp.Process] = None

    def run(self) -> ControlLoopStats:
        cfg = self.config
        if cfg.duration_s > MAX_DURATION_S:
            raise ValueError(
                f"duration_s={cfg.duration_s} exceeds MAX_DURATION_S={MAX_DURATION_S}"
            )
        if not torch.cuda.is_available() and cfg.device.startswith("cuda"):
            raise RuntimeError("CUDA required (target: Tesla V100).")

        ctx = mp.get_context("spawn")
        result_queue: mp.Queue = ctx.Queue()
        image_shape = (*VLA_IMAGE_SIZE, 3)
        video_path = cfg.video_path or (
            RESULTS_DIR / f"table_wipe_{cfg.bridge}_{'mock' if cfg.mock else 'live'}.mp4"
        )
        video_path.parent.mkdir(parents=True, exist_ok=True)

        if cfg.use_wipe_table_scene or cfg.init_dataset_id:
            try:
                init_pose = load_wipe_table_init_joints(
                    cfg.init_episode, dataset_id=cfg.init_dataset_id
                )
                write_init_joints(self.registers, init_pose)
                logger.info(
                    "Seeded init pose from %s episode %d.",
                    cfg.init_dataset_id,
                    cfg.init_episode,
                )
            except Exception as exc:
                logger.warning("Could not load dataset init pose in parent: %s", exc)

        self._vla_process = ctx.Process(
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
                "use_wipe_table_scene": cfg.use_wipe_table_scene,
            },
            daemon=False,
        )
        self._control_process = ctx.Process(
            target=_mujoco_control_worker,
            name="MuJoCo_Control_Process",
            args=(self.registers, result_queue),
            kwargs={
                "mjcf_path": str(cfg.mjcf_path.resolve()),
                "esn_checkpoint": cfg.esn_checkpoint,
                "duration_s": cfg.duration_s,
                "control_hz": cfg.control_hz,
                "device_str": cfg.device,
                "mock": cfg.mock,
                "profile": cfg.profile,
                "profile_steps": cfg.profile_steps,
                "record_video": cfg.record_video,
                "video_path": str(video_path),
                "video_fps": cfg.video_fps,
                "image_shape": image_shape,
                "init_episode": cfg.init_episode,
                "use_wipe_table_scene": cfg.use_wipe_table_scene,
                "bridge": cfg.bridge,
                "vla_hz": cfg.vla_hz,
            },
            daemon=False,
        )

        logger.info(
            "Starting dual-process control | bridge=%s | control=%.0f Hz | VLA=%.1f Hz | "
            "mock=%s | duration=%.1fs",
            cfg.bridge,
            cfg.control_hz,
            cfg.vla_hz,
            cfg.mock,
            cfg.duration_s,
        )
        self._vla_process.start()
        self._control_process.start()

        queue_timeout_s = cfg.duration_s + VLA_LOAD_TIMEOUT_S + 120.0
        try:
            stats = result_queue.get(timeout=queue_timeout_s)
        except Exception as exc:
            raise RuntimeError(
                f"Control process did not return stats (exit={self._control_process.exitcode}): {exc}"
            ) from exc
        finally:
            self.registers.stop.set()
            if self._control_process.is_alive():
                self._control_process.join(timeout=10.0)
            if self._vla_process.is_alive():
                self._vla_process.join(timeout=10.0)

        if not isinstance(stats, ControlLoopStats):
            raise RuntimeError(f"Control process failed: {stats}")

        return stats


# Backward-compatible aliases (threading API removed — use DualProcessController).
SharedVLATokenRegister = SharedMemoryRegisters
SharedObservationRegister = SharedMemoryRegisters
VLABackgroundWorker = DualProcessController


def print_run_summary(
    stats: ControlLoopStats,
    *,
    control_hz: float,
    vla_hz: float,
    report_path: Path,
    profile: bool,
    bridge: str = "esn",
    mock: bool = False,
) -> None:
    gil_status = "RESOLVED" if stats.gil_bypass_ok else "NOT RESOLVED"
    vla_mode = "mock" if mock else "live UnifoLM"
    print("\n" + "=" * 60)
    print("  Phase 3 — Dual-Process MuJoCo Control (GIL-Free)")
    print("=" * 60)
    print(f"  Bridge mode      : {bridge}")
    print(f"  VLA mode         : {vla_mode}")
    print(f"  Physics steps     : {stats.steps:,}")
    print(f"  Mean step latency : {stats.mean_step_ms:.3f} ms  ({stats.esn_hz:.1f} Hz)")
    print(f"  Max step latency  : {stats.max_step_ms:.3f} ms  (steady-state: {stats.max_step_ms_steady:.3f} ms)")
    print(f"  P99 step latency  : {stats.p99_step_ms:.3f} ms")
    print(f"  GIL bypass        : {gil_status} (steady max < {MAX_STEP_MS_THRESHOLD:.0f} ms)")
    print(f"  VLA perception    : {stats.vla_ticks} ticks @ ~{vla_hz:.0f} Hz")
    print(f"  VLA register seq  : {stats.vla_seq_final}")
    print(f"  Report JSON       : {report_path}")
    if stats.video_path:
        print(f"  Benchmark video   : {stats.video_path}")
    if profile:
        print(f"  Profiler trace    : {RESULTS_DIR / 'step3_dual_thread_trace.json'}")
    print("=" * 60)
    if stats.gil_bypass_ok:
        print(
            f"  VERIFIED: steady-state max step latency {stats.max_step_ms_steady:.3f} ms is well "
            f"below {MAX_STEP_MS_THRESHOLD:.0f} ms — Python GIL bottleneck eliminated."
        )
    else:
        print(
            f"  WARNING: steady-state max step latency {stats.max_step_ms_steady:.3f} ms exceeds "
            f"{MAX_STEP_MS_THRESHOLD:.0f} ms threshold."
        )
    print("=" * 60)


# ── Main ──────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3 dual-process MuJoCo control (VLA + bridge, GIL-free)",
    )
    from src.unifolm_tasks import (
        DEFAULT_TASK_ID,
        add_task_arg,
        get_task,
        maybe_print_tasks_and_exit,
        resolve_unnorm_key,
    )

    add_task_arg(parser, default=DEFAULT_TASK_ID)
    parser.add_argument("--mjcf", type=str, default=None, help="Path to G1 MJCF XML")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Mock VLA inference (timing smoke test only; omit for live UnifoLM)",
    )
    parser.add_argument(
        "--bridge",
        choices=list(ALL_BRIDGES),
        default="esn",
        help="100 Hz bridge: esn (ours), zoh, linear, or pid",
    )
    parser.add_argument("--record_video", action="store_true", help="Export benchmark MP4 (see Step 4 for full eval)")
    parser.add_argument("--episode", type=int, default=0, help="Dataset episode for init pose seeding")
    parser.add_argument("--duration_s", type=float, default=MAX_DURATION_S,
                        help=f"Sim duration after VLA load (max {MAX_DURATION_S:.0f}s)")
    parser.add_argument("--control_hz", type=float, default=TARGET_HZ)
    parser.add_argument("--vla_hz", type=float, default=DEFAULT_VLA_HZ)
    parser.add_argument("--instruction", type=str, default=None, help="Override task instruction")
    parser.add_argument(
        "--unnorm_key",
        type=str,
        default=None,
        help="Override UnifoLM norm key (default: from --task)",
    )
    parser.add_argument(
        "--esn_checkpoint",
        type=str,
        default=None,
        help="ESN checkpoint dir (default: models/esn_cuda_ridge[_<task>])",
    )
    parser.add_argument("--profile", action="store_true", help="Export PyTorch Chrome trace")
    parser.add_argument("--profile_steps", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_video", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--video_fps", type=float, default=VIDEO_FPS)
    args = parser.parse_args()
    maybe_print_tasks_and_exit(args)
    from src.unifolm_tasks import esn_checkpoint_basename

    task = get_task(args.task)
    unnorm_key = resolve_unnorm_key(task, args.unnorm_key)
    instruction = args.instruction or task.instruction
    esn_default = models_path(esn_checkpoint_basename(task.id))

    mjcf_path = resolve_mjcf_path(args.mjcf)
    if args.duration_s > MAX_DURATION_S:
        raise ValueError(f"--duration_s must be <= {MAX_DURATION_S}")

    esn_ckpt = Path(args.esn_checkpoint or esn_default)
    esn_meta: Dict[str, Any] = {}
    logger.info("MJCF: %s", mjcf_path)
    logger.info(
        "Task=%s | unnorm=%s | Bridge=%s | VLA=%s",
        task.id,
        unnorm_key,
        args.bridge,
        "mock" if args.mock else "live UnifoLM",
    )
    if args.bridge == "esn":
        esn_ckpt = resolve_esn_checkpoint(args.esn_checkpoint)
        logger.info("ESN checkpoint (Step 2): %s", esn_ckpt)
        esn_meta = load_esn_checkpoint_metadata(esn_ckpt)
        if esn_meta.get("metrics"):
            m = esn_meta["metrics"]
            logger.info(
                "  Step 2 metrics: MSE=%.2e jerk=%.2e α=%.2f λ=%.1e dataset=%s",
                m.get("mse", float("nan")),
                m.get("jerk", float("nan")),
                m.get("leaky_rate", float("nan")),
                m.get("ridge_alpha", float("nan")),
                esn_meta.get("dataset_id", DATASET_ID),
            )
    else:
        logger.info("Bridge=%s — ESN checkpoint not required.", args.bridge)
    if torch.cuda.is_available():
        logger.info("Device: %s (%s)", args.device, torch.cuda.get_device_name(torch.device(args.device)))

    config = DualProcessConfig(
        mjcf_path=mjcf_path,
        esn_checkpoint=str(esn_ckpt),
        mock=args.mock,
        duration_s=args.duration_s,
        control_hz=args.control_hz,
        vla_hz=args.vla_hz,
        instruction=instruction,
        device=args.device,
        profile=args.profile,
        profile_steps=args.profile_steps,
        record_video=args.record_video and not args.no_video,
        video_fps=args.video_fps,
        unnorm_key=unnorm_key,
        init_episode=args.episode,
        init_dataset_id=task.primary_dataset_id,
        use_wipe_table_scene=bool(task.supports_wipe_cloth_metrics),
        bridge=args.bridge,
        task_id=task.id,
    )
    controller = DualProcessController(config)
    stats = controller.run()

    vla_tag = "mock" if config.mock else "live"
    report: Dict[str, Any] = {
        "architecture": "multiprocessing",
        "task": task.id,
        "unnorm_key": unnorm_key,
        "dataset_id": task.primary_dataset_id,
        "bridge": config.bridge,
        "mjcf": str(mjcf_path),
        "control_hz_target": args.control_hz,
        "vla_hz_target": args.vla_hz,
        "duration_s": args.duration_s,
        "mock_vla": config.mock,
        "init_episode": config.init_episode,
        "esn_checkpoint": str(esn_ckpt) if config.bridge == "esn" else None,
        "esn_dataset_id": esn_meta.get("dataset_id", DATASET_ID) if config.bridge == "esn" else None,
        "esn_step2_mse": esn_meta.get("metrics", {}).get("mse") if config.bridge == "esn" else None,
        "esn_step2_jerk": esn_meta.get("metrics", {}).get("jerk") if config.bridge == "esn" else None,
        "steps": stats.steps,
        "mean_step_ms": stats.mean_step_ms,
        "max_step_ms": stats.max_step_ms,
        "max_step_ms_steady": stats.max_step_ms_steady,
        "p99_step_ms": stats.p99_step_ms,
        "achieved_control_hz": stats.esn_hz,
        "achieved_esn_hz": stats.esn_hz if config.bridge == "esn" else None,
        "vla_ticks": stats.vla_ticks,
        "vla_register_sequence": stats.vla_seq_final,
        "gil_bypass_ok": stats.gil_bypass_ok,
        "max_step_ms_threshold": MAX_STEP_MS_THRESHOLD,
        "video_path": stats.video_path,
        "device": args.device,
    }
    report_name = f"dual_thread_report_{config.bridge}_{vla_tag}.json"
    report_path = RESULTS_DIR / report_name
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    # Keep legacy filename pointing at the latest ESN live/mock run for notebooks.
    if config.bridge == "esn":
        legacy = RESULTS_DIR / "dual_thread_report.json"
        with open(legacy, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    print_run_summary(
        stats,
        control_hz=args.control_hz,
        vla_hz=args.vla_hz,
        report_path=report_path,
        profile=args.profile,
        bridge=config.bridge,
        mock=config.mock,
    )
    print("  Live UnifoLM (paper timing):")
    print("    python3 -m src.step3_dual_thread_mujoco --bridge esn --duration_s 10")
    print("  Baselines in the same loop:")
    print("    python3 -m src.step3_dual_thread_mujoco --bridge zoh --duration_s 10")
    print("    python3 -m src.step3_dual_thread_mujoco --bridge linear --duration_s 10")
    print("    python3 -m src.step3_dual_thread_mujoco --bridge pid --duration_s 10")
    print("  Offline ZOH/linear/PID table (no GPU VLA needed):")
    print("    python3 -m src.step3_control_baselines --all --episode 0")
    print("  Or run all sim comparisons via:")
    print("    python3 -m src.step3_sim_comparison --episode 0 --duration_s 10")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
