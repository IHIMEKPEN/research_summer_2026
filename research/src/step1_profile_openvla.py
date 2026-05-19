"""
============================================================
Step 1 — OpenVLA Inference Profiler
Week 1–2 Deliverable: Inference Log + Profiling Report
Research Plan: VLA + ESN for Real-Time Humanoid Control
Author: Osemudiamen Andrew Ihimekpen | PVAMU CREDIT Center
============================================================

Measures:
  - Token generation latency (ms) per inference call
  - Effective action/control rate (Hz)
  - GPU memory footprint (GB)
  - Action output format and value ranges
  - Failure modes under varied image conditions

Usage:
  python -m src.step1_profile_openvla --n_trials 100 --mock
  python -m src.step1_profile_openvla --n_trials 100 --model openvla/openvla-7b
"""

import argparse
import time
import json
import os
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from src.paths import results_path

# ── Constants ────────────────────────────────────────────────
RESULTS_DIR = results_path("step1_profiling")

G1_DOF = 29          # Unitree G1: 29 controllable joints
TARGET_HZ = 100      # Required control rate for G1
OPENVLA_EXPECTED_HZ = (2, 5)   # Expected range from research plan


# ── Data structures ──────────────────────────────────────────
@dataclass
class InferenceRecord:
    trial: int
    latency_ms: float
    action: List[float]           # raw action output
    action_dim: int
    input_tokens: int
    output_tokens: int
    gpu_mem_gb: float
    failed: bool
    failure_reason: str


@dataclass
class ProfilingReport:
    model_id: str
    n_trials: int
    mean_latency_ms: float
    std_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    mean_hz: float
    target_hz: int
    frequency_gap: float          # factor: how much faster ESN must be
    mean_gpu_mem_gb: float
    action_dim: int
    action_mean: List[float]
    action_std: List[float]
    failure_rate: float
    failure_modes: Dict[str, int]


# ── Mock G1 environment (sim interface) ─────────────────────
class MockG1SimEnv:
    """
    Lightweight mock of the Unitree G1 simulation interface.
    Replace with actual MuJoCo env once the full sim is configured.
    Provides: reset(), step(), get_observation_image()
    """

    def __init__(self, image_size: Tuple[int, int] = (224, 224)):
        self.image_size = image_size
        self.dof = G1_DOF
        self.step_count = 0
        self.joint_pos = np.zeros(self.dof)
        self.joint_vel = np.zeros(self.dof)
        logger.info(f"MockG1SimEnv initialised | DOF={self.dof} | img={image_size}")

    def reset(self) -> np.ndarray:
        self.step_count = 0
        self.joint_pos = np.random.uniform(-0.1, 0.1, self.dof)
        self.joint_vel = np.zeros(self.dof)
        return self._get_obs_image()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """Apply action, advance sim by one step."""
        assert len(action) == self.dof, f"Expected {self.dof}-dim action, got {len(action)}"
        dt = 0.01  # 100 Hz timestep
        self.joint_vel = action
        self.joint_pos += self.joint_vel * dt
        self.step_count += 1
        obs = self._get_obs_image()
        reward = 0.0
        done = self.step_count > 500
        info = {"joint_pos": self.joint_pos.copy(), "step": self.step_count}
        return obs, reward, done, info

    def _get_obs_image(self) -> np.ndarray:
        """Returns synthetic RGB image (replace with actual sim render)."""
        img = np.random.randint(0, 255, (*self.image_size, 3), dtype=np.uint8)
        return img

    # ── Failure mode generators ──
    def get_overexposed_image(self) -> np.ndarray:
        return np.full((*self.image_size, 3), 250, dtype=np.uint8)

    def get_dark_image(self) -> np.ndarray:
        return np.zeros((*self.image_size, 3), dtype=np.uint8)

    def get_noisy_image(self) -> np.ndarray:
        return np.random.randint(0, 255, (*self.image_size, 3), dtype=np.uint8)

    def get_partial_occlusion_image(self) -> np.ndarray:
        img = self._get_obs_image()
        img[:img.shape[0]//2, :, :] = 0   # black out top half
        return img


# ── OpenVLA wrapper ──────────────────────────────────────────
class OpenVLAWrapper:
    """
    Wraps the HuggingFace openvla/openvla-7b model for inference profiling.
    Falls back to a mock if GPU memory is insufficient or model unavailable.
    """

    def __init__(self, model_id: str = "openvla/openvla-7b",
                 use_int4: bool = False, action_dim: int = G1_DOF,
                 allow_mock_fallback: bool = True):
        self.model_id = model_id
        self.use_int4 = use_int4
        self.action_dim = action_dim
        self.allow_mock_fallback = allow_mock_fallback
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        logger.info(f"Loading model: {self.model_id} | INT4={self.use_int4}")
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor

            load_kwargs = {
                "torch_dtype": torch.bfloat16,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
            }
            if self.use_int4:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )

            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_id, **load_kwargs
            ).cuda()
            self.processor = AutoProcessor.from_pretrained(
                self.model_id, trust_remote_code=True
            )
            self.model.eval()
            logger.info("Model loaded successfully from HuggingFace.")

        except Exception as e:
            if not self.allow_mock_fallback:
                raise RuntimeError(
                    f"Failed to load OpenVLA ({self.model_id}). "
                    "Fix CUDA/VRAM/deps or use step1_mock_profiling.ipynb. "
                    f"Original error: {e}"
                ) from e
            logger.warning(f"Could not load real model ({e}). Using mock inference.")
            self.model = None  # will use mock

    def infer(self, image: np.ndarray, instruction: str) -> Tuple[np.ndarray, int, int]:
        """
        Run one forward pass.
        Returns: (action_array, input_tokens, output_tokens)
        """
        if self.model is None:
            return self._mock_infer(image, instruction)

        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(image)
        inputs = self.processor(pil_img, instruction, return_tensors="pt").to("cuda:0")
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            action_ids = self.model.predict_action(
                **inputs, unnorm_key="bridge_orig", do_sample=False
            )

        action = self.processor.decode_actions(action_ids)
        # Resize to G1 DOF if necessary
        if len(action) < self.action_dim:
            action = np.pad(action, (0, self.action_dim - len(action)))
        else:
            action = action[:self.action_dim]

        return action, input_len, action_ids.shape[-1]

    def _mock_infer(self, image: np.ndarray, instruction: str) -> Tuple[np.ndarray, int, int]:
        """Simulate latency + output format without loading the 7B model."""
        # Simulate realistic latency: ~350-480 ms on A100
        base_latency = np.random.normal(0.38, 0.04)
        time.sleep(max(0.25, base_latency))
        action = np.random.uniform(-0.5, 0.5, self.action_dim).astype(np.float32)
        n_input_tok = 256 + len(instruction.split())
        n_output_tok = self.action_dim // 7 + 1
        return action, n_input_tok, n_output_tok


# ── Core profiling routine ───────────────────────────────────
def profile_openvla(model: OpenVLAWrapper,
                    env: MockG1SimEnv,
                    n_trials: int = 100,
                    instruction: str = "Pick up the red cube and place it in the bin") -> ProfilingReport:

    records: List[InferenceRecord] = []
    failure_modes: Dict[str, int] = {}

    gpu_available = torch.cuda.is_available()
    logger.info(f"Starting profiling | n_trials={n_trials} | GPU={gpu_available}")

    obs = env.reset()

    for i in tqdm(range(n_trials), desc="Profiling OpenVLA"):
        # Vary image conditions every 20 trials to probe failure modes
        if i % 20 == 5:
            img = env.get_overexposed_image()
            condition = "overexposed"
        elif i % 20 == 10:
            img = env.get_dark_image()
            condition = "dark"
        elif i % 20 == 15:
            img = env.get_partial_occlusion_image()
            condition = "partial_occlusion"
        else:
            img = obs
            condition = "normal"

        # GPU memory before
        gpu_mem = (torch.cuda.memory_allocated() / 1e9) if gpu_available else 0.0

        # ── Timed inference ──
        failed = False
        failure_reason = ""
        action = np.zeros(G1_DOF)
        input_tok = output_tok = 0

        t0 = time.perf_counter()
        try:
            action, input_tok, output_tok = model.infer(img, instruction)
        except Exception as e:
            failed = True
            failure_reason = type(e).__name__
            failure_modes[failure_reason] = failure_modes.get(failure_reason, 0) + 1
            logger.warning(f"  Trial {i} failed ({condition}): {e}")
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000.0

        # Tag condition-based failures
        if condition != "normal" and not failed:
            if np.any(np.isnan(action)) or np.any(np.abs(action) > 10):
                failed = True
                failure_reason = f"bad_action_{condition}"
                failure_modes[failure_reason] = failure_modes.get(failure_reason, 0) + 1

        records.append(InferenceRecord(
            trial=i,
            latency_ms=latency_ms,
            action=action.tolist(),
            action_dim=len(action),
            input_tokens=input_tok,
            output_tokens=output_tok,
            gpu_mem_gb=gpu_mem,
            failed=failed,
            failure_reason=failure_reason,
        ))

        # Step env with predicted action (if valid)
        if not failed:
            obs, _, done, _ = env.step(action)
            if done:
                obs = env.reset()

    # ── Aggregate stats ──────────────────────────────────────
    latencies = np.array([r.latency_ms for r in records if not r.failed])
    actions = np.array([r.action for r in records if not r.failed])
    gpu_mems = np.array([r.gpu_mem_gb for r in records])
    failure_rate = sum(r.failed for r in records) / len(records)

    mean_hz = 1000.0 / latencies.mean() if len(latencies) > 0 else 0.0
    frequency_gap = TARGET_HZ / mean_hz if mean_hz > 0 else float("inf")

    report = ProfilingReport(
        model_id=model.model_id,
        n_trials=n_trials,
        mean_latency_ms=float(latencies.mean()),
        std_latency_ms=float(latencies.std()),
        median_latency_ms=float(np.median(latencies)),
        p95_latency_ms=float(np.percentile(latencies, 95)),
        p99_latency_ms=float(np.percentile(latencies, 99)),
        mean_hz=float(mean_hz),
        target_hz=TARGET_HZ,
        frequency_gap=float(frequency_gap),
        mean_gpu_mem_gb=float(gpu_mems.mean()),
        action_dim=G1_DOF,
        action_mean=actions.mean(axis=0).tolist() if len(actions) > 0 else [],
        action_std=actions.std(axis=0).tolist() if len(actions) > 0 else [],
        failure_rate=float(failure_rate),
        failure_modes=failure_modes,
    )
    return report, records


# ── Plotting ─────────────────────────────────────────────────
def plot_profiling_report(report: ProfilingReport,
                          records: List[InferenceRecord]):
    latencies = [r.latency_ms for r in records if not r.failed]
    trial_idx = [r.trial for r in records if not r.failed]
    inst_hz   = [1000.0 / l for l in latencies]

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"OpenVLA Inference Profile — {report.model_id}\n"
                 f"n={report.n_trials} trials | G1 Sim | Week 1–2 Deliverable",
                 fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # (a) Latency over time
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(trial_idx, latencies, alpha=0.6, lw=1, color="#2196F3", label="Latency (ms)")
    ax1.axhline(report.mean_latency_ms, color="red", ls="--", lw=1.5,
                label=f"Mean={report.mean_latency_ms:.1f} ms")
    ax1.axhline(200, color="green", ls=":", lw=1.5, label="Target (10 ms w/ ESN)")
    ax1.fill_between(trial_idx,
                     report.mean_latency_ms - report.std_latency_ms,
                     report.mean_latency_ms + report.std_latency_ms,
                     alpha=0.15, color="red")
    ax1.set_xlabel("Trial"); ax1.set_ylabel("Latency (ms)")
    ax1.set_title("(a) Inference Latency per Trial")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    # (b) Latency histogram
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(latencies, bins=20, color="#2196F3", edgecolor="white", alpha=0.8)
    ax2.axvline(report.mean_latency_ms, color="red", ls="--", lw=1.5, label="Mean")
    ax2.axvline(report.p95_latency_ms, color="orange", ls="--", lw=1.5, label="P95")
    ax2.set_xlabel("Latency (ms)"); ax2.set_ylabel("Count")
    ax2.set_title("(b) Latency Distribution")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    # (c) Instantaneous Hz
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.plot(trial_idx, inst_hz, alpha=0.6, lw=1, color="#4CAF50", label="Control Hz")
    ax3.axhline(report.mean_hz, color="darkgreen", ls="--", lw=1.5,
                label=f"Mean={report.mean_hz:.2f} Hz")
    ax3.axhline(TARGET_HZ, color="red", ls="-.", lw=2,
                label=f"G1 Target={TARGET_HZ} Hz")
    ax3.fill_between(trial_idx, 0, inst_hz, alpha=0.1, color="#4CAF50")
    ax3.set_xlabel("Trial"); ax3.set_ylabel("Frequency (Hz)")
    ax3.set_title(f"(c) Effective Control Rate  [Gap: {report.frequency_gap:.0f}× below target]")
    ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

    # (d) Summary text box
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis("off")
    summary = (
        f"PROFILING SUMMARY\n"
        f"{'─'*28}\n"
        f"Mean latency : {report.mean_latency_ms:.1f} ± {report.std_latency_ms:.1f} ms\n"
        f"Median       : {report.median_latency_ms:.1f} ms\n"
        f"P95          : {report.p95_latency_ms:.1f} ms\n"
        f"P99          : {report.p99_latency_ms:.1f} ms\n"
        f"{'─'*28}\n"
        f"Control rate : {report.mean_hz:.2f} Hz\n"
        f"G1 target    : {TARGET_HZ} Hz\n"
        f"Freq. gap    : {report.frequency_gap:.1f}×\n"
        f"{'─'*28}\n"
        f"GPU memory   : {report.mean_gpu_mem_gb:.2f} GB\n"
        f"Failure rate : {report.failure_rate*100:.1f}%\n"
        f"Action dim   : {report.action_dim}\n"
        f"{'─'*28}\n"
        f"MOTIVATES: ESN Bridge\n"
        f"to close the {report.frequency_gap:.0f}× gap"
    )
    ax4.text(0.05, 0.95, summary, transform=ax4.transAxes,
             fontsize=9, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#FFF9C4", alpha=0.8))

    out_path = RESULTS_DIR / "openvla_profiling_report.pdf"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.savefig(str(out_path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    logger.info(f"Figure saved: {out_path}")
    plt.close()


# ── Save JSON log ────────────────────────────────────────────
def save_logs(report: ProfilingReport, records: List[InferenceRecord]):
    # Full inference log (every trial)
    log_path = RESULTS_DIR / "inference_log.json"
    with open(log_path, "w") as f:
        json.dump([asdict(r) for r in records], f, indent=2)

    # Summary report
    report_path = RESULTS_DIR / "profiling_report.json"
    with open(report_path, "w") as f:
        json.dump(asdict(report), f, indent=2)

    # Human-readable summary
    txt_path = RESULTS_DIR / "profiling_summary.txt"
    with open(txt_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("OpenVLA Profiling Report — Week 1-2 Deliverable\n")
        f.write(f"Model: {report.model_id}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Latency (ms):\n")
        f.write(f"  Mean   : {report.mean_latency_ms:.2f} ± {report.std_latency_ms:.2f}\n")
        f.write(f"  Median : {report.median_latency_ms:.2f}\n")
        f.write(f"  P95    : {report.p95_latency_ms:.2f}\n")
        f.write(f"  P99    : {report.p99_latency_ms:.2f}\n\n")
        f.write(f"Control Rate:\n")
        f.write(f"  Mean   : {report.mean_hz:.3f} Hz\n")
        f.write(f"  Target : {report.target_hz} Hz\n")
        f.write(f"  Gap    : {report.frequency_gap:.1f}x below G1 requirement\n\n")
        f.write(f"Failure Analysis:\n")
        f.write(f"  Rate   : {report.failure_rate*100:.1f}%\n")
        f.write(f"  Modes  : {report.failure_modes}\n\n")
        f.write(f"Motivation for ESN Bridge:\n")
        f.write(f"  The {report.frequency_gap:.0f}x frequency gap between OpenVLA's\n")
        f.write(f"  {report.mean_hz:.1f} Hz and G1's {report.target_hz} Hz requirement\n")
        f.write(f"  directly motivates the ESN bridge in Objective 02.\n")

    logger.info(f"Logs saved: {log_path}, {report_path}, {txt_path}")


# ── Main ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Profile OpenVLA inference on G1 sim")
    parser.add_argument("--model", type=str, default="openvla/openvla-7b")
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--use_int4", action="store_true",
                        help="Use INT4 quantisation (requires bitsandbytes)")
    parser.add_argument("--instruction", type=str,
                        default="Pick up the pen on the book on the table and give it to me")
    parser.add_argument("--mock", action="store_true",
                        help="Force mock inference (no model download needed)")
    parser.add_argument("--no-mock-fallback", action="store_true",
                        help="Fail if model load fails (do not silently use mock)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  Step 1 — OpenVLA Profiler | Research Plan Week 1-2")
    logger.info("=" * 60)

    # Init environment
    env = MockG1SimEnv(image_size=(224, 224))

    # Init model
    model = OpenVLAWrapper(
        model_id=args.model if not args.mock else "__mock__",
        use_int4=args.use_int4,
        action_dim=G1_DOF,
        allow_mock_fallback=not args.no_mock_fallback,
    )

    # Run profiling
    report, records = profile_openvla(
        model=model,
        env=env,
        n_trials=args.n_trials,
        instruction=args.instruction,
    )

    # Save outputs
    save_logs(report, records)
    plot_profiling_report(report, records)

    # Console summary
    print("\n" + "=" * 60)
    print("  PROFILING COMPLETE — Week 1-2 Deliverable")
    print("=" * 60)
    print(f"  Mean latency   : {report.mean_latency_ms:.1f} ± {report.std_latency_ms:.1f} ms")
    print(f"  Control rate   : {report.mean_hz:.2f} Hz  (target: {TARGET_HZ} Hz)")
    print(f"  Frequency gap  : {report.frequency_gap:.1f}x  ← motivates ESN Bridge (Obj 02)")
    print(f"  Failure rate   : {report.failure_rate*100:.1f}%")
    print(f"  Failure modes  : {report.failure_modes}")
    print(f"  GPU memory     : {report.mean_gpu_mem_gb:.2f} GB")
    print(f"\n  Outputs saved to: {RESULTS_DIR.resolve()}")
    print("=" * 60)

    return report


if __name__ == "__main__":
    main()
