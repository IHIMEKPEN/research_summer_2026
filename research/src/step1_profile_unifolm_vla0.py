"""
============================================================
Step 1 — UnifoLM-VLA-0 Inference Profiler
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

Task:
  - G1_Clean_Table baseline

Usage:
  python3 -m src.step1_profile_unifolm_vla0 --n_trials 100 --mock
  python3 -m src.step1_profile_unifolm_vla0 --n_trials 100 --model unitreerobotics/UnifoLM-VLM-Base
"""

import argparse
import json
import logging
import re
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.paths import results_path

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────
RESULTS_DIR = results_path("step1_profiling_unifolm_vla0")

TASK_NAME = "G1_Clean_Table"
G1_DOF = 29
TARGET_HZ = 100


# ── Data structures ──────────────────────────────────────────
@dataclass
class InferenceRecord:
    trial: int
    task: str
    latency_ms: float
    action: List[float]
    action_dim: int
    input_tokens: int
    output_tokens: int
    gpu_mem_gb: float
    failed: bool
    failure_reason: str


@dataclass
class ProfilingReport:
    model_id: str
    task: str
    n_trials: int
    mean_latency_ms: float
    std_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    mean_hz: float
    target_hz: int
    frequency_gap: float
    mean_gpu_mem_gb: float
    action_dim: int
    action_mean: List[float]
    action_std: List[float]
    failure_rate: float
    failure_modes: Dict[str, int]


# ── Mock G1 environment (sim interface) ─────────────────────
class MockG1CleanTableEnv:
    """
    Lightweight mock of Unitree G1 clean-table environment.
    Replace with actual MuJoCo environment once available.
    """

    def __init__(self, image_size: Tuple[int, int] = (224, 224)):
        self.image_size = image_size
        self.dof = G1_DOF
        self.step_count = 0
        self.joint_pos = np.zeros(self.dof)
        self.joint_vel = np.zeros(self.dof)
        logger.info(f"MockG1CleanTableEnv initialised | DOF={self.dof} | img={image_size}")

    def reset(self) -> np.ndarray:
        self.step_count = 0
        self.joint_pos = np.random.uniform(-0.1, 0.1, self.dof)
        self.joint_vel = np.zeros(self.dof)
        return self._get_obs_image()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        assert len(action) == self.dof, f"Expected {self.dof}-dim action, got {len(action)}"
        dt = 0.01
        self.joint_vel = action
        self.joint_pos += self.joint_vel * dt
        self.step_count += 1
        obs = self._get_obs_image()
        reward = 0.0
        done = self.step_count > 500
        info = {"joint_pos": self.joint_pos.copy(), "step": self.step_count, "task": TASK_NAME}
        return obs, reward, done, info

    def _get_obs_image(self) -> np.ndarray:
        return np.random.randint(0, 255, (*self.image_size, 3), dtype=np.uint8)

    def get_overexposed_image(self) -> np.ndarray:
        return np.full((*self.image_size, 3), 250, dtype=np.uint8)

    def get_dark_image(self) -> np.ndarray:
        return np.zeros((*self.image_size, 3), dtype=np.uint8)

    def get_noisy_image(self) -> np.ndarray:
        return np.random.randint(0, 255, (*self.image_size, 3), dtype=np.uint8)

    def get_partial_occlusion_image(self) -> np.ndarray:
        img = self._get_obs_image()
        img[: img.shape[0] // 2, :, :] = 0
        return img


# ── UnifoLM wrapper ──────────────────────────────────────────
class UnifoLMVLAWrapper:
    """
    Wrapper for unitreerobotics/UnifoLM-VLM-Base.
    Falls back to mock inference when model cannot be loaded.
    """

    def __init__(
        self,
        model_id: str = "unitreerobotics/UnifoLM-VLM-Base",
        use_int4: bool = False,
        action_dim: int = G1_DOF,
        allow_mock_fallback: bool = True,
    ):
        self.model_id = model_id
        self.use_int4 = use_int4
        self.action_dim = action_dim
        self.allow_mock_fallback = allow_mock_fallback
        self.model: Optional[Any] = None
        self.processor: Optional[Any] = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._load_model()

    def _load_model(self):
        logger.info(f"Loading model: {self.model_id} | INT4={self.use_int4}")
        try:
            from transformers import AutoProcessor

            load_kwargs: Dict[str, Any] = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            }
            if self.use_int4:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )

            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)

            model = None
            from transformers import AutoModelForImageTextToText, AutoModelForVision2Seq

            for model_cls in (AutoModelForVision2Seq, AutoModelForImageTextToText):
                try:
                    model = model_cls.from_pretrained(self.model_id, **load_kwargs)
                    logger.info(f"Loaded with {model_cls.__name__}")
                    break
                except Exception:
                    continue

            if model is None:
                raise RuntimeError("Unable to load model with supported transformers classes.")

            if torch.cuda.is_available():
                model = model.to(self.device)
            self.model = model.eval()
            logger.info("Model loaded successfully from HuggingFace.")

        except Exception as e:
            if not self.allow_mock_fallback:
                raise RuntimeError(
                    f"Failed to load UnifoLM ({self.model_id}). "
                    f"Original error: {e}"
                ) from e
            logger.warning(f"Could not load real model ({e}). Using mock inference.")
            self.model = None
            self.processor = None

    def infer(self, image: np.ndarray, instruction: str) -> Tuple[np.ndarray, int, int]:
        if self.model is None or self.processor is None:
            return self._mock_infer(image, instruction)

        from PIL import Image as PILImage

        pil_img = PILImage.fromarray(image)
        prompt = (
            "You are a humanoid robot controller for Unitree G1. "
            f"Task: {TASK_NAME}. "
            f"Instruction: {instruction}. "
            f"Return exactly {self.action_dim} joint velocity values as a Python list of floats."
        )

        inputs, input_len = self._prepare_inputs(prompt=prompt, image=pil_img)
        action, output_tok = self._forward_and_parse_action(inputs=inputs)
        action = self._reshape_action(action)
        return action, input_len, output_tok

    def _prepare_inputs(self, prompt: str, image: Any) -> Tuple[Dict[str, Any], int]:
        if hasattr(self.processor, "apply_chat_template"):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=text, images=[image], return_tensors="pt")
        else:
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")

        input_len = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0
        if torch.cuda.is_available():
            inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        return inputs, input_len

    def _forward_and_parse_action(self, inputs: Dict[str, Any]) -> Tuple[np.ndarray, int]:
        with torch.no_grad():
            if hasattr(self.model, "predict_action"):
                action = self.model.predict_action(**inputs, do_sample=False)
                action_arr = np.asarray(action, dtype=np.float32).flatten()
                output_tok = len(action_arr)
                return action_arr, output_tok

            generated = self.model.generate(**inputs, max_new_tokens=96, do_sample=False)

        input_ids = inputs.get("input_ids", None)
        output_tok = int(generated.shape[-1] - input_ids.shape[-1]) if input_ids is not None else int(generated.shape[-1])
        decoded = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        action_arr = self._parse_action_text(decoded)
        return action_arr, output_tok

    def _parse_action_text(self, text: str) -> np.ndarray:
        bracket_match = re.search(r"\[([^\]]+)\]", text, flags=re.DOTALL)
        if bracket_match:
            text = bracket_match.group(1)

        numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        if not numbers:
            raise ValueError("Model output did not contain numeric action values.")
        action = np.array([float(x) for x in numbers], dtype=np.float32)
        return action

    def _reshape_action(self, action: np.ndarray) -> np.ndarray:
        if len(action) < self.action_dim:
            action = np.pad(action, (0, self.action_dim - len(action)))
        else:
            action = action[: self.action_dim]
        return action

    def _mock_infer(self, image: np.ndarray, instruction: str) -> Tuple[np.ndarray, int, int]:
        base_latency = np.random.normal(0.42, 0.05)
        time.sleep(max(0.25, base_latency))
        action = np.random.uniform(-0.5, 0.5, self.action_dim).astype(np.float32)
        n_input_tok = 280 + len(instruction.split())
        n_output_tok = self.action_dim
        return action, n_input_tok, n_output_tok


# ── Core profiling routine ───────────────────────────────────
def profile_unifolm_vla0(
    model: UnifoLMVLAWrapper,
    env: MockG1CleanTableEnv,
    n_trials: int = 100,
    instruction: str = "Clean the table by moving all clutter items into the bin.",
) -> Tuple[ProfilingReport, List[InferenceRecord]]:
    records: List[InferenceRecord] = []
    failure_modes: Dict[str, int] = {}

    gpu_available = torch.cuda.is_available()
    logger.info(f"Starting profiling | task={TASK_NAME} | n_trials={n_trials} | GPU={gpu_available}")

    obs = env.reset()

    for i in tqdm(range(n_trials), desc="Profiling UnifoLM-VLA-0"):
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

        gpu_mem = (torch.cuda.memory_allocated() / 1e9) if gpu_available else 0.0

        failed = False
        failure_reason = ""
        action = np.zeros(G1_DOF, dtype=np.float32)
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

        if condition != "normal" and not failed:
            if np.any(np.isnan(action)) or np.any(np.abs(action) > 10):
                failed = True
                failure_reason = f"bad_action_{condition}"
                failure_modes[failure_reason] = failure_modes.get(failure_reason, 0) + 1

        records.append(
            InferenceRecord(
                trial=i,
                task=TASK_NAME,
                latency_ms=latency_ms,
                action=action.tolist(),
                action_dim=len(action),
                input_tokens=input_tok,
                output_tokens=output_tok,
                gpu_mem_gb=gpu_mem,
                failed=failed,
                failure_reason=failure_reason,
            )
        )

        if not failed:
            obs, _, done, _ = env.step(action)
            if done:
                obs = env.reset()

    latencies = np.array([r.latency_ms for r in records if not r.failed], dtype=np.float32)
    actions = np.array([r.action for r in records if not r.failed], dtype=np.float32)
    gpu_mems = np.array([r.gpu_mem_gb for r in records], dtype=np.float32)
    failure_rate = float(sum(r.failed for r in records) / len(records))

    mean_hz = float(1000.0 / latencies.mean()) if len(latencies) > 0 else 0.0
    frequency_gap = float(TARGET_HZ / mean_hz) if mean_hz > 0 else float("inf")

    report = ProfilingReport(
        model_id=model.model_id,
        task=TASK_NAME,
        n_trials=n_trials,
        mean_latency_ms=float(latencies.mean()) if len(latencies) > 0 else float("inf"),
        std_latency_ms=float(latencies.std()) if len(latencies) > 0 else float("inf"),
        median_latency_ms=float(np.median(latencies)) if len(latencies) > 0 else float("inf"),
        p95_latency_ms=float(np.percentile(latencies, 95)) if len(latencies) > 0 else float("inf"),
        p99_latency_ms=float(np.percentile(latencies, 99)) if len(latencies) > 0 else float("inf"),
        mean_hz=mean_hz,
        target_hz=TARGET_HZ,
        frequency_gap=frequency_gap,
        mean_gpu_mem_gb=float(gpu_mems.mean()) if len(gpu_mems) > 0 else 0.0,
        action_dim=G1_DOF,
        action_mean=actions.mean(axis=0).tolist() if len(actions) > 0 else [],
        action_std=actions.std(axis=0).tolist() if len(actions) > 0 else [],
        failure_rate=failure_rate,
        failure_modes=failure_modes,
    )
    return report, records


# ── Plotting ─────────────────────────────────────────────────
def plot_profiling_report(report: ProfilingReport, records: List[InferenceRecord]):
    latencies = [r.latency_ms for r in records if not r.failed]
    trial_idx = [r.trial for r in records if not r.failed]
    inst_hz = [1000.0 / l for l in latencies]

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        f"UnifoLM-VLA-0 Inference Profile — {report.model_id}\n"
        f"Task={report.task} | n={report.n_trials} trials | Week 1–2 Deliverable",
        fontsize=13,
        fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(trial_idx, latencies, alpha=0.6, lw=1, color="#2196F3", label="Latency (ms)")
    ax1.axhline(report.mean_latency_ms, color="red", ls="--", lw=1.5, label=f"Mean={report.mean_latency_ms:.1f} ms")
    ax1.fill_between(
        trial_idx,
        report.mean_latency_ms - report.std_latency_ms,
        report.mean_latency_ms + report.std_latency_ms,
        alpha=0.15,
        color="red",
    )
    ax1.set_xlabel("Trial")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("(a) Inference Latency per Trial")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(latencies, bins=20, color="#2196F3", edgecolor="white", alpha=0.8)
    ax2.axvline(report.mean_latency_ms, color="red", ls="--", lw=1.5, label="Mean")
    ax2.axvline(report.p95_latency_ms, color="orange", ls="--", lw=1.5, label="P95")
    ax2.set_xlabel("Latency (ms)")
    ax2.set_ylabel("Count")
    ax2.set_title("(b) Latency Distribution")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[1, :2])
    ax3.plot(trial_idx, inst_hz, alpha=0.6, lw=1, color="#4CAF50", label="Control Hz")
    ax3.axhline(report.mean_hz, color="darkgreen", ls="--", lw=1.5, label=f"Mean={report.mean_hz:.2f} Hz")
    ax3.axhline(TARGET_HZ, color="red", ls="-.", lw=2, label=f"G1 Target={TARGET_HZ} Hz")
    ax3.fill_between(trial_idx, 0, inst_hz, alpha=0.1, color="#4CAF50")
    ax3.set_xlabel("Trial")
    ax3.set_ylabel("Frequency (Hz)")
    ax3.set_title(f"(c) Effective Control Rate  [Gap: {report.frequency_gap:.0f}x below target]")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis("off")
    summary = (
        f"PROFILING SUMMARY\n"
        f"{'─' * 30}\n"
        f"Model        : {report.model_id}\n"
        f"Task         : {report.task}\n"
        f"Mean latency : {report.mean_latency_ms:.1f} ± {report.std_latency_ms:.1f} ms\n"
        f"Median       : {report.median_latency_ms:.1f} ms\n"
        f"P95          : {report.p95_latency_ms:.1f} ms\n"
        f"P99          : {report.p99_latency_ms:.1f} ms\n"
        f"{'─' * 30}\n"
        f"Control rate : {report.mean_hz:.2f} Hz\n"
        f"G1 target    : {TARGET_HZ} Hz\n"
        f"Freq. gap    : {report.frequency_gap:.1f}x\n"
        f"{'─' * 30}\n"
        f"GPU memory   : {report.mean_gpu_mem_gb:.2f} GB\n"
        f"Failure rate : {report.failure_rate * 100:.1f}%\n"
        f"Action dim   : {report.action_dim}\n"
    )
    ax4.text(
        0.05,
        0.95,
        summary,
        transform=ax4.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#FFF9C4", alpha=0.8),
    )

    out_path = RESULTS_DIR / "unifolm_vla0_profiling_report.pdf"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.savefig(str(out_path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    logger.info(f"Figure saved: {out_path}")
    plt.close()


# ── Save JSON log ────────────────────────────────────────────
def save_logs(report: ProfilingReport, records: List[InferenceRecord]):
    log_path = RESULTS_DIR / "inference_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records], f, indent=2)

    report_path = RESULTS_DIR / "profiling_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)

    txt_path = RESULTS_DIR / "profiling_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("UnifoLM-VLA-0 Profiling Report — Week 1-2 Deliverable\n")
        f.write(f"Model: {report.model_id}\n")
        f.write(f"Task : {report.task}\n")
        f.write("=" * 60 + "\n\n")
        f.write("Latency (ms):\n")
        f.write(f"  Mean   : {report.mean_latency_ms:.2f} ± {report.std_latency_ms:.2f}\n")
        f.write(f"  Median : {report.median_latency_ms:.2f}\n")
        f.write(f"  P95    : {report.p95_latency_ms:.2f}\n")
        f.write(f"  P99    : {report.p99_latency_ms:.2f}\n\n")
        f.write("Control Rate:\n")
        f.write(f"  Mean   : {report.mean_hz:.3f} Hz\n")
        f.write(f"  Target : {report.target_hz} Hz\n")
        f.write(f"  Gap    : {report.frequency_gap:.1f}x below G1 requirement\n\n")
        f.write("Failure Analysis:\n")
        f.write(f"  Rate   : {report.failure_rate * 100:.1f}%\n")
        f.write(f"  Modes  : {report.failure_modes}\n")

    logger.info(f"Logs saved: {log_path}, {report_path}, {txt_path}")


# ── Main ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Profile UnifoLM-VLA-0 inference on G1_Clean_Table sim")
    parser.add_argument("--model", type=str, default="unitreerobotics/UnifoLM-VLM-Base")
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--use_int4", action="store_true", help="Use INT4 quantisation (requires bitsandbytes)")
    parser.add_argument(
        "--instruction",
        type=str,
        default="Clean the table by moving all clutter items into the bin.",
    )
    parser.add_argument("--mock", action="store_true", help="Force mock inference (no model download needed)")
    parser.add_argument(
        "--no-mock-fallback",
        action="store_true",
        help="Fail if model load fails (do not silently use mock)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  Step 1 — UnifoLM-VLA-0 Profiler | Research Plan Week 1-2")
    logger.info("=" * 60)

    env = MockG1CleanTableEnv(image_size=(224, 224))
    model = UnifoLMVLAWrapper(
        model_id=args.model if not args.mock else "__mock__",
        use_int4=args.use_int4,
        action_dim=G1_DOF,
        allow_mock_fallback=not args.no_mock_fallback,
    )

    report, records = profile_unifolm_vla0(
        model=model,
        env=env,
        n_trials=args.n_trials,
        instruction=args.instruction,
    )

    save_logs(report, records)
    plot_profiling_report(report, records)

    print("\n" + "=" * 60)
    print("  PROFILING COMPLETE — Week 1-2 Deliverable")
    print("=" * 60)
    print(f"  Model          : {report.model_id}")
    print(f"  Task           : {report.task}")
    print(f"  Mean latency   : {report.mean_latency_ms:.1f} ± {report.std_latency_ms:.1f} ms")
    print(f"  Control rate   : {report.mean_hz:.2f} Hz  (target: {TARGET_HZ} Hz)")
    print(f"  Frequency gap  : {report.frequency_gap:.1f}x")
    print(f"  Failure rate   : {report.failure_rate * 100:.1f}%")
    print(f"  Failure modes  : {report.failure_modes}")
    print(f"  GPU memory     : {report.mean_gpu_mem_gb:.2f} GB")
    print(f"\n  Outputs saved to: {RESULTS_DIR.resolve()}")
    print("=" * 60)

    return report


if __name__ == "__main__":
    main()
