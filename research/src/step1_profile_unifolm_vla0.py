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
  python3 -m src.step1_profile_unifolm_vla0 --n_trials 100 --model unitreerobotics/UnifoLM-VLA-Base
"""

import argparse
import json
import logging
import re
import sys
import time
import warnings
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.paths import results_path
from src.unifolm_cuda_graph import VLACUDAGraphEngine, cuda_free_mib
from src.unifolm_vla_runtime import AsyncVLARuntime, GPUActionRegister, PinnedHostTransfer

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────
RESULTS_BASE_DIR = results_path("step1_profiling_unifolm_vla0")
RESULTS_DIR = RESULTS_BASE_DIR

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
    profiler_source: str
    generated_at: str
    run_tag: str


def _normalize_profiler_source(profiler_source: str) -> str:
    cleaned = profiler_source.strip().lower().replace(" ", "_")
    return cleaned or "unknown_profiler"


def make_profile_run_tag(profiler_source: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{_normalize_profiler_source(profiler_source)}_{ts}"


def make_run_results_dir(profiler_source: str, run_tag: Optional[str] = None) -> Path:
    """Create a timestamped run folder under results/step1_profiling_unifolm_vla0/."""
    run_tag = run_tag or make_profile_run_tag(profiler_source)
    run_dir = RESULTS_BASE_DIR / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


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


# UnifoLM-VLA-Base (Qwen2.5-VL + action head; config: transformers>=4.49). OpenVLA uses 4.40 — separate envs.
UNIFOLM_MIN_TRANSFORMERS = (4, 49, 0)
UNIFOLM_MIN_JINJA2 = (3, 1, 0)


def _parse_version_tuple(version: str) -> Tuple[int, ...]:
    parts: List[int] = []
    for piece in version.split(".")[:3]:
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def _parse_transformers_version(version: str) -> Tuple[int, ...]:
    return _parse_version_tuple(version)


def _unifolm_transformers_ok() -> bool:
    import transformers

    return _parse_version_tuple(transformers.__version__) >= UNIFOLM_MIN_TRANSFORMERS


def _unifolm_jinja2_ok() -> bool:
    try:
        import jinja2
    except ImportError:
        return False
    return _parse_version_tuple(jinja2.__version__) >= UNIFOLM_MIN_JINJA2


def _unifolm_model_loader_classes() -> List[Any]:
    """Return AutoModel classes to try, newest API first; skip symbols missing in this transformers build."""
    import importlib

    import transformers

    if not _unifolm_transformers_ok():
        raise ImportError(
            f"transformers {transformers.__version__} is too old for UnifoLM (Qwen2.5-VL). "
            f"Need >= {'.'.join(map(str, UNIFOLM_MIN_TRANSFORMERS))}. "
            "Use a separate venv: pip install -r requirements-unifolm-gpu.txt"
        )

    candidates = (
        "transformers.Qwen2_5_VLForConditionalGeneration",
        "transformers.AutoModelForImageTextToText",
        "transformers.AutoModelForVision2Seq",
    )
    loader_classes: List[Any] = []
    for qualname in candidates:
        module_path, _, attr = qualname.rpartition(".")
        try:
            module = importlib.import_module(module_path)
            loader_classes.append(getattr(module, attr))
        except (ImportError, AttributeError):
            continue
    if not loader_classes:
        raise ImportError("No compatible HuggingFace model loader found in transformers.")
    return loader_classes


def _normalize_unifolm_config(config: Any) -> Any:
    """
    UnifoLM-VLA-Base may ship nested configs as raw dicts.

    transformers 4.49 expects PretrainedConfig objects when building
    GenerationConfig, which otherwise raises:
    AttributeError: 'dict' object has no attribute 'to_dict'
    """
    from transformers import PretrainedConfig

    if isinstance(getattr(config, "text_config", None), dict):
        config.text_config = PretrainedConfig.from_dict(config.text_config)
    if isinstance(getattr(config, "vision_config", None), dict):
        config.vision_config = PretrainedConfig.from_dict(config.vision_config)
    return config


def _ensure_unifolm_vla_on_path() -> Optional[Path]:
    """
    Make cloned ``unifolm-vla/src`` importable without ``pip install -e .``.

    Looks for a sibling checkout at ``research_summer_2026/unifolm-vla``.
    """
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "unifolm-vla" / "src",
        here.parents[1] / "unifolm-vla" / "src",
        Path.cwd() / "unifolm-vla" / "src",
        Path.cwd().parent / "unifolm-vla" / "src",
    ]
    for src in candidates:
        if (src / "unifolm_vla").is_dir():
            src_str = str(src)
            if src_str not in sys.path:
                sys.path.insert(0, src_str)
            return src
    return None


def _is_unifolm_vla_hub(model_id: str) -> bool:
    """True for Unitree VLA checkpoints (``UnifoLM-VLA-*``), not VLM-only backbones."""
    return "UnifoLM-VLA" in model_id or "Unifolm-VLA" in model_id


def _bounds_normalize(values: np.ndarray, norm_stats: Dict[str, Any]) -> np.ndarray:
    mask = norm_stats.get("mask", np.ones_like(norm_stats["min"], dtype=bool))
    high = np.array(norm_stats["max"], dtype=np.float32)
    low = np.array(norm_stats["min"], dtype=np.float32)
    return np.clip(
        np.where(mask, 2 * (values - low) / (high - low + 1e-8) - 1, values),
        a_min=-1.0,
        a_max=1.0,
    )


def _bounds_unnormalize(values: np.ndarray, norm_stats: Dict[str, Any]) -> np.ndarray:
    mask = norm_stats.get("mask", np.ones_like(norm_stats["min"], dtype=bool))
    high = np.array(norm_stats["max"], dtype=np.float32)
    low = np.array(norm_stats["min"], dtype=np.float32)
    return np.where(
        mask,
        0.5 * (values + 1) * (high - low + 1e-8) + low,
        values,
    ).astype(np.float32)


def _download_unifolm_vla_snapshot(model_id: str) -> Path:
    """Download VLA checkpoint + config.yaml + dataset_statistics.json from the Hub."""
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(
        repo_id=model_id,
        allow_patterns=["config.yaml", "dataset_statistics.json", "checkpoints/*"],
    )
    ckpt_path = Path(local_dir) / "checkpoints" / "pytorch_model.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Expected VLA checkpoint at {ckpt_path}. "
            f"Repo {model_id} must contain checkpoints/pytorch_model.pt."
        )
    return ckpt_path


# ── UnifoLM wrapper (V100 FP16 + CUDA Graphs + async streams) ────────────────
class UnifoLMVLAWrapper:
    """
    Wrapper for unitreerobotics/UnifoLM-VLA-Base.

    V100 optimizations (Volta SM 7.0):
      - Native ``torch.float16`` weights + ``autocast(fp16)`` for Tensor Cores
      - ``torch.cuda.CUDAGraph`` replay to eliminate per-kernel CPU dispatch
      - Pinned host memory + ``non_blocking`` H2D copies (PCIe Gen 3)
      - GPU-resident ``infer_gpu`` hot path (no ``.item()`` / ``.cpu()`` in loop)
      - Optional :class:`AsyncVLARuntime` for 100 Hz ESN register sampling

    Falls back to mock inference when the model cannot be loaded.
    """

    def __init__(
        self,
        model_id: str = "unitreerobotics/UnifoLM-VLA-Base",
        use_int4: bool = False,
        action_dim: int = G1_DOF,
        allow_mock_fallback: bool = True,
        use_fp16: bool = True,
        use_cuda_graph: bool = True,
        max_new_tokens: int = 64,
        cuda_graph_warmup: int = 2,
        cuda_graph_min_free_mib: float = 512.0,
        vlm_backbone_id: str = "unitreerobotics/UnifoLM-VLM-Base",
        unnorm_key: str = "g1_clean_table",
    ):
        self.model_id = model_id
        self.vlm_backbone_id = vlm_backbone_id
        self.unnorm_key = unnorm_key
        self.use_int4 = use_int4
        self.action_dim = action_dim
        self.allow_mock_fallback = allow_mock_fallback
        self.use_fp16 = use_fp16 and torch.cuda.is_available()
        self.use_cuda_graph = use_cuda_graph and torch.cuda.is_available()
        self.max_new_tokens = max_new_tokens
        self.cuda_graph_warmup = max(cuda_graph_warmup, 1)
        self.cuda_graph_min_free_mib = cuda_graph_min_free_mib

        self.model: Optional[Any] = None
        self.processor: Optional[Any] = None
        self._uses_unifolm_vla_framework: bool = False
        self._norm_stats_action: Optional[Dict[str, Any]] = None
        self._norm_stats_proprio: Optional[Dict[str, Any]] = None
        self._cuda_graph_engine: Optional[VLACUDAGraphEngine] = None
        self._graph_instruction: Optional[str] = None
        self._cuda_graph_capture_attempted: bool = False

        cuda_ok = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if cuda_ok else "cpu")
        self.inference_dtype = torch.float16 if self.use_fp16 else torch.float32
        self._host_transfer = PinnedHostTransfer(self.device) if cuda_ok else None
        self.vla_stream = torch.cuda.Stream(device=self.device) if cuda_ok else None

        self.action_register: Optional[GPUActionRegister] = None
        self.async_runtime: Optional[AsyncVLARuntime] = None

        self._load_model()

    def _reset_vlm_generation_state(self) -> None:
        """
        Clear mRoPE / KV side-effects left by manual CUDA Graph warmup forwards.

        Without this, a failed graph capture can leave ``rope_deltas`` set and
        ``generate()`` emits token soup instead of action text.
        """
        if self.model is None:
            return
        if hasattr(self.model, "rope_deltas"):
            self.model.rope_deltas = None
        inner = getattr(self.model, "model", None)
        if inner is not None and hasattr(inner, "rope_deltas"):
            inner.rope_deltas = None
        if torch.cuda.is_available():
            if self.vla_stream is not None:
                self.vla_stream.synchronize()
            torch.cuda.current_stream().synchronize()

    def _load_model(self) -> None:
        logger.info(
            f"Loading model: {self.model_id} | INT4={self.use_int4} | "
            f"FP16={self.use_fp16} | cuda_graph={self.use_cuda_graph}"
        )
        try:
            if _is_unifolm_vla_hub(self.model_id):
                self._load_unifolm_vla_framework()
            else:
                self._load_unifolm_vlm_transformers()
        except Exception as e:
            if not self.allow_mock_fallback:
                raise RuntimeError(
                    f"Failed to load UnifoLM ({self.model_id}). "
                    f"Original error: {e}"
                ) from e
            logger.warning(f"Could not load real model ({e}). Using mock inference.")
            self.model = None
            self.processor = None

    def _load_unifolm_vla_framework(self) -> None:
        """
        Load ``UnifoLM-VLA-*`` via the official ``unifolm-vla`` package.

        Hub layout: ``checkpoints/pytorch_model.pt``, ``config.yaml``,
        ``dataset_statistics.json`` (not a plain transformers repo).
        """
        try:
            vla_src = _ensure_unifolm_vla_on_path()
            if vla_src is None:
                raise ImportError(
                    "Could not find unifolm-vla checkout. Clone next to this repo:\n"
                    "  git clone https://github.com/unitreerobotics/unifolm-vla.git "
                    "../unifolm-vla\n"
                    "Then: pip install omegaconf qwen-vl-utils"
                )
            from unifolm_vla.model.framework.base_framework import baseframework
        except ImportError as exc:
            raise ImportError(
                "UnifoLM-VLA-Base needs the unifolm-vla source tree on PYTHONPATH.\n"
                "  git clone https://github.com/unitreerobotics/unifolm-vla.git "
                "(sibling of research/)\n"
                "  pip install omegaconf qwen-vl-utils\n"
                "Do NOT use pip install -e . — editable install fails on this repo."
            ) from exc

        ckpt_path = _download_unifolm_vla_snapshot(self.model_id)
        logger.info(f"Loading VLA checkpoint: {ckpt_path}")
        logger.info(f"VLM backbone: {self.vlm_backbone_id} | unnorm_key={self.unnorm_key}")

        vla = baseframework.from_pretrained(str(ckpt_path), vlm_pretrained_path=self.vlm_backbone_id)
        if self.unnorm_key not in vla.norm_stats:
            available = ", ".join(sorted(vla.norm_stats.keys()))
            raise KeyError(
                f"unnorm_key={self.unnorm_key!r} not in VLA dataset statistics. "
                f"Available: {available}"
            )

        if self.use_fp16 and torch.cuda.is_available():
            vla = vla.to(torch.float16)
        elif torch.cuda.is_available():
            vla = vla.to(torch.float32)

        if torch.cuda.is_available():
            vla = vla.to(self.device)
        vla = vla.eval()

        self.model = vla
        self.processor = vla.qwen_vl_interface.processor
        self._uses_unifolm_vla_framework = True
        self._norm_stats_action = vla.norm_stats[self.unnorm_key]["action"]
        self._norm_stats_proprio = vla.norm_stats[self.unnorm_key]["proprio"]
        self.use_cuda_graph = False

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            free_mib = cuda_free_mib(self.device)
            logger.info(
                f"UnifoLM-VLA loaded (predict_action path). "
                f"Free VRAM={free_mib:.0f} MiB."
            )
        else:
            logger.info("UnifoLM-VLA loaded (predict_action path).")

    def _load_unifolm_vlm_transformers(self) -> None:
        from transformers import AutoConfig, AutoProcessor

        load_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "torch_dtype": self.inference_dtype if torch.cuda.is_available() else torch.float32,
        }
        if self.use_int4:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.inference_dtype,
            )

        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        config = _normalize_unifolm_config(
            AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
        )

        model = None
        load_errors: List[str] = []
        for model_cls in _unifolm_model_loader_classes():
            try:
                model = model_cls.from_pretrained(self.model_id, config=config, **load_kwargs)
                if model_cls.__name__ != "Qwen2_5_VLForConditionalGeneration" and load_errors:
                    logger.warning(
                        f"Loaded with {model_cls.__name__} (preferred: Qwen2_5_VLForConditionalGeneration). "
                        f"Earlier: {'; '.join(load_errors)}"
                    )
                else:
                    logger.info(f"Loaded with {model_cls.__name__}")
                break
            except Exception as exc:
                load_errors.append(f"{model_cls.__name__}: {exc}")
                continue

        if model is None:
            detail = "; ".join(load_errors) if load_errors else "no loader attempted"
            raise RuntimeError(
                f"Unable to load model with supported transformers classes ({detail})."
            )

        if torch.cuda.is_available():
            model = model.to(self.device)
        model = model.eval()

        self.model = model
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            free_mib = cuda_free_mib(self.device)
            logger.info(
                f"Model loaded successfully from HuggingFace. "
                f"Free VRAM={free_mib:.0f} MiB (CUDA Graph needs >= {self.cuda_graph_min_free_mib:.0f} MiB)."
            )
        else:
            logger.info("Model loaded successfully from HuggingFace.")

    def _ensure_cuda_graph(self, instruction: str, template_inputs: Dict[str, Any]) -> None:
        """
        One-time CUDA Graph capture for a fixed instruction + input shape.

        Performs ``cuda_graph_warmup`` eager passes, then records the graph on
        ``vla_stream`` for replay with in-place ``copy_`` updates.
        """
        if not self.use_cuda_graph or self.model is None:
            return
        if self._cuda_graph_engine is not None and self._graph_instruction == instruction:
            return
        if self._cuda_graph_capture_attempted and self._cuda_graph_engine is None:
            return
        if hasattr(self.model, "predict_action"):
            logger.info("predict_action path detected — CUDA Graph capture skipped.")
            self._cuda_graph_capture_attempted = True
            return

        if self.vla_stream is None:
            return

        free_mib = cuda_free_mib(self.device)
        logger.info(f"CUDA Graph capture attempt | free VRAM={free_mib:.0f} MiB")
        if free_mib < self.cuda_graph_min_free_mib:
            logger.warning(
                f"Skipping CUDA Graph capture: {free_mib:.0f} MiB free < "
                f"{self.cuda_graph_min_free_mib:.0f} MiB required on V100 32GB."
            )
            self._cuda_graph_capture_attempted = True
            return

        VLACUDAGraphEngine._release_memory_before_capture()
        self._cuda_graph_capture_attempted = True
        engine = VLACUDAGraphEngine(
            self.model,
            device=self.device,
            stream=self.vla_stream,
            max_new_tokens=self.max_new_tokens,
            warmup_iters=self.cuda_graph_warmup,
            use_fp16=self.use_fp16,
            joint_state_dim=self.action_dim,
            min_capture_headroom_mib=self.cuda_graph_min_free_mib,
        )
        try:
            engine.capture_from_template(template_inputs)
        except RuntimeError as exc:
            logger.warning(
                f"CUDA Graph capture failed — falling back to eager generate: {exc}"
            )
            self._cuda_graph_engine = None
            self._graph_instruction = None
            VLACUDAGraphEngine._cleanup_failed_capture()
            self._reset_vlm_generation_state()
            return

        self._cuda_graph_engine = engine
        self._graph_instruction = instruction
        logger.info(f"CUDA Graph ready (mode={engine.capture_mode}).")

    def warmup_cuda_graph(
        self,
        image: np.ndarray,
        instruction: str,
        joint_state: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Capture CUDA Graph **without** running a full eager generate (saves VRAM).

        Returns True when a graph is captured and ready for replay.
        """
        if self.model is None or self.processor is None:
            return False

        from PIL import Image as PILImage

        pil_img = PILImage.fromarray(image)
        prompt = (
            "You are a humanoid robot controller for Unitree G1. "
            f"Task: {TASK_NAME}. "
            f"Instruction: {instruction}. "
            f"Return exactly {self.action_dim} joint velocity values as a Python list of floats."
        )
        inputs, _ = self._prepare_inputs(prompt=prompt, image=pil_img)

        if self.use_cuda_graph and not hasattr(self.model, "predict_action"):
            self._ensure_cuda_graph(instruction, inputs)

        return self._cuda_graph_engine is not None and self._cuda_graph_engine.captured

    def create_async_runtime(
        self,
        instruction: str,
        *,
        poll_interval_s: float = 0.0,
        auto_start: bool = True,
    ) -> AsyncVLARuntime:
        """
        Build the dual-stream framework: VLA on ``vla_stream``, ESN reads
        :meth:`action_register.sample` at 100 Hz on the default stream.
        """
        if self.action_register is None:
            self.action_register = GPUActionRegister(
                action_dim=self.action_dim,
                device=self.device,
                dtype=torch.float32,
            )
        runtime = AsyncVLARuntime(
            infer_fn=self.infer_gpu,
            register=self.action_register,
            instruction=instruction,
            device=self.device,
            poll_interval_s=poll_interval_s,
        )
        self.async_runtime = runtime
        if auto_start:
            runtime.start()
        return runtime

    def infer(self, image: np.ndarray, instruction: str) -> Tuple[np.ndarray, int, int]:
        """
        Profiling / logging entry point.

        Sync boundary is intentionally deferred to the single CPU copy at the
        end — never inside the GPU forward pass.
        """
        action_gpu, input_tok, output_tok = self.infer_gpu(image, instruction)
        if action_gpu.device.type == "cuda":
            action = action_gpu.detach().float().cpu().numpy()
        else:
            action = action_gpu.detach().float().numpy()
        return action, input_tok, output_tok

    def infer_gpu(
        self,
        image: np.ndarray,
        instruction: str,
        joint_state: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, int, int]:
        """
        Hot-path inference — action tensor stays on GPU; no sync primitives.
        """
        if self.model is None or self.processor is None:
            return self._mock_infer_gpu(image, instruction)

        from PIL import Image as PILImage

        pil_img = PILImage.fromarray(image)

        if self._uses_unifolm_vla_framework:
            action_gpu, output_tok = self._infer_unifolm_vla_action_gpu(
                pil_img, instruction, joint_state
            )
            action_gpu = self._reshape_action_gpu(action_gpu)
            return action_gpu, 0, output_tok

        prompt = (
            "You are a humanoid robot controller for Unitree G1. "
            f"Task: {TASK_NAME}. "
            f"Instruction: {instruction}. "
            f"Return exactly {self.action_dim} joint velocity values as a Python list of floats."
        )

        inputs, input_len = self._prepare_inputs(prompt=prompt, image=pil_img)

        joint_state_gpu: Optional[torch.Tensor] = None
        if joint_state is not None and self._host_transfer is not None:
            joint_state_gpu = self._host_transfer.numpy_to_device(
                joint_state.astype(np.float32)
            )

        if self.use_cuda_graph and not hasattr(self.model, "predict_action"):
            self._ensure_cuda_graph(instruction, inputs)

        action_gpu, output_tok = self._forward_and_parse_action_gpu(
            inputs=inputs,
            joint_state_gpu=joint_state_gpu,
        )
        action_gpu = self._reshape_action_gpu(action_gpu)
        return action_gpu, input_len, output_tok

    def _prepare_inputs(self, prompt: str, image: Any) -> Tuple[Dict[str, Any], int]:
        use_chat_template = hasattr(self.processor, "apply_chat_template") and _unifolm_jinja2_ok()
        if use_chat_template:
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

        # PCIe Gen 3: pin host tensors, async H2D on the VLA stream.
        if self._host_transfer is not None and self.vla_stream is not None:
            with torch.cuda.stream(self.vla_stream):
                inputs = self._host_transfer.transfer_batch(inputs)
        elif self._host_transfer is not None:
            inputs = self._host_transfer.transfer_batch(inputs)
        elif torch.cuda.is_available():
            inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        return inputs, input_len

    def _infer_unifolm_vla_action_gpu(
        self,
        pil_img: Any,
        instruction: str,
        joint_state: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, int]:
        """Official UnifoLM-VLA ``predict_action`` path (action head, not text generation)."""
        assert self.model is not None and self.processor is not None
        assert self._norm_stats_action is not None and self._norm_stats_proprio is not None

        lang = instruction.lower().strip()
        if not lang.startswith("the task is"):
            lang = f'The task is "{lang}".'

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": lang},
                ],
            }
        ]

        try:
            from qwen_vl_utils import process_vision_info

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            batch_input = self.processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        except ImportError:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            batch_input = self.processor(
                text=text,
                images=[pil_img],
                return_tensors="pt",
            )

        proprio_dim = len(self._norm_stats_proprio["min"])
        if joint_state is None:
            proprio = np.zeros(proprio_dim, dtype=np.float32)
        else:
            proprio = np.asarray(joint_state, dtype=np.float32).flatten()
            if proprio.size < proprio_dim:
                proprio = np.pad(proprio, (0, proprio_dim - proprio.size))
            proprio = proprio[:proprio_dim]

        proprio_norm = _bounds_normalize(proprio, self._norm_stats_proprio)
        batch_input["state"] = (
            torch.from_numpy(proprio_norm.astype(np.float32))
            .unsqueeze(0)
            .to(self.device)
        )

        stream_ctx = torch.cuda.stream(self.vla_stream) if self.vla_stream is not None else nullcontext()
        autocast_ctx = (
            torch.cuda.amp.autocast(dtype=torch.float16, cache_enabled=False)
            if self.use_fp16 and torch.cuda.is_available()
            else nullcontext()
        )

        with torch.no_grad(), stream_ctx, autocast_ctx:
            for key, value in batch_input.items():
                if isinstance(value, torch.Tensor):
                    batch_input[key] = value.to(self.device, non_blocking=True)
            action_out = self.model.predict_action(qwen_inputs=batch_input)

        if self.vla_stream is not None:
            self.vla_stream.synchronize()

        normalized = action_out["normalized_actions"][0]
        if isinstance(normalized, torch.Tensor):
            normalized = normalized.detach().float().cpu().numpy()
        action_cpu = _bounds_unnormalize(np.asarray(normalized, dtype=np.float32), self._norm_stats_action)
        action_gpu = torch.from_numpy(action_cpu).to(
            device=self.device, dtype=torch.float32, non_blocking=True
        )
        output_tok = int(action_cpu.size)
        return action_gpu, output_tok

    def _forward_and_parse_action_gpu(
        self,
        inputs: Dict[str, Any],
        joint_state_gpu: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int]:
        if hasattr(self.model, "predict_action"):
            stream_ctx = torch.cuda.stream(self.vla_stream) if self.vla_stream is not None else nullcontext()
            autocast_ctx = (
                torch.cuda.amp.autocast(dtype=torch.float16, cache_enabled=False)
                if self.use_fp16 and torch.cuda.is_available()
                else nullcontext()
            )
            with torch.no_grad(), stream_ctx, autocast_ctx:
                action = self.model.predict_action(**inputs, do_sample=False)
                action_t = action if isinstance(action, torch.Tensor) else torch.as_tensor(action)
                action_t = action_t.to(device=self.device, dtype=torch.float32)
                output_tok = int(action_t.numel())
                return action_t, output_tok

        # CUDA Graph replay path — single GPU-side replay, no per-kernel CPU dispatch.
        if self._cuda_graph_engine is not None and self._cuda_graph_engine.captured:
            generated = self._cuda_graph_engine.replay(inputs, joint_state=joint_state_gpu)
            output_tok = self.max_new_tokens
        else:
            self._reset_vlm_generation_state()
            stream_ctx = torch.cuda.stream(self.vla_stream) if self.vla_stream is not None else nullcontext()
            autocast_ctx = (
                torch.cuda.amp.autocast(dtype=torch.float16, cache_enabled=False)
                if self.use_fp16 and torch.cuda.is_available()
                else nullcontext()
            )
            with torch.no_grad(), stream_ctx, autocast_ctx:
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
            input_ids = inputs.get("input_ids")
            output_tok = (
                int(generated.shape[-1] - input_ids.shape[-1])
                if input_ids is not None
                else int(generated.shape[-1])
            )

        # Text decode is CPU-bound; runs after graph replay completes on vla_stream.
        if self.vla_stream is not None:
            self.vla_stream.synchronize()

        input_ids = inputs.get("input_ids")
        if input_ids is not None and generated.shape[-1] > input_ids.shape[-1]:
            new_tokens = generated[:, input_ids.shape[-1] :]
            decoded = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        else:
            decoded = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        action_cpu = self._parse_action_text(decoded)
        action_gpu = torch.from_numpy(action_cpu).to(
            device=self.device, dtype=torch.float32, non_blocking=True
        )
        return action_gpu, output_tok

    def _parse_action_text(self, text: str) -> np.ndarray:
        """Parse model-generated text into a float action vector."""
        bracket_groups = re.findall(r"\[([^\]]+)\]", text, flags=re.DOTALL)
        best_numbers: List[str] = []
        for group in bracket_groups:
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", group)
            if len(nums) > len(best_numbers):
                best_numbers = nums

        numbers = best_numbers if best_numbers else re.findall(
            r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text
        )
        if not numbers:
            preview = text[:200].replace("\n", " ")
            raise ValueError(
                f"Model output did not contain numeric action values. Preview: {preview!r}"
            )
        return np.array([float(x) for x in numbers], dtype=np.float32)

    def _reshape_action_gpu(self, action: torch.Tensor) -> torch.Tensor:
        flat = action.flatten()
        if flat.numel() < self.action_dim:
            flat = torch.nn.functional.pad(flat, (0, self.action_dim - flat.numel()))
        return flat[: self.action_dim]

    def _mock_infer(self, image: np.ndarray, instruction: str) -> Tuple[np.ndarray, int, int]:
        action_gpu, n_in, n_out = self._mock_infer_gpu(image, instruction)
        return action_gpu.cpu().numpy(), n_in, n_out

    def _mock_infer_gpu(self, image: np.ndarray, instruction: str) -> Tuple[torch.Tensor, int, int]:
        base_latency = np.random.normal(0.42, 0.05)
        time.sleep(max(0.25, base_latency))
        action = np.random.uniform(-0.5, 0.5, self.action_dim).astype(np.float32)
        action_gpu = torch.from_numpy(action)
        if torch.cuda.is_available():
            action_gpu = action_gpu.pin_memory().to(self.device, non_blocking=True)
        n_input_tok = 280 + len(instruction.split())
        n_output_tok = self.action_dim
        return action_gpu, n_input_tok, n_output_tok


def summarize_action_failures(records: List[InferenceRecord]) -> Dict[str, Any]:
    """Classify failed trials for debugging parse vs numeric issues."""
    failed = [r for r in records if r.failed]
    ok = [r for r in records if not r.failed]
    summary: Dict[str, Any] = {
        "n_failed": len(failed),
        "n_ok": len(ok),
        "failure_modes": {},
    }
    if not ok:
        for r in failed:
            summary["failure_modes"][r.failure_reason] = summary["failure_modes"].get(r.failure_reason, 0) + 1
        return summary

    ok_actions = np.array([r.action for r in ok], dtype=np.float32)
    summary["ok_action_std_mean"] = float(ok_actions.std(axis=0).mean())
    summary["ok_action_nan_rate"] = float(np.isnan(ok_actions).mean())
    summary["ok_unique_actions"] = int(len({tuple(np.round(a, 4)) for a in ok_actions}))

    for r in failed:
        summary["failure_modes"][r.failure_reason] = summary["failure_modes"].get(r.failure_reason, 0) + 1

    return summary


# ── Core profiling routine ───────────────────────────────────
def profile_unifolm_vla0(
    model: UnifoLMVLAWrapper,
    env: MockG1CleanTableEnv,
    n_trials: int = 100,
    instruction: str = "Clean the table by moving all clutter items into the bin.",
    profiler_source: str = "pytorch_profiler",
    run_tag: Optional[str] = None,
) -> Tuple[ProfilingReport, List[InferenceRecord]]:
    records: List[InferenceRecord] = []
    failure_modes: Dict[str, int] = {}
    profiler_source = _normalize_profiler_source(profiler_source)
    run_tag = run_tag or make_profile_run_tag(profiler_source)
    generated_at = datetime.now().isoformat(timespec="seconds")

    gpu_available = torch.cuda.is_available()
    use_nvtx = gpu_available and hasattr(torch.cuda, "nvtx")
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
        if use_nvtx:
            torch.cuda.nvtx.range_push("vla_action_generation")
        try:
            action, input_tok, output_tok = model.infer(img, instruction)
        except Exception as e:
            failed = True
            failure_reason = type(e).__name__
            failure_modes[failure_reason] = failure_modes.get(failure_reason, 0) + 1
            logger.warning(f"  Trial {i} failed ({condition}): {e}")
        finally:
            if use_nvtx:
                torch.cuda.nvtx.range_pop()
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
        profiler_source=profiler_source,
        generated_at=generated_at,
        run_tag=run_tag,
    )
    fail_diag = summarize_action_failures(records)
    logger.info(f"Failure diagnostics: {fail_diag}")
    if model._cuda_graph_engine is not None and model._cuda_graph_engine.captured:
        logger.info(f"CUDA Graph active: mode={model._cuda_graph_engine.capture_mode}")
    else:
        logger.info("CUDA Graph inactive — eager generate path used for all trials")
    return report, records


# ── PyTorch profiler (per-op CPU/GPU trace) ─────────────────
def run_vla_torch_profiler(
    model: UnifoLMVLAWrapper,
    env: MockG1CleanTableEnv,
    instruction: str,
    n_warmup: int = 2,
    n_profile_steps: int = 3,
    profiler_source: str = "pytorch_profiler",
    run_tag: Optional[str] = None,
    results_dir: Optional[Path] = None,
    export_chrome_trace: bool = False,
) -> Dict[str, Any]:
    """
    Trace VLA action generation with torch.profiler.

    Warms up outside the profiler, then records ``n_profile_steps`` inference
    calls inside a profiler context with CPU and (when available) CUDA activity.

    Set ``export_chrome_trace=True`` to write the ~2 GB Chrome trace (default off).
    """
    activities = [torch.profiler.ProfilerActivity.CPU]
    gpu_available = torch.cuda.is_available()
    profiler_source = _normalize_profiler_source(profiler_source)
    run_tag = run_tag or make_profile_run_tag(profiler_source)
    results_dir = results_dir or make_run_results_dir(profiler_source, run_tag)
    generated_at = datetime.now().isoformat(timespec="seconds")
    use_nvtx = gpu_available and hasattr(torch.cuda, "nvtx")
    if gpu_available:
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    obs = env.reset()
    logger.info(
        f"torch.profiler warmup | n_warmup={n_warmup} | n_profile_steps={n_profile_steps} | GPU={gpu_available}"
    )

    # torch.profiler traces eager kernels; disable CUDA Graph capture for this run.
    saved_use_cuda_graph = model.use_cuda_graph
    model.use_cuda_graph = False
    fallback_action = np.zeros(G1_DOF, dtype=np.float32)

    def _profiler_step(obs_in: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        try:
            if use_nvtx:
                torch.cuda.nvtx.range_push("vla_action_generation")
            action, _, _ = model.infer(obs_in, instruction)
            return action, obs_in
        except Exception as exc:
            logger.warning(f"Profiler inference failed: {exc}")
            return fallback_action, obs_in
        finally:
            if use_nvtx:
                torch.cuda.nvtx.range_pop()

    try:
        for _ in range(n_warmup):
            action, _ = _profiler_step(obs)
            if gpu_available:
                torch.cuda.synchronize()
            obs, _, done, _ = env.step(action)
            if done:
                obs = env.reset()

        trace_path = results_dir / "vla_action_chrome_trace.json"
        ops_txt_path = results_dir / "vla_action_profiler_ops.txt"
        ops_json_path = results_dir / "vla_action_profiler_ops.json"
        sort_key = "cuda_time_total" if gpu_available else "cpu_time_total"

        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for _ in range(n_profile_steps):
                action, _ = _profiler_step(obs)
                if gpu_available:
                    torch.cuda.synchronize()
                prof.step()
                obs, _, done, _ = env.step(action)
                if done:
                    obs = env.reset()

        table = prof.key_averages().table(sort_by=sort_key, row_limit=30)
        stack_table = prof.key_averages(group_by_stack_n=5).table(sort_by=sort_key, row_limit=30)
        if export_chrome_trace:
            prof.export_chrome_trace(str(trace_path))
            logger.info(f"torch.profiler trace saved: {trace_path}")
        else:
            logger.info("Chrome trace export skipped (export_chrome_trace=False)")

        with open(ops_txt_path, "w", encoding="utf-8") as f:
            f.write("VLA action generation — torch.profiler key averages\n")
            f.write(f"profiler_source={profiler_source}\n")
            f.write(f"generated_at={generated_at}\n")
            f.write(f"run_tag={run_tag}\n")
            f.write(f"sort_by={sort_key} | n_profile_steps={n_profile_steps}\n")
            f.write("=" * 72 + "\n\n")
            f.write(table)
            f.write("\n\n--- grouped by stack ---\n\n")
            f.write(stack_table)

        op_records = []
        for event in prof.key_averages():
            op_records.append(
                {
                    "name": event.key,
                    "count": event.count,
                    "cpu_time_us": event.cpu_time,
                    "cuda_time_us": getattr(event, "device_time", getattr(event, "cuda_time", 0.0)),
                    "cpu_memory_bytes": getattr(event, "cpu_memory_usage", 0),
                    "cuda_memory_bytes": getattr(
                        event, "device_memory_usage", getattr(event, "cuda_memory_usage", 0)
                    ),
                }
            )
        op_records.sort(
            key=lambda row: row["cuda_time_us"] if gpu_available else row["cpu_time_us"],
            reverse=True,
        )

        profiler_report = {
            "model_id": model.model_id,
            "task": TASK_NAME,
            "instruction": instruction,
            "n_warmup": n_warmup,
            "n_profile_steps": n_profile_steps,
            "gpu_available": gpu_available,
            "profiler_source": profiler_source,
            "generated_at": generated_at,
            "run_tag": run_tag,
            "results_dir": str(results_dir),
            "sort_key": sort_key,
            "chrome_trace": str(trace_path) if export_chrome_trace else None,
            "export_chrome_trace": export_chrome_trace,
            "cuda_graph_captured": bool(
                model._cuda_graph_engine is not None and model._cuda_graph_engine.captured
            ),
            "operations": op_records,
        }
        with open(ops_json_path, "w", encoding="utf-8") as f:
            json.dump(profiler_report, f, indent=2)

        logger.info(f"torch.profiler ops saved: {ops_txt_path}, {ops_json_path}")

        return {
            "profiler": prof,
            "table": table,
            "stack_table": stack_table,
            "trace_path": trace_path,
            "ops_txt_path": ops_txt_path,
            "ops_json_path": ops_json_path,
            "results_dir": results_dir,
            "report": profiler_report,
        }
    finally:
        model.use_cuda_graph = saved_use_cuda_graph


# ── Plotting ─────────────────────────────────────────────────
def plot_profiling_report(
    report: ProfilingReport,
    records: List[InferenceRecord],
    results_dir: Optional[Path] = None,
):
    latencies = [r.latency_ms for r in records if not r.failed]
    trial_idx = [r.trial for r in records if not r.failed]
    inst_hz = [1000.0 / l for l in latencies]

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        f"UnifoLM-VLA-0 Inference Profile — {report.model_id}\n"
        f"Task={report.task} | n={report.n_trials} trials | Source={report.profiler_source} | Week 1–2 Deliverable",
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
        f"Profiler src : {report.profiler_source}\n"
        f"Generated at : {report.generated_at}\n"
        f"Run tag      : {report.run_tag}\n"
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

    results_dir = results_dir or make_run_results_dir(report.profiler_source, report.run_tag)
    out_path = results_dir / "unifolm_vla0_profiling_report.pdf"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.savefig(str(out_path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    logger.info(f"Figure saved: {out_path}")
    plt.close()
    return {"pdf_path": out_path, "png_path": Path(str(out_path).replace(".pdf", ".png"))}


# ── Save JSON log ────────────────────────────────────────────
def save_logs(
    report: ProfilingReport,
    records: List[InferenceRecord],
    results_dir: Optional[Path] = None,
):
    results_dir = results_dir or make_run_results_dir(report.profiler_source, report.run_tag)
    log_path = results_dir / "inference_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records], f, indent=2)

    report_path = results_dir / "profiling_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)

    txt_path = results_dir / "profiling_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("UnifoLM-VLA-0 Profiling Report — Week 1-2 Deliverable\n")
        f.write(f"Model: {report.model_id}\n")
        f.write(f"Task : {report.task}\n")
        f.write(f"Profiler source : {report.profiler_source}\n")
        f.write(f"Generated at    : {report.generated_at}\n")
        f.write(f"Run tag         : {report.run_tag}\n")
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
    return {
        "log_path": log_path,
        "report_path": report_path,
        "summary_path": txt_path,
        "results_dir": results_dir,
    }


# ── Main ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Profile UnifoLM-VLA-0 inference on G1_Clean_Table sim")
    parser.add_argument("--model", type=str, default="unitreerobotics/UnifoLM-VLA-Base")
    parser.add_argument(
        "--vlm-backbone",
        type=str,
        default="unitreerobotics/UnifoLM-VLM-Base",
        help="VLM backbone for UnifoLM-VLA-* checkpoints",
    )
    parser.add_argument(
        "--unnorm-key",
        type=str,
        default="g1_clean_table",
        help="Dataset key in VLA dataset_statistics.json for action denormalization",
    )
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
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable V100 FP16 Tensor Core path (not recommended on Volta)",
    )
    parser.add_argument(
        "--no-cuda-graph",
        action="store_true",
        help="Disable CUDA Graph capture/replay (eager generate fallback)",
    )
    parser.add_argument(
        "--cuda-graph-warmup",
        type=int,
        default=3,
        help="Warmup forward passes before CUDA Graph capture (minimum 3)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  Step 1 — UnifoLM-VLA-0 Profiler | Research Plan Week 1-2")
    logger.info("=" * 60)

    env = MockG1CleanTableEnv(image_size=(224, 224))
    model = UnifoLMVLAWrapper(
        model_id=args.model if not args.mock else "__mock__",
        vlm_backbone_id=args.vlm_backbone,
        unnorm_key=args.unnorm_key,
        use_int4=args.use_int4,
        action_dim=G1_DOF,
        allow_mock_fallback=not args.no_mock_fallback,
        use_fp16=not args.no_fp16,
        use_cuda_graph=not args.no_cuda_graph,
        cuda_graph_warmup=max(args.cuda_graph_warmup, 3),
    )

    profiler_source = "pytorch_profiler"
    run_tag = make_profile_run_tag(profiler_source)
    run_results_dir = make_run_results_dir(profiler_source, run_tag)

    report, records = profile_unifolm_vla0(
        model=model,
        env=env,
        n_trials=args.n_trials,
        instruction=args.instruction,
        profiler_source=profiler_source,
        run_tag=run_tag,
    )

    log_paths = save_logs(report, records, results_dir=run_results_dir)
    fig_paths = plot_profiling_report(report, records, results_dir=run_results_dir)

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
    print(f"  Profiler source: {report.profiler_source}")
    print(f"  Generated at   : {report.generated_at}")
    print(f"  Run tag        : {report.run_tag}")
    print(f"  Report JSON    : {log_paths['report_path']}")
    print(f"  Figure (PNG)   : {fig_paths['png_path']}")
    print(f"\n  Outputs saved to: {run_results_dir.resolve()}")
    print("=" * 60)

    return report


if __name__ == "__main__":
    main()
