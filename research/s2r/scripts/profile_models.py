#!/usr/bin/env python3
"""Profile open-source perception / reasoning models on NVIDIA GPUs.

Targets:
  - NVIDIA Tesla V100 (Volta, 16/32GB) — FP16 friendly, no native FP8/FP4
  - NVIDIA Jetson Thor / AGX Thor (Blackwell) — FP4/FP8, large unified memory

Examples:
  python scripts/profile_models.py --device cuda --backend mock --out data/processed/profile_mock.json
  python scripts/profile_models.py --device cuda --yolo yolov8n.pt --reasoner Qwen/Qwen2.5-3B-Instruct
  python scripts/profile_models.py --platform thor --vlm-api http://127.0.0.1:9000/v1
"""

from __future__ import annotations
from s2r.robot import G1_DOF

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np


def gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_total_mem_gb"] = round(props.total_memory / (1024**3), 2)
            info["gpu_capability"] = f"{props.major}.{props.minor}"
    except Exception as e:
        info["torch_error"] = str(e)
    return info


def summarize(samples_ms: list[float]) -> dict[str, float]:
    if not samples_ms:
        return {"n": 0}
    xs = sorted(samples_ms)
    return {
        "n": len(xs),
        "mean_ms": statistics.mean(xs),
        "p50_ms": xs[len(xs) // 2],
        "p95_ms": xs[int(0.95 * (len(xs) - 1))],
        "min_ms": xs[0],
        "max_ms": xs[-1],
        "hz_est": 1000.0 / max(statistics.mean(xs), 1e-6),
    }


def profile_yolo(model_id: str, device: str, mock: bool, iters: int, warmup: int) -> dict[str, Any]:
    from s2r.models.yolo_detector import YOLODetector, encode_image_stub

    det = YOLODetector(model_id=model_id, device=device, mock=mock)
    img = encode_image_stub(480, 640)
    for _ in range(warmup):
        det.infer(img)
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = det.infer(img)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return {"backend": det.name, "latency": summarize(samples), "last_objects": out.objects_of_interest}


def profile_reasoner(model_id: str, device: str, mock: bool, api_base: str | None, iters: int, warmup: int) -> dict[str, Any]:
    from s2r.models.qwen_reasoner import QwenReasoner
    from s2r.models.base import PerceptionFrame, Detection

    reasoner = QwenReasoner(model_id=model_id, device=device, mock=mock, api_base=api_base)
    perc = PerceptionFrame(
        detections=[Detection("table", 0.8, [0.2, 0.4, 0.8, 0.9]), Detection("pen", 0.9, [0.4, 0.5, 0.5, 0.7])],
        caption="pen on table",
        objects_of_interest=["pen", "table"],
    )
    for _ in range(warmup):
        reasoner.plan("bring me a pen", perc, {"holding_pen": False}, "locate")
    samples = []
    out = None
    for _ in range(iters):
        t0 = time.perf_counter()
        out = reasoner.plan("bring me a pen", perc, {"holding_pen": False}, "locate")
        samples.append((time.perf_counter() - t0) * 1000.0)
    return {
        "backend": reasoner.name,
        "latency": summarize(samples),
        "last_intent": getattr(out, "intent", None),
    }


def profile_vlm(model_id: str, device: str, mock: bool, api_base: str | None, iters: int, warmup: int) -> dict[str, Any]:
    from s2r.models.qwen_vl import QwenVLBackend
    from s2r.models.yolo_detector import encode_image_stub

    vlm = QwenVLBackend(model_id=model_id, device=device, mock=mock, api_base=api_base)
    img = encode_image_stub(480, 640)
    for _ in range(warmup):
        vlm.infer(img, "find a pen")
    samples = []
    out = None
    for _ in range(iters):
        t0 = time.perf_counter()
        out = vlm.infer(img, "find a pen")
        samples.append((time.perf_counter() - t0) * 1000.0)
    return {
        "backend": vlm.name,
        "latency": summarize(samples),
        "last_caption": getattr(out, "caption", None),
    }


def profile_esn(iters: int, warmup: int, dim: int = G1_DOF, reservoir: int = 300) -> dict[str, Any]:
    from s2r.nodes.esn_engine import EchoStateNetwork

    esn = EchoStateNetwork(n_inputs=dim, n_outputs=dim, reservoir_size=reservoir)
    u = np.zeros(dim)
    for _ in range(warmup):
        esn.update(u)
    samples = []
    for i in range(iters):
        u = np.sin(np.linspace(0, 1, dim) + 0.01 * i)
        t0 = time.perf_counter()
        esn.update(u)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return {"backend": "esn", "latency": summarize(samples)}


def reference_targets(platform_name: str) -> dict[str, Any]:
    """Planning envelopes (not measured) for deployment sizing."""
    if platform_name == "v100":
        return {
            "gpu": "Tesla V100 16/32GB",
            "notes": [
                "Prefer FP16 / AWQ INT4 for Qwen 3B/7B",
                "No native FP8/FP4 tensor cores",
                "Good for offline training + server-side VLM",
            ],
            "targets": {
                "yolov8n_fp16_ms": [3, 8],
                "qwen2.5_3b_reason_ms": [40, 120],
                "qwen2.5_vl_3b_ms": [80, 250],
                "esn_100hz_budget_ms": 10,
            },
        }
    if platform_name == "thor":
        return {
            "gpu": "Jetson AGX Thor (Blackwell, up to 128GB, ~2070 FP4 TFLOPS)",
            "notes": [
                "Serve Qwen2.5-VL-3B via vLLM with quantization",
                "NVIDIA reports Qwen2.5-VL-3B ~357 tok/s class on Thor (server benchmarks)",
                "Keep ESN on CPU/CUDA small kernel for 50-100Hz control",
            ],
            "targets": {
                "yolov8n_fp16_ms": [2, 6],
                "qwen2.5_3b_reason_ms": [25, 80],
                "qwen2.5_vl_3b_ms": [40, 150],
                "esn_100hz_budget_ms": 10,
            },
        }
    return {"gpu": "generic", "targets": {}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", choices=["auto", "v100", "thor", "generic"], default="auto")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backend", choices=["auto", "mock", "real"], default="auto")
    ap.add_argument("--yolo", default="yolov8n.pt")
    ap.add_argument("--reasoner", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--vlm", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--reasoner-api", default=None)
    ap.add_argument("--vlm-api", default=None)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--skip-vlm", action="store_true")
    ap.add_argument("--skip-reasoner", action="store_true")
    ap.add_argument("--skip-yolo", action="store_true")
    ap.add_argument("--out", default="data/processed/profile_report.json")
    args = ap.parse_args()

    info = gpu_info()
    name = (info.get("gpu_name") or "").lower()
    platform_name = args.platform
    if platform_name == "auto":
        if "v100" in name:
            platform_name = "v100"
        elif "thor" in name:
            platform_name = "thor"
        else:
            platform_name = "generic"

    mock = args.backend == "mock" or (args.backend == "auto" and not info.get("cuda_available"))
    report: dict[str, Any] = {
        "platform": platform_name,
        "gpu_info": info,
        "mock": mock,
        "reference_targets": reference_targets(platform_name),
        "benchmarks": {},
        "ts": time.time(),
    }

    report["benchmarks"]["esn"] = profile_esn(args.iters * 5, args.warmup * 2)

    if not args.skip_yolo:
        report["benchmarks"]["yolo"] = profile_yolo(args.yolo, args.device, mock, args.iters, args.warmup)
    if not args.skip_reasoner:
        report["benchmarks"]["reasoner"] = profile_reasoner(
            args.reasoner, args.device, mock and args.reasoner_api is None, args.reasoner_api, args.iters, args.warmup
        )
    if not args.skip_vlm:
        report["benchmarks"]["vlm"] = profile_vlm(
            args.vlm, args.device, mock and args.vlm_api is None, args.vlm_api, max(5, args.iters // 2), args.warmup
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
