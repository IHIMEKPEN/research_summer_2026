"""Factory for vision / reasoner backends from config."""

from __future__ import annotations

from typing import Any

from s2r.models.base import ReasonerBackend, VisionBackend
from s2r.models.qwen_reasoner import QwenReasoner
from s2r.models.qwen_vl import QwenVLBackend
from s2r.models.yolo_detector import YOLODetector


def build_detector(cfg: dict[str, Any]) -> VisionBackend:
    m = cfg.get("models", {}).get("detector", {})
    backend = str(m.get("backend", "yolo")).lower()
    mock = bool(m.get("mock", True))
    device = str(m.get("device", "cuda"))
    if backend == "qwen_vl":
        return QwenVLBackend(
            model_id=str(m.get("model_id", "Qwen/Qwen2.5-VL-3B-Instruct")),
            device=device,
            mock=mock,
            api_base=m.get("api_base"),
            api_key=str(m.get("api_key", "EMPTY")),
            max_new_tokens=int(m.get("max_new_tokens", 128)),
        )
    labels = m.get("target_labels")
    return YOLODetector(
        model_id=str(m.get("model_id", "yolov8n.pt")),
        device=device,
        conf=float(m.get("conf", 0.35)),
        mock=mock,
        target_labels=list(labels) if labels else None,
    )


def build_vlm(cfg: dict[str, Any]) -> VisionBackend:
    m = cfg.get("models", {}).get("vlm", {})
    return QwenVLBackend(
        model_id=str(m.get("model_id", "Qwen/Qwen2.5-VL-3B-Instruct")),
        device=str(m.get("device", "cuda")),
        mock=bool(m.get("mock", True)),
        api_base=m.get("api_base"),
        api_key=str(m.get("api_key", "EMPTY")),
        max_new_tokens=int(m.get("max_new_tokens", 128)),
    )


def build_reasoner(cfg: dict[str, Any]) -> ReasonerBackend:
    m = cfg.get("models", {}).get("reasoner", {})
    backend = str(m.get("backend", "qwen")).lower()
    if backend in {"qwen", "qwen2.5", "llm"}:
        return QwenReasoner(
            model_id=str(m.get("model_id", "Qwen/Qwen2.5-3B-Instruct")),
            device=str(m.get("device", "cuda")),
            mock=bool(m.get("mock", True)),
            api_base=m.get("api_base"),
            api_key=str(m.get("api_key", "EMPTY")),
            max_new_tokens=int(m.get("max_new_tokens", 256)),
            temperature=float(m.get("temperature", 0.2)),
        )
    # fallback
    return QwenReasoner(mock=True)
