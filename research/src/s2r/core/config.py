"""YAML config loader (research/config/)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# research/
_RESEARCH_ROOT = Path(__file__).resolve().parents[3]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        candidates = [
            Path("config/default.yaml"),
            _RESEARCH_ROOT / "config" / "default.yaml",
        ]
        for c in candidates:
            if c.exists():
                path = c
                break
        else:
            raise FileNotFoundError(
                "No config/default.yaml found (run from research/ or pass --config)"
            )
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config root in {path}")
    return data


def deep_get(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
