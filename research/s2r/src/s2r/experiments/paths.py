"""Canonical project / data paths for notebooks and scripts."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
NOTEBOOKS = ROOT / "notebooks"
CONFIG = ROOT / "config"

BENCHMARK = DATA / "benchmark"
BENCHMARK_TASKS = BENCHMARK / "tasks"
BENCHMARK_RESULTS = BENCHMARK / "results"
UNITREE_VLA = DATA / "unitree_vla"
ESN_DATA = DATA / "esn"
RAW = DATA / "raw"
MODELS = DATA / "models"
PROCESSED = DATA / "processed"


def ensure_experiment_dirs() -> dict[str, Path]:
    paths = {
        "root": ROOT,
        "data": DATA,
        "benchmark_tasks": BENCHMARK_TASKS,
        "benchmark_results": BENCHMARK_RESULTS,
        "unitree_vla": UNITREE_VLA,
        "esn_train": ESN_DATA / "train",
        "esn_val": ESN_DATA / "val",
        "esn_checkpoints": ESN_DATA / "checkpoints",
        "esn_curves": ESN_DATA / "curves",
        "raw": RAW,
        "models": MODELS,
        "processed": PROCESSED,
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths
