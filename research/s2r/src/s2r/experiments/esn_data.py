"""Prepare ESN train/val arrays from raw episodes or benchmark task folders."""

from __future__ import annotations
from s2r.robot import G1_DOF

import json
from pathlib import Path
from typing import Any

import numpy as np

from s2r.experiments.paths import ESN_DATA, RAW
from s2r.training.train_esn import build_training_arrays


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def collect_episode_rows(sources: list[Path] | None = None) -> list[dict[str, Any]]:
    if sources is None:
        sources = [RAW]
    rows: list[dict[str, Any]] = []
    for src in sources:
        src = Path(src)
        if src.is_file() and src.suffix == ".jsonl":
            rows.extend(iter_jsonl(src))
            continue
        if src.is_dir():
            for path in sorted(src.rglob("*.jsonl")):
                rows.extend(iter_jsonl(path))
    return rows


def export_esn_split(
    n_joints: int = G1_DOF,
    sources: list[Path] | None = None,
    val_ratio: float = 0.2,
    seed: int = 0,
) -> dict[str, Path]:
    rows = collect_episode_rows(sources)
    X, Y = build_training_arrays(rows, n_joints=n_joints)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = max(1, int(len(X) * val_ratio)) if len(X) > 5 else 1
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    if len(train_idx) == 0:
        train_idx = idx
    train_dir = ESN_DATA / "train"
    val_dir = ESN_DATA / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    x_train, y_train = X[train_idx], Y[train_idx]
    x_val, y_val = X[val_idx], Y[val_idx]
    np.savez_compressed(train_dir / "pairs.npz", X=x_train, Y=y_train)
    np.savez_compressed(val_dir / "pairs.npz", X=x_val, Y=y_val)
    meta = {
        "n_train": int(len(x_train)),
        "n_val": int(len(x_val)),
        "n_joints": n_joints,
        "sources": [str(s) for s in (sources or [RAW])],
    }
    (ESN_DATA / "split_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {
        "train": train_dir / "pairs.npz",
        "val": val_dir / "pairs.npz",
        "meta": ESN_DATA / "split_meta.json",
    }


def load_pairs(split: str = "train") -> tuple[np.ndarray, np.ndarray]:
    path = ESN_DATA / split / "pairs.npz"
    data = np.load(path)
    return data["X"], data["Y"]
