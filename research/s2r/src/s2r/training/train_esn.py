"""Train ESN readout from collected episode JSONL shards."""

from __future__ import annotations
from s2r.robot import G1_DOF

import json
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console

from s2r.core.config import load_config
from s2r.nodes.esn_engine import EchoStateNetwork

console = Console()


def _load_episodes(raw_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("episode_*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    return rows


def build_training_arrays(rows: list[dict[str, Any]], n_joints: int) -> tuple[np.ndarray, np.ndarray]:
    """Pair action tokens (input) with subsequent high-rate joint cmds (target).

    Strategy: hold last action token and supervise against joint_cmd samples.
    """
    last_token: np.ndarray | None = None
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for row in rows:
        topic = row.get("topic")
        payload = row.get("payload") or {}
        if topic == "action_token":
            action = payload.get("action")
            if action is not None:
                last_token = np.asarray(action, dtype=np.float64)
        elif topic == "joint_cmd" and last_token is not None:
            q = payload.get("q")
            if q is not None:
                xs.append(last_token[:n_joints])
                ys.append(np.asarray(q[:n_joints], dtype=np.float64))
    if not xs:
        raise RuntimeError("No paired action_token/joint_cmd samples found. Run data collection first.")
    return np.asarray(xs), np.asarray(ys)


def train_esn(config_path: str | None = None, raw_dir: str | None = None, out_path: str | None = None) -> Path:
    cfg = load_config(config_path)
    esn_cfg = cfg.get("esn", {})
    n_joints = int(cfg.get("robot", {}).get("n_joints", G1_DOF))
    raw = Path(raw_dir or cfg.get("data_collection", {}).get("out_dir", "data/raw"))
    out = Path(out_path or esn_cfg.get("model_path", "data/models/esn_upsample.npz"))

    rows = _load_episodes(raw)
    console.print(f"Loaded [cyan]{len(rows)}[/] rows from {raw}")
    X, Y = build_training_arrays(rows, n_joints)
    console.print(f"Training pairs: [cyan]{len(X)}[/]")

    esn = EchoStateNetwork(
        n_inputs=n_joints,
        n_outputs=n_joints,
        reservoir_size=int(esn_cfg.get("reservoir_size", 300)),
        spectral_radius=float(esn_cfg.get("spectral_radius", 0.9)),
        sparsity=float(esn_cfg.get("sparsity", 0.1)),
        input_scale=float(esn_cfg.get("input_scale", 0.5)),
        leaking_rate=float(esn_cfg.get("leaking_rate", 0.3)),
    )
    mse = esn.fit_ridge(X, Y, washout=int(esn_cfg.get("washout", 20)))
    esn.save(out)
    console.print(f"[green]Saved[/] {out}  MSE={mse:.6f}")
    return out


if __name__ == "__main__":
    train_esn()
