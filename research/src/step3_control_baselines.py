"""
============================================================
Step 3 — Control baselines for closed-loop comparison
Phase 1 ICRA plan: Pure VLA ZOH + linear interpolation (+ optional PID)
Research Plan: VLA + ESN for Real-Time Humanoid Control
Author: Osemudiamen Andrew Ihimekpen | PVAMU CREDIT Center
============================================================

Upsamples sparse 2 Hz VLA joint targets to 100 Hz without an ESN.

Usage (from research/):
  python3 -m src.step3_control_baselines --method zoh --episode 0
  python3 -m src.step3_control_baselines --method linear --episode 0
  python3 -m src.step3_control_baselines --method pid --episode 0
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from src.g1_constants import CONTROL_HZ, G1_DOF, VLA_HZ
from src.paths import results_path
from src.step2_esn_cuda_ridge import (
    compute_jerk_metric,
    evaluate_predictions,
    load_episode_trajectory_numpy,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = results_path("step1_baselines")
Method = Literal["zoh", "linear", "pid"]


@dataclass
class BaselineResult:
    method: str
    episode: int
    mse: float
    rmse: float
    jerk: float
    jerk_rms: float
    control_hz: float
    vla_hz: float
    n_joints: int = G1_DOF


def upsample_zoh(vla_targets: np.ndarray) -> np.ndarray:
    """VLA targets are already ZOH-held on the 100 Hz grid in the dataloader."""
    return np.asarray(vla_targets, dtype=np.float32)


def upsample_linear(gt: np.ndarray, vla_hz: float = VLA_HZ, control_hz: float = CONTROL_HZ) -> np.ndarray:
    """Piecewise-linear interpolation between sparse VLA sample instants."""
    t_len = gt.shape[0]
    hold = max(1, int(round(control_hz / vla_hz)))
    idx = np.arange(0, t_len, hold)
    if idx[-1] != t_len - 1:
        idx = np.concatenate([idx, [t_len - 1]])
    knots = gt[idx]
    out = np.zeros_like(gt, dtype=np.float32)
    for i in range(len(idx) - 1):
        a, b = int(idx[i]), int(idx[i + 1])
        span = max(1, b - a)
        for k in range(a, b + 1):
            alpha = (k - a) / span
            out[k] = (1.0 - alpha) * knots[i] + alpha * knots[i + 1]
    return out


def upsample_pid(
    gt: np.ndarray,
    vla_targets: np.ndarray,
    kp: float = 0.4,
    kd: float = 0.05,
) -> np.ndarray:
    """Simple joint-space PID toward held VLA targets (baseline, not ESN)."""
    q = np.asarray(vla_targets[0], dtype=np.float64).copy()
    dq = np.zeros_like(q)
    out = np.zeros_like(gt, dtype=np.float32)
    dt = 1.0 / CONTROL_HZ
    for t in range(gt.shape[0]):
        err = vla_targets[t] - q
        dq = kd * (err / dt) if t else dq
        q = q + dt * (kp * err + dq)
        out[t] = q.astype(np.float32)
    return out


def run_baseline(method: Method, episode: int) -> BaselineResult:
    gt, vla = load_episode_trajectory_numpy(episode)
    if method == "zoh":
        pred = upsample_zoh(vla)
    elif method == "linear":
        pred = upsample_linear(gt)
    elif method == "pid":
        pred = upsample_pid(gt, vla)
    else:
        raise ValueError(method)

    import torch

    metrics = evaluate_predictions(
        torch.from_numpy(pred),
        torch.from_numpy(gt),
    )
    return BaselineResult(
        method=method,
        episode=episode,
        mse=metrics["mse"],
        rmse=metrics["rmse"],
        jerk=metrics["jerk"],
        jerk_rms=metrics.get("jerk_rms", 0.0),
        control_hz=CONTROL_HZ,
        vla_hz=VLA_HZ,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="VLA control baselines (ZOH / linear / PID)")
    parser.add_argument("--method", choices=["zoh", "linear", "pid"], default="zoh")
    parser.add_argument("--episode", type=int, default=0)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = run_baseline(args.method, args.episode)
    out = RESULTS_DIR / f"baseline_{args.method}_ep{args.episode}.json"
    out.write_text(json.dumps(asdict(result), indent=2))
    logger.info(
        "%s ep=%d RMSE=%.6f jerk=%.6e → %s",
        result.method,
        result.episode,
        result.rmse,
        result.jerk,
        out,
    )
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
