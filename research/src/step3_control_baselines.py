"""
============================================================
Step 3 — Control baselines for closed-loop comparison
Phase 1 ICRA plan: Pure VLA ZOH + linear interpolation (+ optional PID)
Research Plan: VLA + ESN for Real-Time Humanoid Control
Author: Osemudiamen Andrew Ihimekpen | PVAMU CREDIT Center
============================================================

Upsamples sparse ~2 Hz VLA joint targets to 100 Hz without an ESN.

Usage (from research/):
  python3 -m src.step3_control_baselines --method zoh --episode 0
  python3 -m src.step3_control_baselines --method linear --episode 0
  python3 -m src.step3_control_baselines --all --episode 0
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Literal, Sequence

import numpy as np

from src.g1_constants import CONTROL_HZ, G1_DOF, VLA_HZ
from src.paths import results_path
from src.step2_esn_cuda_ridge import (
    evaluate_predictions,
    load_episode_trajectory_numpy,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = results_path("step1_baselines")
Method = Literal["zoh", "linear", "pid"]
ALL_METHODS: Sequence[Method] = ("zoh", "linear", "pid")


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


def _sparse_knot_indices(t_len: int, vla_hz: float, control_hz: float) -> np.ndarray:
    hold = max(1, int(round(control_hz / vla_hz)))
    idx = np.arange(0, t_len, hold, dtype=np.int64)
    if idx.size == 0 or int(idx[-1]) != t_len - 1:
        idx = np.concatenate([idx, np.asarray([t_len - 1], dtype=np.int64)])
    return np.unique(idx)


def upsample_zoh(vla_targets: np.ndarray) -> np.ndarray:
    """Pass through VLA targets already ZOH-held on the 100 Hz grid."""
    return np.asarray(vla_targets, dtype=np.float32)


def upsample_linear(
    vla_targets: np.ndarray,
    vla_hz: float = VLA_HZ,
    control_hz: float = CONTROL_HZ,
) -> np.ndarray:
    """
    Piecewise-linear interpolation between sparse VLA sample instants.

    Knots are taken from ``vla_targets`` at every hold boundary (not GT),
    then linearly blended on the 100 Hz grid. This is the fair baseline
    against ESN: both see only sparse VLA intents.
    """
    vla = np.asarray(vla_targets, dtype=np.float32)
    t_len = int(vla.shape[0])
    idx = _sparse_knot_indices(t_len, vla_hz=vla_hz, control_hz=control_hz)
    knots = vla[idx]
    out = np.zeros_like(vla, dtype=np.float32)
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


def online_linear_command(
    *,
    prev_token: np.ndarray,
    curr_token: np.ndarray,
    ticks_since_update: int,
    hold_ticks: int,
) -> np.ndarray:
    """
    Causal online linear bridge for the dual-process loop.

    After each new VLA token, ramp from the previous token toward the
    current one over ``hold_ticks`` (≈ control_hz / vla_hz). Once the
    ramp finishes, hold the current token (ZOH tail).
    """
    hold = max(1, int(hold_ticks))
    alpha = min(1.0, float(ticks_since_update) / float(hold))
    prev = np.asarray(prev_token, dtype=np.float32).reshape(-1)
    curr = np.asarray(curr_token, dtype=np.float32).reshape(-1)
    return ((1.0 - alpha) * prev + alpha * curr).astype(np.float32)


def run_baseline(method: Method, episode: int) -> BaselineResult:
    gt, vla = load_episode_trajectory_numpy(episode)
    if method == "zoh":
        pred = upsample_zoh(vla)
    elif method == "linear":
        pred = upsample_linear(vla)
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


def run_all_baselines(episode: int, methods: Sequence[Method] = ALL_METHODS) -> List[BaselineResult]:
    return [run_baseline(m, episode) for m in methods]


def write_comparison_table(results: List[BaselineResult], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(r) for r in results]
    json_path = out_dir / "baseline_comparison.json"
    json_path.write_text(json.dumps(rows, indent=2))

    csv_path = out_dir / "baseline_comparison.csv"
    header = "method,episode,rmse,mse,jerk,jerk_rms,control_hz,vla_hz\n"
    body = "".join(
        f"{r.method},{r.episode},{r.rmse:.8f},{r.mse:.8e},{r.jerk:.8e},"
        f"{r.jerk_rms:.8e},{r.control_hz},{r.vla_hz}\n"
        for r in results
    )
    csv_path.write_text(header + body)
    logger.info("Wrote comparison table: %s", csv_path)
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="VLA control baselines (ZOH / linear / PID)")
    parser.add_argument("--method", choices=["zoh", "linear", "pid"], default="zoh")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run zoh + linear + pid and write baseline_comparison.{json,csv}",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.all:
        results = run_all_baselines(args.episode)
        for result in results:
            out = RESULTS_DIR / f"baseline_{result.method}_ep{args.episode}.json"
            out.write_text(json.dumps(asdict(result), indent=2))
            logger.info(
                "%s ep=%d RMSE=%.6f jerk=%.6e → %s",
                result.method,
                result.episode,
                result.rmse,
                result.jerk,
                out,
            )
        write_comparison_table(results, RESULTS_DIR)
        print(json.dumps([asdict(r) for r in results], indent=2))
        return

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
