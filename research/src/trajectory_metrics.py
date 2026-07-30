"""Trajectory smoothness metrics (acceleration / jerk) for paper tables."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch


def finite_diff(x: np.ndarray, dt: float) -> np.ndarray:
    """Central-ish first difference along time (axis 0)."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] < 2:
        return np.zeros_like(x)
    d = np.diff(x, axis=0) / dt
    return np.vstack([d[:1], d])


def trajectory_smoothness(
    q: np.ndarray,
    *,
    control_hz: float = 100.0,
) -> Dict[str, float]:
    """
    Report mean |Δq|, mean |accel|, and mean |jerk| (finite differences).

    Jerk uses third difference of joint trajectory: d³q/dt³ ≈ Δ³q / dt³.
    """
    q = np.asarray(q, dtype=np.float64)
    dt = 1.0 / float(control_hz)
    if q.ndim == 1:
        q = q[:, None]
    if q.shape[0] < 4:
        return {"delta_rms": 0.0, "accel_rms": 0.0, "jerk_rms": 0.0}

    dq = np.diff(q, axis=0) / dt
    ddq = np.diff(dq, axis=0) / dt
    dddq = np.diff(ddq, axis=0) / dt
    return {
        "delta_rms": float(np.sqrt(np.mean(dq ** 2))),
        "accel_rms": float(np.sqrt(np.mean(ddq ** 2))),
        "jerk_rms": float(np.sqrt(np.mean(dddq ** 2))),
    }


def compute_jerk_metric(predictions: torch.Tensor, control_hz: float = 100.0) -> float:
    """Backward-compatible proxy used in ESN sweeps (mean squared Δq)."""
    if predictions.shape[0] < 2:
        return 0.0
    diffs = predictions[1:] - predictions[:-1]
    return float(torch.mean(diffs ** 2).item())


def compute_physical_jerk_rms(predictions: torch.Tensor, control_hz: float = 100.0) -> float:
    """RMS of finite-difference jerk (rad/s³) for paper reporting."""
    arr = predictions.detach().cpu().numpy()
    return trajectory_smoothness(arr, control_hz=control_hz)["jerk_rms"]
