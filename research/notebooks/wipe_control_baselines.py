"""Achievable control baselines for the stabilized wipe benchmark.

Run these *before* another independent-ESN optimizer comparison. If oracle/PD
replay cannot satisfy the strict objective, the environment or thresholds are
invalid — a student controller cannot beat a failing oracle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from wipe_esn_experiment import (
    DT,
    G1_DOF,
    TEACHER_PERIOD,
    control_grid,
    rollout_policy,
    teacher_cache,
)


def stationary_hold(ep: dict[str, np.ndarray]) -> Callable:
    q0 = ep["q"][0].astype(np.float32)

    def command_fn(_t, _q, _qd):
        return q0.copy()

    return command_fn


def dataset_oracle_zoh(ep: dict[str, np.ndarray]) -> Callable:
    """Hold each native demo sample until the next (zero-order hold)."""
    rel = ep["t"] - ep["t"][0]
    q = ep["q"]

    def command_fn(t, _q, _qd):
        idx = int(np.searchsorted(rel, t, side="right") - 1)
        idx = int(np.clip(idx, 0, len(q) - 1))
        return q[idx].astype(np.float32)

    return command_fn


def dataset_oracle_linear(ep: dict[str, np.ndarray]) -> Callable:
    """Linear interpolation of demonstrated joint trajectories at 100 Hz."""
    rel = ep["t"] - ep["t"][0]
    q = ep["q"]

    def command_fn(t, _q, _qd):
        out = np.empty(G1_DOF, dtype=np.float32)
        for j in range(G1_DOF):
            out[j] = np.interp(t, rel, q[:, j])
        return out

    return command_fn


def dataset_oracle_pd_track(ep: dict[str, np.ndarray]) -> Callable:
    """PD tracking of the linearly interpolated demonstration (command = target).

    The shared rollout already applies joint PD torque tracking of ``cmd``.
    """
    return dataset_oracle_linear(ep)


def sparse_teacher_zoh(ep: dict[str, np.ndarray], targets: np.ndarray | None = None) -> Callable:
    """Zero-order hold of sparse 570 ms joint targets (demo proxy or real cache)."""
    times, proxy = teacher_cache(ep)
    q = proxy if targets is None else np.asarray(targets, dtype=np.float32)
    if len(q) != len(times):
        raise ValueError(f"sparse teacher length {len(q)} != {len(times)}")

    def command_fn(t, _q, _qd):
        idx = int(np.searchsorted(times, t + 1e-9, side="right") - 1)
        idx = int(np.clip(idx, 0, len(q) - 1))
        return q[idx].copy()

    return command_fn


def run_baseline(
    name: str,
    ep: dict[str, np.ndarray],
    mjcf: Path,
    *,
    press_table: bool = False,
    teacher_joint_targets=None,
    teacher_weight: float = 0.0,
    use_contact_layer: bool | None = None,
):
    factories = {
        "stationary": stationary_hold,
        "oracle_zoh": dataset_oracle_zoh,
        "oracle_linear": dataset_oracle_linear,
        "oracle_pd": dataset_oracle_pd_track,
        "sparse_teacher_zoh": sparse_teacher_zoh,
        "oracle_pd_contact": dataset_oracle_pd_track,
    }
    if name not in factories:
        raise KeyError(f"Unknown baseline {name}; choose from {sorted(factories)}")
    if name == "sparse_teacher_zoh":
        fn = factories[name](ep, teacher_joint_targets)
    else:
        fn = factories[name](ep)
    contact = None
    want_contact = use_contact_layer if use_contact_layer is not None else (name == "oracle_pd_contact")
    if want_contact:
        from wipe_contact_controller import ContactImpedanceController
        contact = ContactImpedanceController()
    return rollout_policy(
        fn, ep, mjcf,
        teacher_joint_targets=teacher_joint_targets if teacher_weight > 0 else None,
        teacher_weight=teacher_weight,
        press_table=press_table,
        policy_name=name,
        contact_controller=contact,
    )


BASELINE_NAMES = (
    "stationary",
    "oracle_zoh",
    "oracle_linear",
    "oracle_pd",
    "oracle_pd_contact",
    "sparse_teacher_zoh",
)
