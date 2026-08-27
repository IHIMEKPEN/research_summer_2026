"""Budget-matched black-box optimizer comparison for the wipe ESN readout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wipe_esn_experiment import ARM_SLICE, ESN, rollout


@dataclass
class ArmReadoutAdapter:
    """28-D safe search space: 14 arm-row log-scales + 14 biases."""

    base: np.ndarray

    @property
    def dim(self): return 28

    def apply(self, esn: ESN, theta: np.ndarray):
        theta = np.asarray(theta, dtype=np.float32)
        esn.Wout = self.base.copy()
        esn.Wout[ARM_SLICE] *= np.exp(np.clip(theta[:14], -0.25, 0.25))[:, None]
        esn.Wout[ARM_SLICE, -1] += np.clip(theta[14:], -0.20, 0.20)


def _objective(esn, adapter, theta, ep, mjcf, teacher_targets=None):
    adapter.apply(esn, theta)
    # Without a validated real cache, optimizer comparison is task-only.
    weight = 1.0 if teacher_targets is not None else 0.0
    try:
        result = rollout(esn, ep, mjcf, teacher_joint_targets=teacher_targets, teacher_weight=weight)
    except Exception as exc:
        return {
            "L_task": 1e6, "L_teacher": 1e6, "L_total": 1e6,
            "teacher_weight": weight, "teacher_source": "error",
            "anchors": 0, "grasp_success": False, "task_success": False,
            "wipe_path_length_m": 0.0, "table_contact_ratio": 0.0,
            "wipe_coverage_m2": 0.0, "max_cloth_jump_m": 1.0,
            "joint_limit_violation": True, "error": str(exc),
        }
    if not np.isfinite(result["L_total"]):
        result = {**result, "L_task": 1e6, "L_teacher": 1e6, "L_total": 1e6, "task_success": False}
    return result


def random_search(esn, ep, mjcf: Path, *, budget=12, seed=0, teacher_targets=None):
    rng = np.random.default_rng(seed); adapter = ArmReadoutAdapter(esn.Wout.copy())
    best_t = np.zeros(adapter.dim); best = _objective(esn, adapter, best_t, ep, mjcf, teacher_targets)
    history = [{"evaluation": 0, **best}]
    for i in range(1, budget + 1):
        theta = rng.normal(0, 0.05, adapter.dim)
        result = _objective(esn, adapter, theta, ep, mjcf, teacher_targets)
        if result["L_total"] < best["L_total"]: best_t, best = theta, result
        history.append({"evaluation": i, "best_L": best["L_total"], **result})
    adapter.apply(esn, best_t)
    return best_t, best, history


def spsa_adapter(esn, ep, mjcf: Path, *, budget=12, seed=0, teacher_targets=None):
    rng = np.random.default_rng(seed); adapter = ArmReadoutAdapter(esn.Wout.copy())
    theta = np.zeros(adapter.dim); best_t = theta.copy()
    best = _objective(esn, adapter, theta, ep, mjcf, teacher_targets)
    history = [{"evaluation": 0, **best}]; used = 0
    while used + 2 <= budget:
        k = used // 2 + 1; delta = rng.choice([-1.0, 1.0], adapter.dim)
        c, a = 0.04 / k**0.101, 0.015 / k**0.602
        rp = _objective(esn, adapter, theta + c*delta, ep, mjcf, teacher_targets)
        rm = _objective(esn, adapter, theta - c*delta, ep, mjcf, teacher_targets)
        used += 2
        theta -= a * ((rp["L_total"] - rm["L_total"]) / (2*c)) * delta
        theta = np.clip(theta, -0.2, 0.2)
        candidate = _objective(esn, adapter, theta, ep, mjcf, teacher_targets)
        if candidate["L_total"] < best["L_total"]: best_t, best = theta.copy(), candidate
        history.append({"evaluation": used, "best_L": best["L_total"], **candidate})
    adapter.apply(esn, best_t)
    return best_t, best, history


def cem(esn, ep, mjcf: Path, *, budget=12, seed=0, teacher_targets=None, population=4):
    rng = np.random.default_rng(seed); adapter = ArmReadoutAdapter(esn.Wout.copy())
    mean = np.zeros(adapter.dim); std = np.full(adapter.dim, 0.06)
    best_t = mean.copy(); best = _objective(esn, adapter, best_t, ep, mjcf, teacher_targets)
    history = [{"evaluation": 0, **best}]; used = 0
    while used < budget:
        n = min(population, budget-used)
        samples = np.clip(rng.normal(mean, std, (n, adapter.dim)), -0.2, 0.2)
        scored = []
        for theta in samples:
            result = _objective(esn, adapter, theta, ep, mjcf, teacher_targets)
            used += 1; scored.append((result["L_total"], theta, result))
            if result["L_total"] < best["L_total"]: best_t, best = theta.copy(), result
            history.append({"evaluation": used, "best_L": best["L_total"], **result})
        scored.sort(key=lambda x: x[0]); elite = np.stack([x[1] for x in scored[:max(1,n//2)]])
        mean = elite.mean(axis=0); std = np.maximum(elite.std(axis=0), 0.01)
    adapter.apply(esn, best_t)
    return best_t, best, history


def cmaes_lite(esn, ep, mjcf: Path, *, budget=12, seed=0, teacher_targets=None, population=4):
    """Budget-matched CMA-ES-style search (rank-μ mean/covariance update, no extra deps)."""
    rng = np.random.default_rng(seed)
    adapter = ArmReadoutAdapter(esn.Wout.copy())
    dim = adapter.dim
    mean = np.zeros(dim, dtype=np.float64)
    sigma = 0.06
    cov = np.eye(dim, dtype=np.float64)
    best_t = mean.copy()
    best = _objective(esn, adapter, best_t, ep, mjcf, teacher_targets)
    history = [{"evaluation": 0, **best}]
    used = 0
    while used < budget:
        n = min(population, budget - used)
        samples = []
        scored = []
        for _ in range(n):
            z = rng.multivariate_normal(np.zeros(dim), cov)
            theta = np.clip(mean + sigma * z, -0.2, 0.2)
            result = _objective(esn, adapter, theta, ep, mjcf, teacher_targets)
            used += 1
            scored.append((result["L_total"], theta, result))
            samples.append(theta)
            if result["L_total"] < best["L_total"]:
                best_t, best = theta.copy(), result
            history.append({"evaluation": used, "best_L": best["L_total"], **result})
        scored.sort(key=lambda x: x[0])
        elite_n = max(1, n // 2)
        elite = np.stack([x[1] for x in scored[:elite_n]])
        mean = elite.mean(axis=0)
        centered = elite - mean
        cov = (centered.T @ centered) / max(elite_n, 1) + 1e-4 * np.eye(dim)
        # Keep step-size from collapsing under tiny budgets.
        sigma = float(np.clip(0.9 * sigma + 0.1 * np.mean(np.linalg.norm(centered, axis=1)), 0.01, 0.12))
    adapter.apply(esn, best_t)
    return best_t, best, history

