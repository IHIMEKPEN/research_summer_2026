"""
============================================================
Step 2 — CUDA Echo State Network + Ridge Regression
Phase 2: Bridge 2 Hz VLA tokens → 100 Hz G1 joint tracking
Research Plan: VLA + ESN for Real-Time Humanoid Control
Author: Osemudiamen Andrew Ihimekpen | PVAMU CREDIT Center
============================================================

Loads unitreerobotics/G1_Dex1_Wipe_Table, resamples proprioception to
100 Hz, simulates sparse 2 Hz VLA target tokens, sweeps leaky_rate and
ridge λ for low MSE + low jerk, and saves the best smoothed readout.

Usage (from research/):
  python3 -m src.step2_esn_cuda_ridge
  python3 -m src.step2_esn_cuda_ridge --episode 0 --reservoir_size 2000
  # Multi-episode (canonical split: train 0-159, held-out 160-199):
  python3 -m src.step2_esn_cuda_ridge --train_episodes train --heldout_episodes heldout
  python3 -m src.step2_esn_cuda_ridge --train_episodes 0-15 --heldout_episodes 160-163  # smoke
  # Per-task / full UnifoLM suite (one checkpoint folder per task):
  python3 -m src.step2_esn_cuda_ridge --task clean_table
  python3 -m src.step2_esn_cuda_ridge --all_tasks --continue_on_error
  python3 -m src.step2_esn_cuda_ridge --tasks wipe_table,stack_block,fold_towel

Inference:
  from src.step2_esn_cuda_ridge import load_checkpoint, load_esn_for_task
  esn = load_esn_for_task("wipe_table")           # models/esn_cuda_ridge/
  esn = load_esn_for_task("stack_block")          # models/esn_cuda_ridge_stack_block/
  esn.update_vla_target(vla_joint_target)         # 2 Hz
  cmd = esn.step_proprio(current_joint_state)     # 100 Hz
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset

from src.paths import models_path, results_path
from src.trajectory_metrics import compute_jerk_metric, compute_physical_jerk_rms
from src.wipe_dataset import HELDOUT_EPISODES, TRAIN_EPISODES, parse_episode_spec

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────
G1_DOF = 29
CONTROL_HZ = 100.0
VLA_HZ = 2.0
VLA_HOLD_STEPS = int(CONTROL_HZ / VLA_HZ)  # 50 steps between sparse VLA updates
DATASET_ID = "unitreerobotics/G1_Dex1_Wipe_Table"
CHECKPOINT_VERSION = 2
CHECKPOINT_BASENAME = "esn_cuda_ridge"
BEST_CHECKPOINT_BASENAME = "esn_cuda_ridge_best"

# Hyperparameter sweep grid (leaky integrator α × ridge λ)
SWEEP_LEAKY_RATES = (0.05, 0.1, 0.3)
SWEEP_RIDGE_ALPHAS = (1e-4, 1e-2, 1.0)

JOINT_STATE_KEYS = (
    "observation.body",  # 29-D: legs + waist [0:15], arms [15:29]
)


# ── Configuration ─────────────────────────────────────────────
@dataclass
class ESNCudaConfig:
  reservoir_size: int = 1000
  spectral_radius: float = 0.95
  sparsity: float = 0.90
  leaky_rate: float = 0.1          # sluggish reservoir → low-pass filter
  input_scaling: float = 1.0
  ridge_alpha: float = 1e-2        # stronger Tikhonov penalty → smoother readout
  washout: int = 50
  seed: int = 42
  joint_dim: int = G1_DOF
  vla_token_dim: int = G1_DOF
  output_dim: int = G1_DOF

  @property
  def input_dim(self) -> int:
    return self.joint_dim + self.vla_token_dim

  @property
  def readout_dim(self) -> int:
    return self.reservoir_size + self.input_dim


# ── Data loading ──────────────────────────────────────────────
def extract_joint_state_29d(row: Dict) -> torch.Tensor:
    """
    Build the 29-DoF continuous joint vector from LeRobot G1 rows.

    Preferred layout (G1_Dex1_Wipe_Table and most Unitree G1 demos):
      body[0:15]  legs + waist
      body[15:22] left arm  == observation.left_arm (when present)
      body[22:29] right arm == observation.right_arm (when present)

    Dex1 / multi-finger channels (``observation.*_gripper``) are separate
    end-effector signals and are **not** part of this 29-DoF tracking vector.

    Falls back to ``observation.body`` alone when arm keys are missing but
    body is already 29-D (other UnifoLM G1 task corpora).
    """
    if "observation.body" not in row:
        raise KeyError("Row missing observation.body (required for 29-DoF ESN)")

    body = torch.as_tensor(row["observation.body"], dtype=torch.float32).flatten()
    has_arms = "observation.left_arm" in row and "observation.right_arm" in row
    if has_arms:
        left_arm = torch.as_tensor(row["observation.left_arm"], dtype=torch.float32).flatten()
        right_arm = torch.as_tensor(row["observation.right_arm"], dtype=torch.float32).flatten()
        if left_arm.numel() == 7 and right_arm.numel() == 7 and body.numel() >= 15:
            state = torch.cat([body[:15], left_arm, right_arm], dim=0)
        elif body.numel() == G1_DOF:
            state = body
        else:
            raise ValueError(
                f"Cannot build 29-DoF state from body={body.numel()} "
                f"left_arm={left_arm.numel()} right_arm={right_arm.numel()}"
            )
    elif body.numel() == G1_DOF:
        state = body
    else:
        raise ValueError(
            f"Expected observation.body with {G1_DOF} joints "
            f"(or body[:15]+arms), got {body.numel()}"
        )

    if state.numel() != G1_DOF:
        raise ValueError(f"Expected {G1_DOF} joints, got {state.numel()}")
    return state


def linear_resample(
  values: torch.Tensor,
  t_src: torch.Tensor,
  t_dst: torch.Tensor,
) -> torch.Tensor:
  """Linearly resample (T_src, D) onto t_dst (T_dst,)."""
  t_src = t_src.flatten()
  t_dst = t_dst.flatten()
  idx = torch.searchsorted(t_src, t_dst, right=True).clamp(1, t_src.numel() - 1)
  t0 = t_src[idx - 1]
  t1 = t_src[idx]
  y0 = values[idx - 1]
  y1 = values[idx]
  alpha = ((t_dst - t0) / (t1 - t0 + 1e-12)).unsqueeze(-1)
  return y0 + alpha * (y1 - y0)


def resample_episode_to_hz(
  states: torch.Tensor,
  timestamps: torch.Tensor,
  target_hz: float,
) -> torch.Tensor:
  """Upsample / downsample an episode to a uniform target_hz grid."""
  duration = (timestamps[-1] - timestamps[0]).item()
  n_out = max(2, int(round(duration * target_hz)) + 1)
  t_dst = torch.linspace(timestamps[0], timestamps[-1], n_out, dtype=torch.float32)
  return linear_resample(states, timestamps, t_dst)


def build_vla_target_hold(
  ground_truth_100hz: torch.Tensor,
  hold_steps: int = VLA_HOLD_STEPS,
) -> torch.Tensor:
  """
  Simulate sparse 2 Hz VLA output via zero-order hold on the 100 Hz grid.

  A new target token is latched every `hold_steps` ticks (50 → 2 Hz).
  """
  t_len = ground_truth_100hz.shape[0]
  sparse_idx = torch.arange(0, t_len, hold_steps, device=ground_truth_100hz.device)
  sparse_tokens = ground_truth_100hz[sparse_idx]
  counts = torch.diff(
    torch.cat([sparse_idx, torch.tensor([t_len], device=sparse_idx.device)]),
  )
  return torch.repeat_interleave(sparse_tokens, counts.long(), dim=0)


def load_episode_trajectory_numpy(
  episode_index: int,
  control_hz: float = CONTROL_HZ,
  vla_hz: float = VLA_HZ,
  dataset_id: str = DATASET_ID,
) -> Tuple[np.ndarray, np.ndarray]:
  """
  Load episode joint GT and 2 Hz zero-order-held VLA tokens on CPU (numpy).

  Returns:
    ground_truth: (T, 29) float32
    vla_targets:  (T, 29) float32
  """
  from datasets import load_dataset

  dataset = load_dataset(dataset_id, split="train")
  rows = [row for row in dataset if row["episode_index"] == episode_index]
  if not rows:
    raise ValueError(f"Episode {episode_index} not found in {dataset_id}")

  rows.sort(key=lambda r: r["frame_index"])
  raw_states = torch.stack([extract_joint_state_29d(r) for r in rows], dim=0)
  timestamps = torch.tensor([r["timestamp"] for r in rows], dtype=torch.float32)

  gt = resample_episode_to_hz(raw_states, timestamps, control_hz)
  hold_steps = max(1, int(round(control_hz / vla_hz)))
  vla = build_vla_target_hold(gt, hold_steps=hold_steps)
  return gt.numpy().astype(np.float32), vla.numpy().astype(np.float32)


def load_episode_gripper_trajectory_numpy(
  episode_index: int,
  control_hz: float = CONTROL_HZ,
  dataset_id: str = DATASET_ID,
) -> Tuple[np.ndarray, np.ndarray]:
  """Load Dex1 left/right gripper scalars resampled to ``control_hz``."""
  from datasets import load_dataset

  dataset = load_dataset(dataset_id, split="train")
  rows = [row for row in dataset if row["episode_index"] == episode_index]
  if not rows:
    raise ValueError(f"Episode {episode_index} not found in {dataset_id}")

  rows.sort(key=lambda r: r["frame_index"])
  left = torch.tensor(
    [float(r["observation.left_gripper"]) for r in rows], dtype=torch.float32
  )
  right = torch.tensor(
    [float(r["observation.right_gripper"]) for r in rows], dtype=torch.float32
  )
  timestamps = torch.tensor([r["timestamp"] for r in rows], dtype=torch.float32)

  left_hz = resample_episode_to_hz(left.unsqueeze(-1), timestamps, control_hz).squeeze(-1)
  right_hz = resample_episode_to_hz(right.unsqueeze(-1), timestamps, control_hz).squeeze(-1)
  return left_hz.numpy().astype(np.float32), right_hz.numpy().astype(np.float32)


def load_episode_tensors(
  dataset,
  episode_index: int,
  device: torch.device,
  control_hz: float = CONTROL_HZ,
) -> Tuple[torch.Tensor, torch.Tensor]:
  """
  Load one episode, resample joint states to control_hz, and build VLA holds.

  Returns:
    ground_truth: (T, 29) joint states at control_hz
    vla_targets:  (T, 29) zero-order-held sparse VLA tokens at control_hz
  """
  rows = [row for row in dataset if row["episode_index"] == episode_index]
  if not rows:
    raise ValueError(f"Episode {episode_index} not found in dataset")

  rows.sort(key=lambda r: r["frame_index"])
  raw_states = torch.stack([extract_joint_state_29d(r) for r in rows], dim=0)
  timestamps = torch.tensor([r["timestamp"] for r in rows], dtype=torch.float32)

  gt = resample_episode_to_hz(raw_states, timestamps, control_hz)
  vla = build_vla_target_hold(gt)
  return gt.to(device), vla.to(device)


# ── CUDA ESN ──────────────────────────────────────────────────
class EchoStateNetwork(nn.Module):
  """
  Leaky ESN with a fixed sparse reservoir on CUDA.

  State update (leaky integrator):
    x(t) = (1 - α) x(t-1) + α tanh(W_res x(t-1) + W_in u(t))

  Input u(t) = [joint_state_t ; vla_target_t]  ∈ R^{58}
  Readout (trained offline): y(t) = W_out [x(t); u(t)]
  """

  def __init__(self, cfg: ESNCudaConfig, device: torch.device):
    super().__init__()
    self.cfg = cfg
    self.device = device
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed)

    self.register_buffer("x", torch.zeros(cfg.reservoir_size, device=device))
    self.register_buffer("vla_target", torch.zeros(cfg.vla_token_dim, device=device))
    self._step_count = 0
    self.W_res = self._init_sparse_reservoir(generator)
    self.W_in = (
      torch.randn(cfg.reservoir_size, cfg.input_dim, generator=generator, device=device)
      * cfg.input_scaling
    )
    self.W_out: Optional[torch.Tensor] = None

  def _init_sparse_reservoir(self, generator: torch.Generator) -> torch.Tensor:
    """Random sparse reservoir scaled to the requested spectral radius."""
    n = self.cfg.reservoir_size
    dense = torch.randn(n, n, generator=generator, device=self.device)
    keep = torch.rand(n, n, generator=generator, device=self.device) > self.cfg.sparsity
    dense = dense * keep

    with torch.no_grad():
      rho = torch.linalg.eigvals(dense).abs().max()
      if rho > 1e-10:
        dense = dense * (self.cfg.spectral_radius / rho)

    return dense.to_sparse_csr()

  def reset_state(self) -> None:
    self.x.zero_()
    self.vla_target.zero_()
    self._step_count = 0

  def update_vla_target(self, vla_token: torch.Tensor) -> None:
    """Latch a new 2 Hz VLA target token (29-D joint target). Call at ~2 Hz."""
    token = vla_token.flatten().to(device=self.device, dtype=torch.float32)
    if token.numel() != self.cfg.vla_token_dim:
      raise ValueError(
        f"VLA token must be {self.cfg.vla_token_dim}-D, got {token.numel()}"
      )
    self.vla_target.copy_(token)

  def build_input(self, joint_state: torch.Tensor) -> torch.Tensor:
    """Concatenate current proprioception with the latched VLA target."""
    state = joint_state.flatten().to(device=self.device, dtype=torch.float32)
    if state.numel() != self.cfg.joint_dim:
      raise ValueError(
        f"Joint state must be {self.cfg.joint_dim}-D, got {state.numel()}"
      )
    return torch.cat([state, self.vla_target], dim=0)

  @torch.no_grad()
  def step_proprio(self, joint_state: torch.Tensor) -> torch.Tensor:
    """
    100 Hz inference entry point.

    Advances the reservoir using [joint_state; latched_vla_target] and returns
    the 29-D joint prediction from the trained readout.
    """
    if self.W_out is None:
      raise RuntimeError("W_out is not fitted — load a checkpoint or train first.")
    return self.step(self.build_input(joint_state))

  def _reservoir_pre_activation(self, u: torch.Tensor) -> torch.Tensor:
    # Sparse matmul: (N, N) @ (N, 1) → (N,)
    res_term = torch.sparse.mm(self.W_res, self.x.unsqueeze(1)).squeeze(1)
    return res_term + self.W_in @ u

  @torch.no_grad()
  def step(self, u: torch.Tensor) -> torch.Tensor:
    """Advance one 10 ms tick and return the readout (if W_out is fitted)."""
    alpha = self.cfg.leaky_rate
    pre = self._reservoir_pre_activation(u)
    self.x = (1.0 - alpha) * self.x + alpha * torch.tanh(pre)

    extended = torch.cat([self.x, u], dim=0)
    if self.W_out is None:
      return extended.new_zeros(self.cfg.output_dim)
    return self.W_out @ extended

  @torch.no_grad()
  def collect_extended_states(
    self,
    joint_states: torch.Tensor,
    vla_targets: torch.Tensor,
    washout: Optional[int] = None,
  ) -> torch.Tensor:
    """
    Drive the reservoir across a trajectory and stack extended states.

    Args:
      joint_states: (T, 29) current proprioception at 100 Hz
      vla_targets:  (T, 29) held 2 Hz VLA target tokens at 100 Hz
    Returns:
      (T - washout, reservoir_size + input_dim)
    """
    wo = self.cfg.washout if washout is None else washout
    t_len = joint_states.shape[0]
    extended_dim = self.cfg.readout_dim
    states = torch.empty(t_len, extended_dim, device=self.device)

    self.reset_state()
    for t in range(t_len):
      u = torch.cat([joint_states[t], vla_targets[t]], dim=0)
      alpha = self.cfg.leaky_rate
      pre = self._reservoir_pre_activation(u)
      self.x = (1.0 - alpha) * self.x + alpha * torch.tanh(pre)
      states[t] = torch.cat([self.x, u], dim=0)

    return states[wo:]


def fit_readout_ridge(
  extended_states: torch.Tensor,
  targets: torch.Tensor,
  ridge_alpha: float,
) -> torch.Tensor:
  """
  Closed-form ridge regression on GPU.

  W_out = Y^T X (X^T X + λI)^{-1}   with X = extended_states, Y = targets
  """
  x = extended_states
  y = targets
  xtx = x.T @ x
  reg = ridge_alpha * torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
  xty = x.T @ y
  w_out_t = torch.linalg.solve(xtx + reg, xty)  # (features, output_dim)
  return w_out_t.T.contiguous()  # (output_dim, features)


def predict_from_extended(
  w_out: torch.Tensor,
  extended_states: torch.Tensor,
) -> torch.Tensor:
  """Apply trained readout to collected extended reservoir states."""
  return (w_out @ extended_states.T).T


def evaluate_predictions(
  predictions: torch.Tensor,
  ground_truth: torch.Tensor,
  control_hz: float = CONTROL_HZ,
) -> Dict[str, float]:
  """Tracking MSE / RMSE, Δq proxy jerk, and physical jerk RMS (rad/s³)."""
  mse = float(torch.mean((predictions - ground_truth) ** 2).item())
  return {
    "mse": mse,
    "rmse": float(mse ** 0.5),
    "jerk": compute_jerk_metric(predictions),
    "jerk_rms": compute_physical_jerk_rms(predictions, control_hz=control_hz),
  }


def combined_selection_score(
  mse: float,
  jerk: float,
  sweep_rows: List[Dict[str, float]],
  jerk_weight: float = 1.0,
) -> float:
  """
  Normalized trade-off score in [0, 2·jerk_weight] — lower is better.

  Both MSE and jerk are min-max normalized across the sweep grid so neither
  metric dominates due to scale differences.
  """
  mses = [row["mse"] for row in sweep_rows]
  jerks = [row["jerk"] for row in sweep_rows]
  mse_span = max(mses) - min(mses) + 1e-12
  jerk_span = max(jerks) - min(jerks) + 1e-12
  norm_mse = (mse - min(mses)) / mse_span
  norm_jerk = (jerk - min(jerks)) / jerk_span
  return float(norm_mse + jerk_weight * norm_jerk)


def evaluate_mse(
  esn: EchoStateNetwork,
  joint_states: torch.Tensor,
  vla_targets: torch.Tensor,
  ground_truth: torch.Tensor,
) -> float:
  """Compute mean squared error between readout predictions and ground truth."""
  wo = esn.cfg.washout
  states = esn.collect_extended_states(joint_states, vla_targets)
  preds = predict_from_extended(esn.W_out, states)
  tgt = ground_truth[wo:]
  metrics = evaluate_predictions(preds, tgt)
  return metrics["mse"]


@torch.no_grad()
def run_hyperparameter_sweep(
  base_cfg: ESNCudaConfig,
  joint_states: torch.Tensor,
  vla_targets: torch.Tensor,
  ground_truth: torch.Tensor,
  device: torch.device,
  leaky_rates: Tuple[float, ...] = SWEEP_LEAKY_RATES,
  ridge_alphas: Tuple[float, ...] = SWEEP_RIDGE_ALPHAS,
  jerk_weight: float = 1.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
  """
  Grid search over leaky_rate × ridge_alpha.

  For each leaky_rate the reservoir states are collected once on GPU; ridge
  regression is re-fit for each λ without re-running the reservoir.
  """
  wo = base_cfg.washout
  targets = ground_truth[wo:]
  rows: List[Dict[str, Any]] = []

  logger.info(
    "Hyperparameter sweep | leaky_rate=%s | ridge_alpha=%s",
    leaky_rates,
    ridge_alphas,
  )

  for leaky_rate in leaky_rates:
    cfg = replace(base_cfg, leaky_rate=leaky_rate)
    esn = EchoStateNetwork(cfg, device=device)
    extended = esn.collect_extended_states(joint_states, vla_targets)

    for ridge_alpha in ridge_alphas:
      w_out = fit_readout_ridge(extended, targets, ridge_alpha)
      preds = predict_from_extended(w_out, extended)
      metrics = evaluate_predictions(preds, targets)
      row = {
        "leaky_rate": leaky_rate,
        "ridge_alpha": ridge_alpha,
        **metrics,
      }
      rows.append(row)
      logger.info(
        "  α_leak=%.2f λ=%.1e → MSE=%.6f jerk=%.6f",
        leaky_rate,
        ridge_alpha,
        metrics["mse"],
        metrics["jerk"],
      )

  df = pd.DataFrame(rows)
  df["selection_score"] = df.apply(
    lambda r: combined_selection_score(r["mse"], r["jerk"], rows, jerk_weight),
    axis=1,
  )
  best_idx = int(df["selection_score"].idxmin())
  best_row = df.loc[best_idx].to_dict()
  best_row["extended_states"] = None  # placeholder — not serializable

  logger.info(
    "Best config: leaky_rate=%.2f ridge_alpha=%.1e | MSE=%.6f jerk=%.6f score=%.4f",
    best_row["leaky_rate"],
    best_row["ridge_alpha"],
    best_row["mse"],
    best_row["jerk"],
    best_row["selection_score"],
  )
  return df, best_row


def train_best_esn_from_sweep(
  base_cfg: ESNCudaConfig,
  best_row: Dict[str, Any],
  joint_states: torch.Tensor,
  vla_targets: torch.Tensor,
  ground_truth: torch.Tensor,
  device: torch.device,
) -> Tuple[EchoStateNetwork, Dict[str, float]]:
  """Re-train and return the ESN selected by the hyperparameter sweep."""
  cfg = replace(
    base_cfg,
    leaky_rate=float(best_row["leaky_rate"]),
    ridge_alpha=float(best_row["ridge_alpha"]),
  )
  esn = EchoStateNetwork(cfg, device=device)
  wo = cfg.washout

  extended = esn.collect_extended_states(joint_states, vla_targets)
  targets = ground_truth[wo:]
  esn.W_out = fit_readout_ridge(extended, targets, cfg.ridge_alpha)
  preds = predict_from_extended(esn.W_out, extended)

  gt_jerk = compute_jerk_metric(targets)
  metrics = evaluate_predictions(preds, targets)
  metrics.update({
    "gt_jerk": gt_jerk,
    "jerk_ratio": metrics["jerk"] / (gt_jerk + 1e-12),
    "selection_score": float(best_row["selection_score"]),
    "leaky_rate": cfg.leaky_rate,
    "ridge_alpha": cfg.ridge_alpha,
    "train_steps": int(extended.shape[0]),
  })
  return esn, metrics


# ── Training pipeline ─────────────────────────────────────────
def train_esn_on_episode(
  esn: EchoStateNetwork,
  joint_states: torch.Tensor,
  vla_targets: torch.Tensor,
  ground_truth: torch.Tensor,
) -> Dict[str, float]:
  """Collect states, fit W_out with ridge regression, return metrics."""
  cfg = esn.cfg
  wo = cfg.washout

  logger.info("Collecting reservoir states on GPU (T=%d)...", joint_states.shape[0])
  t_collect = time.perf_counter()
  extended = esn.collect_extended_states(joint_states, vla_targets)
  collect_s = time.perf_counter() - t_collect

  targets = ground_truth[wo:]
  assert extended.shape[0] == targets.shape[0]

  logger.info("Fitting W_out via ridge regression (λ=%.1e)...", cfg.ridge_alpha)
  t_fit = time.perf_counter()
  esn.W_out = fit_readout_ridge(extended, targets, cfg.ridge_alpha)
  fit_s = time.perf_counter() - t_fit

  preds = predict_from_extended(esn.W_out, extended)
  metrics = evaluate_predictions(preds, targets)
  metrics.update({
    "gt_jerk": compute_jerk_metric(targets),
    "collect_time_s": collect_s,
    "fit_time_s": fit_s,
    "train_steps": int(extended.shape[0]),
    "leaky_rate": cfg.leaky_rate,
    "ridge_alpha": cfg.ridge_alpha,
  })
  metrics["jerk_ratio"] = metrics["jerk"] / (metrics["gt_jerk"] + 1e-12)
  logger.info(
    "Tracking MSE=%.6f | RMSE=%.6f rad | jerk=%.6f",
    metrics["mse"],
    metrics["rmse"],
    metrics["jerk"],
  )
  return metrics


@torch.no_grad()
def accumulate_ridge_stats(
  esn: EchoStateNetwork,
  joint_states: torch.Tensor,
  vla_targets: torch.Tensor,
  ground_truth: torch.Tensor,
  xtx: Optional[torch.Tensor] = None,
  xty: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
  """
  Drive one episode and accumulate XᵀX / XᵀY for multi-episode ridge.

  Avoids concatenating all extended states in memory when training on
  many wipe demos (160 train episodes ≈ hours of 100 Hz states).
  """
  wo = esn.cfg.washout
  extended = esn.collect_extended_states(joint_states, vla_targets)
  targets = ground_truth[wo:]
  feat = extended.shape[1]
  if xtx is None:
    xtx = torch.zeros(feat, feat, device=extended.device, dtype=extended.dtype)
  if xty is None:
    xty = torch.zeros(feat, targets.shape[1], device=extended.device, dtype=extended.dtype)
  xtx += extended.T @ extended
  xty += extended.T @ targets
  return xtx, xty, int(extended.shape[0])


def fit_readout_from_stats(
  xtx: torch.Tensor,
  xty: torch.Tensor,
  ridge_alpha: float,
) -> torch.Tensor:
  """Solve W_out from accumulated ridge normal equations."""
  reg = ridge_alpha * torch.eye(xtx.shape[0], device=xtx.device, dtype=xtx.dtype)
  w_out_t = torch.linalg.solve(xtx + reg, xty)
  return w_out_t.T.contiguous()


@torch.no_grad()
def evaluate_esn_on_episode(
  esn: EchoStateNetwork,
  joint_states: torch.Tensor,
  vla_targets: torch.Tensor,
  ground_truth: torch.Tensor,
) -> Dict[str, float]:
  """Open-loop tracking metrics on one episode (requires fitted W_out)."""
  if esn.W_out is None:
    raise RuntimeError("W_out is not fitted")
  wo = esn.cfg.washout
  extended = esn.collect_extended_states(joint_states, vla_targets)
  preds = predict_from_extended(esn.W_out, extended)
  targets = ground_truth[wo:]
  metrics = evaluate_predictions(preds, targets)
  metrics["gt_jerk"] = compute_jerk_metric(targets)
  metrics["jerk_ratio"] = metrics["jerk"] / (metrics["gt_jerk"] + 1e-12)
  metrics["steps"] = int(targets.shape[0])
  return metrics


def train_esn_on_episodes(
  esn: EchoStateNetwork,
  dataset,
  train_episodes: Sequence[int],
  device: torch.device,
) -> Dict[str, float]:
  """Fit W_out on many episodes via accumulated ridge statistics."""
  episodes = list(train_episodes)
  if not episodes:
    raise ValueError("train_episodes is empty")

  xtx: Optional[torch.Tensor] = None
  xty: Optional[torch.Tensor] = None
  n_steps = 0
  t0 = time.perf_counter()
  for i, ep in enumerate(episodes, start=1):
    joint_states, vla_targets = load_episode_tensors(dataset, ep, device)
    ground_truth = joint_states
    xtx, xty, n = accumulate_ridge_stats(
      esn, joint_states, vla_targets, ground_truth, xtx=xtx, xty=xty,
    )
    n_steps += n
    if i == 1 or i == len(episodes) or i % 10 == 0:
      logger.info(
        "Ridge accumulate [%d/%d] ep=%d (+%d steps, total=%d)",
        i, len(episodes), ep, n, n_steps,
      )

  assert xtx is not None and xty is not None
  esn.W_out = fit_readout_from_stats(xtx, xty, esn.cfg.ridge_alpha)
  fit_s = time.perf_counter() - t0

  # In-sample metrics on first train episode (cheap sanity; full train mean optional).
  js0, vla0 = load_episode_tensors(dataset, episodes[0], device)
  metrics = evaluate_esn_on_episode(esn, js0, vla0, js0)
  metrics.update({
    "train_steps": float(n_steps),
    "n_train_episodes": float(len(episodes)),
    "fit_time_s": fit_s,
    "leaky_rate": esn.cfg.leaky_rate,
    "ridge_alpha": esn.cfg.ridge_alpha,
    "train_metrics_episode": float(episodes[0]),
  })
  logger.info(
    "Multi-ep train done | eps=%d steps=%d | ep%d RMSE=%.6f jerk=%.6f (%.1fs)",
    len(episodes), n_steps, episodes[0], metrics["rmse"], metrics["jerk"], fit_s,
  )
  return metrics


def evaluate_esn_on_episodes(
  esn: EchoStateNetwork,
  dataset,
  episodes: Sequence[int],
  device: torch.device,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
  """Per-episode + mean open-loop metrics (held-out or train)."""
  rows: List[Dict[str, Any]] = []
  for ep in episodes:
    js, vla = load_episode_tensors(dataset, ep, device)
    m = evaluate_esn_on_episode(esn, js, vla, js)
    row = {"episode": int(ep), **m}
    rows.append(row)
    logger.info("Eval ep=%d RMSE=%.6f jerk=%.3e", ep, m["rmse"], m["jerk"])

  if not rows:
    return rows, {}
  mean = {
    "n_episodes": float(len(rows)),
    "rmse_mean": float(np.mean([r["rmse"] for r in rows])),
    "rmse_std": float(np.std([r["rmse"] for r in rows])),
    "mse_mean": float(np.mean([r["mse"] for r in rows])),
    "jerk_mean": float(np.mean([r["jerk"] for r in rows])),
    "jerk_std": float(np.std([r["jerk"] for r in rows])),
  }
  return rows, mean


@torch.no_grad()
def benchmark_step_hz(esn: EchoStateNetwork, input_dim: int, n_warmup: int = 100, n_iters: int = 2000) -> float:
  """Estimate reservoir update throughput on the current CUDA device."""
  u = torch.randn(input_dim, device=esn.device)
  for _ in range(n_warmup):
    esn.step(u)
  torch.cuda.synchronize()
  t0 = time.perf_counter()
  for _ in range(n_iters):
    esn.step(u)
  torch.cuda.synchronize()
  elapsed = time.perf_counter() - t0
  return n_iters / elapsed


def _build_checkpoint_payload(
  esn: EchoStateNetwork,
  metrics: Dict[str, float],
  *,
  episode_index: Optional[int] = None,
  train_episodes: Optional[Sequence[int]] = None,
  heldout_episodes: Optional[Sequence[int]] = None,
  heldout_metrics: Optional[Dict[str, float]] = None,
  dataset_id: str = DATASET_ID,
  task_id: Optional[str] = None,
  unnorm_key: Optional[str] = None,
) -> Dict[str, Any]:
  w_res_coo = esn.W_res.to_sparse_coo().cpu()
  train_eps = list(train_episodes) if train_episodes is not None else (
    [int(episode_index)] if episode_index is not None else None
  )
  return {
    "checkpoint_version": CHECKPOINT_VERSION,
    "model": "EchoStateNetwork",
    "config": esn.cfg.__dict__,
    "metrics": metrics,
    "dataset_id": dataset_id,
    "task_id": task_id,
    "unnorm_key": unnorm_key,
    "episode_index": episode_index if episode_index is not None else (
      train_eps[0] if train_eps else None
    ),
    "train_episodes": train_eps,
    "heldout_episodes": list(heldout_episodes) if heldout_episodes is not None else None,
    "heldout_metrics": heldout_metrics,
    "control_hz": CONTROL_HZ,
    "vla_hz": VLA_HZ,
    "vla_hold_steps": VLA_HOLD_STEPS,
    "W_out": esn.W_out.detach().cpu(),
    "W_in": esn.W_in.detach().cpu(),
    "W_res_indices": w_res_coo.indices(),
    "W_res_values": w_res_coo.values(),
    "W_res_size": torch.tensor(w_res_coo.shape),
  }


def save_checkpoint(
  esn: EchoStateNetwork,
  metrics: Dict[str, float],
  out_dir: Union[str, Path],
  *,
  episode_index: Optional[int] = None,
  train_episodes: Optional[Sequence[int]] = None,
  heldout_episodes: Optional[Sequence[int]] = None,
  heldout_metrics: Optional[Dict[str, float]] = None,
  dataset_id: str = DATASET_ID,
  basename: str = CHECKPOINT_BASENAME,
  task_id: Optional[str] = None,
  unnorm_key: Optional[str] = None,
) -> Dict[str, Path]:
  """
  Persist a trained ESN for later inference.

  Writes:
    {basename}.pt / {basename}.pth — full torch bundle
    config.json                     — human-readable metadata
    W_out.npy                       — readout weights only
  """
  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  if esn.W_out is None:
    raise RuntimeError("Cannot save — W_out has not been fitted.")

  payload = _build_checkpoint_payload(
    esn,
    metrics,
    episode_index=episode_index,
    train_episodes=train_episodes,
    heldout_episodes=heldout_episodes,
    heldout_metrics=heldout_metrics,
    dataset_id=dataset_id,
    task_id=task_id,
    unnorm_key=unnorm_key,
  )
  pt_path = out_dir / f"{basename}.pt"
  pth_path = out_dir / f"{basename}.pth"
  torch.save(payload, pt_path)
  torch.save(payload, pth_path)

  train_eps = payload.get("train_episodes")
  meta = {
    "checkpoint_version": CHECKPOINT_VERSION,
    "model": "EchoStateNetwork",
    "task_id": task_id,
    "unnorm_key": unnorm_key,
    "dataset_id": dataset_id,
    "episode_index": payload.get("episode_index"),
    "train_episodes": train_eps,
    "heldout_episodes": payload.get("heldout_episodes"),
    "heldout_metrics": heldout_metrics,
    "control_hz": CONTROL_HZ,
    "vla_hz": VLA_HZ,
    "vla_hold_steps": VLA_HOLD_STEPS,
    "config": esn.cfg.__dict__,
    "metrics": metrics,
    "artifacts": {
      "torch_checkpoint_pt": pt_path.name,
      "torch_checkpoint_pth": pth_path.name,
      "readout_weights": "W_out.npy",
    },
  }
  json_path = out_dir / "config.json"
  with open(json_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

  npy_path = out_dir / "W_out.npy"
  np.save(npy_path, esn.W_out.detach().cpu().numpy())

  paths = {"pt": pt_path, "pth": pth_path, "json": json_path, "w_out_npy": npy_path}
  logger.info("Checkpoint saved for inference:")
  logger.info("  %s", pt_path)
  logger.info("  %s", pth_path)
  logger.info("  %s", json_path)
  logger.info("  %s", npy_path)
  return paths


def load_checkpoint(
  path: Union[str, Path],
  device: Union[str, torch.device] = "cuda",
) -> EchoStateNetwork:
  """
  Restore a trained ESN from disk for 100 Hz inference.

  Usage:
    esn = load_checkpoint(models_path("esn_cuda_ridge"))
    esn.update_vla_target(vla_joint_target)   # at 2 Hz
    cmd = esn.step_proprio(current_joint_state)  # at 100 Hz
  """
  path = Path(path)
  if path.is_dir():
    for name in (
      f"{BEST_CHECKPOINT_BASENAME}.pth",
      f"{BEST_CHECKPOINT_BASENAME}.pt",
      f"{CHECKPOINT_BASENAME}.pth",
      f"{CHECKPOINT_BASENAME}.pt",
    ):
      candidate = path / name
      if candidate.is_file():
        pt_path = candidate
        break
    else:
      raise FileNotFoundError(f"No checkpoint found in directory: {path}")
  else:
    pt_path = path

  map_location = torch.device(device)
  payload = torch.load(pt_path, map_location=map_location, weights_only=False)

  cfg = ESNCudaConfig(**payload["config"])
  esn = EchoStateNetwork(cfg, device=map_location)
  esn.W_in.copy_(payload["W_in"].to(map_location))
  esn.W_out = payload["W_out"].to(map_location)

  w_res = torch.sparse_coo_tensor(
    payload["W_res_indices"].to(map_location),
    payload["W_res_values"].to(map_location),
    size=tuple(int(s) for s in payload["W_res_size"].tolist()),
  ).to_sparse_csr()
  esn.W_res = w_res
  esn.eval()

  logger.info("Loaded ESN checkpoint from %s (v%s)", pt_path, payload.get("checkpoint_version", "?"))
  if payload.get("task_id"):
    logger.info("  task_id=%s | unnorm_key=%s", payload.get("task_id"), payload.get("unnorm_key"))
  if "metrics" in payload:
    logger.info(
      "  MSE=%.6f | jerk=%.6f | step_hz=%s",
      payload["metrics"].get("mse", float("nan")),
      payload["metrics"].get("jerk", float("nan")),
      payload["metrics"].get("step_hz", "n/a"),
    )
  return esn


def load_esn_for_task(
  task_id: str,
  device: Union[str, torch.device] = "cuda",
  *,
  checkpoint_dir: Optional[Union[str, Path]] = None,
) -> EchoStateNetwork:
  """Load the per-task ESN checkpoint (``models/esn_cuda_ridge_<task>``)."""
  from src.unifolm_tasks import esn_checkpoint_basename, get_task

  task = get_task(task_id)
  out = Path(checkpoint_dir) if checkpoint_dir else models_path(esn_checkpoint_basename(task.id))
  return load_checkpoint(out, device=device)


def update_task_registry(
  entry: Dict[str, Any],
  *,
  registry_path: Optional[Path] = None,
) -> Path:
  """Merge one task entry into ``models/esn_task_registry.json``."""
  registry_path = Path(registry_path or models_path("esn_task_registry.json"))
  registry_path.parent.mkdir(parents=True, exist_ok=True)
  registry: Dict[str, Any] = {}
  if registry_path.is_file():
    try:
      registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
      registry = {}
  task_id = str(entry["task_id"])
  registry[task_id] = entry
  registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
  logger.info("Updated task registry: %s (%s)", registry_path, task_id)
  return registry_path


def write_multitask_summary(
  rows: Sequence[Dict[str, Any]],
  *,
  out_dir: Optional[Path] = None,
) -> Dict[str, Path]:
  """Write suite CSV + JSON under ``results/step2_training/``."""
  out_dir = Path(out_dir or results_path("step2_training"))
  out_dir.mkdir(parents=True, exist_ok=True)
  df = pd.DataFrame(list(rows))
  csv_path = out_dir / "esn_multitask_summary.csv"
  json_path = out_dir / "esn_multitask_summary.json"
  df.to_csv(csv_path, index=False)
  json_path.write_text(json.dumps(list(rows), indent=2), encoding="utf-8")
  logger.info("Multi-task summary: %s", csv_path)
  return {"csv": csv_path, "json": json_path}


# ── Main ──────────────────────────────────────────────────────
def run_training_for_task(
  task_id: str,
  *,
  dataset_id: Optional[str] = None,
  train_episodes_spec: str = "train",
  heldout_episodes_spec: str = "heldout",
  sweep_episodes_spec: Optional[str] = None,
  max_train_episodes: Optional[int] = None,
  max_heldout_episodes: Optional[int] = None,
  episode: Optional[int] = None,
  reservoir_size: int = 1000,
  spectral_radius: float = 0.95,
  sparsity: float = 0.90,
  leaky_rate: float = 0.1,
  input_scaling: float = 1.0,
  ridge_alpha: float = 1e-2,
  washout: int = 50,
  seed: int = 42,
  device: str = "cuda",
  skip_sweep: bool = False,
  jerk_weight: float = 1.0,
) -> Dict[str, Any]:
  """Train + eval one UnifoLM task; save a dedicated checkpoint + results."""
  from src.unifolm_tasks import esn_checkpoint_basename, get_task
  from src.wipe_dataset import resolve_task_episode_spec

  task = get_task(task_id)
  dataset_id = dataset_id or task.primary_dataset_id

  if not torch.cuda.is_available() and str(device).startswith("cuda"):
    raise RuntimeError("CUDA is required for this script (target: Tesla V100).")

  torch_device = torch.device(device)
  if torch_device.type == "cuda":
    logger.info("Device: %s (%s)", torch_device, torch.cuda.get_device_name(torch_device))
  else:
    logger.info("Device: %s", torch_device)
  logger.info("Task=%s | unnorm_key=%s | dataset=%s", task.id, task.unnorm_key, dataset_id)

  if episode is not None:
    train_episodes = [int(episode)]
  else:
    train_episodes = resolve_task_episode_spec(train_episodes_spec, dataset_id=dataset_id)
  if max_train_episodes is not None:
    train_episodes = train_episodes[: max(0, int(max_train_episodes))]
  heldout_episodes = resolve_task_episode_spec(heldout_episodes_spec, dataset_id=dataset_id)
  if max_heldout_episodes is not None:
    heldout_episodes = heldout_episodes[: max(0, int(max_heldout_episodes))]
  if sweep_episodes_spec:
    sweep_episodes = resolve_task_episode_spec(sweep_episodes_spec, dataset_id=dataset_id)
  else:
    sweep_episodes = [train_episodes[0]]

  logger.info("Loading dataset %s ...", dataset_id)
  dataset = load_dataset(dataset_id, split="train")
  logger.info(
    "Train eps=%d %s… | sweep eps=%s | held-out eps=%d %s…",
    len(train_episodes),
    train_episodes[:5],
    sweep_episodes,
    len(heldout_episodes),
    heldout_episodes[:5],
  )

  sweep_js, sweep_vla = load_episode_tensors(dataset, sweep_episodes[0], torch_device)
  if len(sweep_episodes) > 1:
    logger.info(
      "Note: hyperparam sweep uses episode %d only; final fit uses all train episodes.",
      sweep_episodes[0],
    )
  sweep_gt = sweep_js.clone()

  base_cfg = ESNCudaConfig(
    reservoir_size=reservoir_size,
    spectral_radius=spectral_radius,
    sparsity=sparsity,
    leaky_rate=leaky_rate,
    input_scaling=input_scaling,
    ridge_alpha=ridge_alpha,
    washout=washout,
    seed=seed,
  )

  out_dir = models_path(esn_checkpoint_basename(task.id))
  if task.id == "wipe_table":
    sweep_dir = results_path("step2_training")
  else:
    sweep_dir = results_path("step2_training", task.id)
  sweep_dir.mkdir(parents=True, exist_ok=True)

  if skip_sweep:
    esn = EchoStateNetwork(base_cfg, device=torch_device).to(torch_device)
    logger.info(
      "ESN init (no sweep) | N=%d | α_leak=%.2f | λ=%.1e",
      base_cfg.reservoir_size,
      base_cfg.leaky_rate,
      base_cfg.ridge_alpha,
    )
    if len(train_episodes) == 1:
      metrics = train_esn_on_episode(esn, sweep_js, sweep_vla, sweep_gt)
    else:
      metrics = train_esn_on_episodes(esn, dataset, train_episodes, torch_device)
    sweep_df = None
  else:
    sweep_df, best_row = run_hyperparameter_sweep(
      base_cfg,
      sweep_js,
      sweep_vla,
      sweep_gt,
      torch_device,
      jerk_weight=jerk_weight,
    )
    sweep_csv = sweep_dir / "esn_hyperparam_sweep.csv"
    sweep_df.to_csv(sweep_csv, index=False)
    logger.info("Sweep results saved: %s", sweep_csv)

    cfg = replace(
      base_cfg,
      leaky_rate=float(best_row["leaky_rate"]),
      ridge_alpha=float(best_row["ridge_alpha"]),
    )
    esn = EchoStateNetwork(cfg, device=torch_device).to(torch_device)
    if len(train_episodes) == 1:
      esn, metrics = train_best_esn_from_sweep(
        base_cfg, best_row, sweep_js, sweep_vla, sweep_gt, torch_device,
      )
    else:
      metrics = train_esn_on_episodes(esn, dataset, train_episodes, torch_device)
      metrics["selection_score"] = float(best_row["selection_score"])

  metrics["step_hz"] = benchmark_step_hz(esn, esn.cfg.input_dim)
  logger.info("Reservoir step throughput: %.1f Hz (target >100 Hz)", metrics["step_hz"])

  heldout_rows: List[Dict[str, Any]] = []
  heldout_mean: Dict[str, float] = {}
  if heldout_episodes:
    heldout_rows, heldout_mean = evaluate_esn_on_episodes(
      esn, dataset, heldout_episodes, torch_device,
    )
    held_csv = sweep_dir / "esn_heldout_eval.csv"
    pd.DataFrame(heldout_rows).to_csv(held_csv, index=False)
    (sweep_dir / "esn_heldout_summary.json").write_text(json.dumps(heldout_mean, indent=2))
    logger.info("Held-out summary: %s", heldout_mean)
    logger.info("Held-out per-episode CSV: %s", held_csv)

  artifact_paths = save_checkpoint(
    esn,
    metrics,
    out_dir,
    episode_index=train_episodes[0],
    train_episodes=train_episodes,
    heldout_episodes=heldout_episodes,
    heldout_metrics=heldout_mean or None,
    dataset_id=dataset_id,
    basename=BEST_CHECKPOINT_BASENAME,
    task_id=task.id,
    unnorm_key=task.unnorm_key,
  )

  summary = {
    "task_id": task.id,
    "unnorm_key": task.unnorm_key,
    "dataset_id": dataset_id,
    "checkpoint_dir": str(out_dir),
    "checkpoint_pth": str(artifact_paths["pth"]),
    "n_train_episodes": len(train_episodes),
    "n_heldout_episodes": len(heldout_episodes),
    "train_rmse": float(metrics.get("rmse", float("nan"))),
    "train_jerk": float(metrics.get("jerk", float("nan"))),
    "heldout_rmse_mean": float(heldout_mean.get("rmse_mean", float("nan"))) if heldout_mean else float("nan"),
    "heldout_rmse_std": float(heldout_mean.get("rmse_std", float("nan"))) if heldout_mean else float("nan"),
    "heldout_jerk_mean": float(heldout_mean.get("jerk_mean", float("nan"))) if heldout_mean else float("nan"),
    "leaky_rate": float(metrics.get("leaky_rate", esn.cfg.leaky_rate)),
    "ridge_alpha": float(metrics.get("ridge_alpha", esn.cfg.ridge_alpha)),
    "step_hz": float(metrics.get("step_hz", float("nan"))),
    "status": "ok",
  }
  update_task_registry(summary)

  print("\n" + "=" * 60)
  print("  ESN CUDA Ridge Training — Phase 2 (multi-episode)")
  print("=" * 60)
  print(f"  Task            : {task.id} ({task.unnorm_key})")
  print(f"  Dataset         : {dataset_id}")
  print(f"  Train episodes  : {len(train_episodes)}  ({train_episodes[0]}…{train_episodes[-1]})")
  print(f"  Held-out episodes: {len(heldout_episodes)}")
  if task.id == "wipe_table":
    print(f"  Canonical split : train={len(TRAIN_EPISODES)} heldout={len(HELDOUT_EPISODES)}")
  print(f"  VLA hold period : {VLA_HOLD_STEPS} steps ({VLA_HZ:.0f} Hz)")
  print(f"  Reservoir N     : {esn.cfg.reservoir_size}")
  print(f"  Best α_leak     : {metrics.get('leaky_rate', esn.cfg.leaky_rate):.2f}")
  print(f"  Best ridge λ    : {metrics.get('ridge_alpha', esn.cfg.ridge_alpha):.1e}")
  print(f"  Train-ref RMSE  : {metrics['rmse']:.6f} rad (ep {int(metrics.get('train_metrics_episode', train_episodes[0]))})")
  if heldout_mean:
    print(
      f"  Held-out RMSE   : {heldout_mean['rmse_mean']:.6f} ± {heldout_mean['rmse_std']:.6f} rad "
      f"(n={int(heldout_mean['n_episodes'])})"
    )
  print(f"  Step throughput : {metrics['step_hz']:.1f} Hz")
  print(f"  Best .pth       : {artifact_paths['pth']}")
  print(f"  Config JSON     : {artifact_paths['json']}")
  if sweep_df is not None:
    print(f"  Sweep CSV       : {sweep_dir / 'esn_hyperparam_sweep.csv'}")
  print("=" * 60)
  return summary


def main() -> None:
  parser = argparse.ArgumentParser(description="Train CUDA ESN readout with ridge regression")
  from src.unifolm_tasks import (
    DEFAULT_TASK_ID,
    ESN_SUITE_TASK_IDS,
    add_task_arg,
    get_task,
    list_esn_suite_tasks,
    maybe_print_tasks_and_exit,
  )

  add_task_arg(parser, default=DEFAULT_TASK_ID)
  parser.add_argument(
    "--all_tasks",
    action="store_true",
    help=(
      "Train one ESN per single-embodiment UnifoLM G1 task "
      f"({len(ESN_SUITE_TASK_IDS)} tasks; excludes dual_clean_table). "
      "Writes models/esn_cuda_ridge_<task>/ and results/step2_training/esn_multitask_summary.*"
    ),
  )
  parser.add_argument(
    "--tasks",
    type=str,
    default=None,
    help="Comma-separated task ids to train (overrides --task / --all_tasks suite filter)",
  )
  parser.add_argument("--dataset", type=str, default=None, help="Override HF dataset id")
  parser.add_argument(
    "--episode",
    type=int,
    default=None,
    help="Legacy single-episode train (overrides --train_episodes if set)",
  )
  parser.add_argument(
    "--train_episodes",
    type=str,
    default="train",
    help="Episode spec: train|0-159|0,1,2 (default: canonical train split)",
  )
  parser.add_argument(
    "--heldout_episodes",
    type=str,
    default="heldout",
    help="Episode spec for generalization eval (default: heldout split)",
  )
  parser.add_argument(
    "--sweep_episodes",
    type=str,
    default=None,
    help="Episodes for hyperparam sweep (default: first train episode only)",
  )
  parser.add_argument(
    "--max_train_episodes",
    type=int,
    default=None,
    help="Optional cap on train list (smoke / debug)",
  )
  parser.add_argument(
    "--max_heldout_episodes",
    type=int,
    default=None,
    help="Optional cap on held-out eval list",
  )
  parser.add_argument("--reservoir_size", type=int, default=1000)
  parser.add_argument("--spectral_radius", type=float, default=0.95)
  parser.add_argument("--sparsity", type=float, default=0.90)
  parser.add_argument("--leaky_rate", type=float, default=0.1,
                      help="Leaky integrator α (lower = more low-pass smoothing)")
  parser.add_argument("--input_scaling", type=float, default=1.0)
  parser.add_argument("--ridge_alpha", type=float, default=1e-2,
                      help="Ridge regression λ (higher = smoother readout)")
  parser.add_argument("--washout", type=int, default=50)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", type=str, default="cuda")
  parser.add_argument("--skip_sweep", action="store_true",
                      help="Skip grid search; train with --leaky_rate and --ridge_alpha only")
  parser.add_argument("--jerk_weight", type=float, default=1.0,
                      help="Weight for jerk vs MSE in best-model selection score")
  parser.add_argument(
    "--continue_on_error",
    action="store_true",
    help="With --all_tasks / --tasks: log failures and continue the suite",
  )
  args = parser.parse_args()
  maybe_print_tasks_and_exit(args)

  if args.tasks:
    task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]
  elif args.all_tasks:
    task_ids = list(ESN_SUITE_TASK_IDS)
  else:
    task_ids = [get_task(args.task).id]

  # Validate early
  for tid in task_ids:
    get_task(tid)

  common_kwargs = dict(
    dataset_id=args.dataset if len(task_ids) == 1 else None,
    train_episodes_spec=args.train_episodes,
    heldout_episodes_spec=args.heldout_episodes,
    sweep_episodes_spec=args.sweep_episodes,
    max_train_episodes=args.max_train_episodes,
    max_heldout_episodes=args.max_heldout_episodes,
    episode=args.episode if len(task_ids) == 1 else None,
    reservoir_size=args.reservoir_size,
    spectral_radius=args.spectral_radius,
    sparsity=args.sparsity,
    leaky_rate=args.leaky_rate,
    input_scaling=args.input_scaling,
    ridge_alpha=args.ridge_alpha,
    washout=args.washout,
    seed=args.seed,
    device=args.device,
    skip_sweep=args.skip_sweep,
    jerk_weight=args.jerk_weight,
  )

  summaries: List[Dict[str, Any]] = []
  if len(task_ids) > 1:
    logger.info(
      "Multi-task ESN suite (%d): %s",
      len(task_ids),
      ", ".join(task_ids),
    )
    logger.info(
      "Default suite excludes dual_clean_table (%d single-embodiment tasks).",
      len(list_esn_suite_tasks()),
    )

  for tid in task_ids:
    try:
      summary = run_training_for_task(tid, **common_kwargs)
      summaries.append(summary)
    except Exception as exc:
      logger.exception("Task %s failed: %s", tid, exc)
      fail_row = {
        "task_id": tid,
        "status": "error",
        "error": str(exc),
        "heldout_rmse_mean": float("nan"),
        "heldout_rmse_std": float("nan"),
        "checkpoint_dir": "",
      }
      summaries.append(fail_row)
      if not args.continue_on_error and len(task_ids) > 1:
        raise

  if len(summaries) > 1 or args.all_tasks:
    paths = write_multitask_summary(summaries)
    print("\nMulti-task summary written:")
    print(f"  {paths['csv']}")
    print(f"  {paths['json']}")
    print(f"  Registry: {models_path('esn_task_registry.json')}")
    ok = [s for s in summaries if s.get("status") == "ok"]
    print(f"  Succeeded: {len(ok)}/{len(summaries)}")


if __name__ == "__main__":
  main()
