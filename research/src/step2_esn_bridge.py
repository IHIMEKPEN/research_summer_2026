"""
============================================================
Step 2 — ESN Bridge Core Architecture
Week 5–6 Deliverable: Echo State Network for 100 Hz Control
Research Plan: VLA + ESN for Real-Time Humanoid Control
Author: Osemudiamen Andrew Ihimekpen | PVAMU CREDIT Center
============================================================

Architecture:
  OpenVLA (3 Hz) → hidden state h_t ∈ R^d
       ↓  (at each 10 ms tick)
  ESN Reservoir: x(t) = tanh(W·x(t-1) + W_in·u(t) + W_fb·y(t-1))
       ↓
  Readout: y(t) = W_out · [x(t); u(t)]   ← trained via ridge regression
       ↓
  G1 joint commands ∈ R^29 at 100 Hz

Key design choices:
  - Reservoir is fixed (not backpropagated through)
  - W_out is the only trained parameter → analytically solvable
  - Spectral radius ρ < 1 guarantees echo state property
  - W_fb (optional) closes the output loop for smoother trajectories
"""

import numpy as np
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple, List, Dict

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────
G1_DOF = 29
OPENVLA_HIDDEN_DIM = 4096   # LLaMA-7B hidden size (approximate for mock)
TARGET_HZ = 100


# ── ESN Configuration ─────────────────────────────────────────
@dataclass
class ESNConfig:
    """All hyperparameters for the ESN bridge."""
    reservoir_size: int = 1000          # N: number of reservoir neurons
    spectral_radius: float = 0.95       # ρ: must be < 1 for echo state property
    sparsity: float = 0.90              # fraction of W that is zero
    input_scaling: float = 1.0          # scale applied to W_in
    feedback_scaling: float = 0.0       # scale for W_fb (0 = no output feedback)
    leaking_rate: float = 1.0           # α: leaky integrator (1 = no leaking)
    washout: int = 50                   # discard first N steps during training
    ridge_alpha: float = 1e-4           # regularisation for W_out ridge regression
    input_dim: int = OPENVLA_HIDDEN_DIM # dimension of VLA hidden state
    output_dim: int = G1_DOF            # G1 joint commands
    seed: int = 42


@dataclass
class ESNState:
    """Serialisable snapshot of ESN runtime state."""
    x: List[float]          # reservoir activations x(t)
    y: List[float]          # last output y(t-1)
    step: int


# ── Core ESN Bridge ───────────────────────────────────────────
class ESNBridge:
    """
    Echo State Network bridge between OpenVLA (3 Hz) and G1 (100 Hz).

    Workflow:
      1. Call update_from_vla(h) when a new VLA hidden state arrives (~every 330 ms)
      2. Call step() at 100 Hz to get the next joint command
      3. W_out is fit offline via fit(states, targets)
    """

    def __init__(self, config: ESNConfig):
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)

        # Build fixed random reservoir matrices
        self.W     = self._init_reservoir()
        self.W_in  = self._init_input_weights()
        self.W_fb  = self._init_feedback_weights()

        # Trainable readout (initialised to zeros; fit() fills this)
        extended_dim = config.reservoir_size + config.input_dim
        self.W_out = np.zeros((config.output_dim, extended_dim))

        # Runtime state
        self.x = np.zeros(config.reservoir_size)   # reservoir state
        self.y = np.zeros(config.output_dim)        # last output
        self.u = np.zeros(config.input_dim)         # current VLA input
        self._step_count = 0

        logger.info(
            f"ESNBridge init | N={config.reservoir_size} | ρ={config.spectral_radius} "
            f"| sparsity={config.sparsity} | input_dim={config.input_dim}"
        )

    # ── Matrix initialisation ──────────────────────────────────
    def _init_reservoir(self) -> np.ndarray:
        N = self.cfg.reservoir_size
        p_keep = 1.0 - self.cfg.sparsity
        W = self.rng.standard_normal((N, N))
        mask = self.rng.random((N, N)) > self.cfg.sparsity
        W *= mask

        # Scale to desired spectral radius
        eigenvalues = np.linalg.eigvals(W)
        current_rho = np.max(np.abs(eigenvalues))
        if current_rho > 1e-10:
            W = W * (self.cfg.spectral_radius / current_rho)

        logger.debug(f"Reservoir W: shape={W.shape}, ρ={self.cfg.spectral_radius}")
        return W.astype(np.float32)

    def _init_input_weights(self) -> np.ndarray:
        W_in = self.rng.standard_normal(
            (self.cfg.reservoir_size, self.cfg.input_dim)
        ) * self.cfg.input_scaling
        return W_in.astype(np.float32)

    def _init_feedback_weights(self) -> np.ndarray:
        W_fb = self.rng.standard_normal(
            (self.cfg.reservoir_size, self.cfg.output_dim)
        ) * self.cfg.feedback_scaling
        return W_fb.astype(np.float32)

    # ── Runtime ───────────────────────────────────────────────
    def update_from_vla(self, hidden_state: np.ndarray):
        """
        Called when OpenVLA produces a new hidden state (~3 Hz).
        Updates the input signal u used by subsequent step() calls.
        """
        if hidden_state.shape[0] != self.cfg.input_dim:
            # Pad or truncate to match configured input_dim
            h = np.zeros(self.cfg.input_dim, dtype=np.float32)
            n = min(hidden_state.shape[0], self.cfg.input_dim)
            h[:n] = hidden_state[:n]
        else:
            h = hidden_state.astype(np.float32)
        self.u = h

    def step(self) -> np.ndarray:
        """
        Advance reservoir by one 10 ms tick (100 Hz).
        Returns G1 joint command y(t) ∈ R^29.
        """
        α = self.cfg.leaking_rate
        pre = (
            self.W @ self.x
            + self.W_in @ self.u
            + self.W_fb @ self.y
        )
        x_new = (1.0 - α) * self.x + α * np.tanh(pre)
        self.x = x_new

        # Extended state for readout
        extended = np.concatenate([self.x, self.u])
        self.y = self.W_out @ extended

        self._step_count += 1
        return self.y.copy()

    def reset(self):
        self.x = np.zeros(self.cfg.reservoir_size)
        self.y = np.zeros(self.cfg.output_dim)
        self.u = np.zeros(self.cfg.input_dim)
        self._step_count = 0

    # ── Training ──────────────────────────────────────────────
    def collect_states(
        self,
        inputs: np.ndarray,    # (T, input_dim) — VLA hidden states at 100 Hz
        washout: Optional[int] = None,
    ) -> np.ndarray:
        """
        Drive the reservoir with input sequence, return extended states.
        inputs: (T, input_dim) — upsampled VLA hidden states (held between VLA ticks)
        Returns: (T - washout, N + input_dim)
        """
        T = inputs.shape[0]
        wo = washout if washout is not None else self.cfg.washout
        self.reset()

        states = []
        for t in range(T):
            self.u = inputs[t].astype(np.float32)
            pre = self.W @ self.x + self.W_in @ self.u + self.W_fb @ self.y
            α = self.cfg.leaking_rate
            self.x = (1.0 - α) * self.x + α * np.tanh(pre)
            extended = np.concatenate([self.x, self.u])
            states.append(extended.copy())

        states = np.array(states)      # (T, N + input_dim)
        return states[wo:]             # discard washout

    def fit(
        self,
        inputs: np.ndarray,    # (T, input_dim)
        targets: np.ndarray,   # (T, output_dim)  — ground-truth joint commands
        washout: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Train W_out via ridge regression.
        Returns training metrics.
        """
        wo = washout if washout is not None else self.cfg.washout
        assert inputs.shape[0] == targets.shape[0], "inputs/targets length mismatch"

        logger.info(f"Collecting reservoir states over T={inputs.shape[0]} steps...")
        states = self.collect_states(inputs, washout=wo)
        tgt = targets[wo:]

        logger.info(f"Fitting W_out via ridge regression | α={self.cfg.ridge_alpha}")
        # Ridge: W_out = (S^T S + αI)^{-1} S^T Y
        S = states     # (T', N + input_dim)
        Y = tgt        # (T', output_dim)

        A = S.T @ S + self.cfg.ridge_alpha * np.eye(S.shape[1])
        B = S.T @ Y
        self.W_out = np.linalg.solve(A, B).T   # (output_dim, N + input_dim)

        # Training metrics
        Y_pred = (self.W_out @ S.T).T
        mse = float(np.mean((Y_pred - Y) ** 2))
        rmse = float(np.sqrt(mse))
        nmse = float(mse / (np.var(Y) + 1e-12))
        r2   = float(1.0 - np.sum((Y - Y_pred)**2) / (np.sum((Y - np.mean(Y, axis=0))**2) + 1e-12))

        metrics = {"mse": mse, "rmse": rmse, "nmse": nmse, "r2": r2, "T_train": int(T := inputs.shape[0])}
        logger.info(f"Training done | RMSE={rmse:.4f} | NMSE={nmse:.4f} | R²={r2:.4f}")
        return metrics

    # ── Persistence ───────────────────────────────────────────
    def save(self, path: str):
        """Save W_out and config (W, W_in, W_fb are regenerated from seed)."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        np.save(p / "W_out.npy", self.W_out)
        with open(p / "config.json", "w") as f:
            json.dump(asdict(self.cfg), f, indent=2)
        logger.info(f"ESN saved to {p}")

    @classmethod
    def load(cls, path: str) -> "ESNBridge":
        """Reconstruct ESN from saved W_out and config."""
        p = Path(path)
        with open(p / "config.json") as f:
            cfg_dict = json.load(f)
        cfg = ESNConfig(**cfg_dict)
        esn = cls(cfg)
        esn.W_out = np.load(p / "W_out.npy")
        logger.info(f"ESN loaded from {p}")
        return esn

    def get_state(self) -> ESNState:
        return ESNState(x=self.x.tolist(), y=self.y.tolist(), step=self._step_count)

    def set_state(self, state: ESNState):
        self.x = np.array(state.x, dtype=np.float32)
        self.y = np.array(state.y, dtype=np.float32)
        self._step_count = state.step

    @property
    def n_trainable_params(self) -> int:
        return self.W_out.size

    @property
    def latency_estimate_ms(self) -> float:
        """Rough estimate: single W_out matmul at 100 Hz."""
        N = self.cfg.reservoir_size
        return (N * self.cfg.output_dim * 2) / 1e9 * 1000  # ~microseconds on modern CPU


# ── Upsampling utility ────────────────────────────────────────
def upsample_vla_states(
    vla_states: np.ndarray,    # (K, d) — one state per VLA tick
    vla_hz: float = 3.2,
    target_hz: float = 100.0,
    method: str = "hold",      # "hold" | "linear"
) -> np.ndarray:
    """
    Upsample VLA hidden states from ~3 Hz to 100 Hz for reservoir driving.
    "hold": zero-order hold (last value)
    "linear": linear interpolation between consecutive states
    """
    ratio = int(np.round(target_hz / vla_hz))
    K, d = vla_states.shape
    T = K * ratio

    if method == "hold":
        upsampled = np.repeat(vla_states, ratio, axis=0)[:T]
    elif method == "linear":
        upsampled = np.zeros((T, d), dtype=np.float32)
        for k in range(K - 1):
            start, end = k * ratio, (k + 1) * ratio
            for i, t in enumerate(range(start, end)):
                α = i / ratio
                upsampled[t] = (1 - α) * vla_states[k] + α * vla_states[k + 1]
        upsampled[(K-1)*ratio:] = vla_states[-1]
    else:
        raise ValueError(f"Unknown upsample method: {method}")

    return upsampled


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    cfg = ESNConfig(reservoir_size=500, spectral_radius=0.95, sparsity=0.9, seed=42)
    esn = ESNBridge(cfg)

    print(f"ESN Bridge | N={cfg.reservoir_size} | params={esn.n_trainable_params:,}")
    print(f"Estimated readout latency: {esn.latency_estimate_ms:.4f} ms  (vs OpenVLA ~380 ms)")

    # Quick smoke test: random inputs → outputs
    T_vla = 100
    vla_states = np.random.randn(T_vla, OPENVLA_HIDDEN_DIM).astype(np.float32)
    gt_actions = np.random.randn(T_vla * 31, G1_DOF).astype(np.float32)

    up = upsample_vla_states(vla_states, method="hold")
    gt_matched = gt_actions[:up.shape[0]]

    metrics = esn.fit(up, gt_matched)
    print(f"Training | RMSE={metrics['rmse']:.4f} | R²={metrics['r2']:.4f}")

    esn.reset()
    esn.update_from_vla(vla_states[0])
    cmd = esn.step()
    print(f"Output command shape: {cmd.shape}  (expected: ({G1_DOF},))")
    print("step2_esn_bridge.py smoke test PASSED")
