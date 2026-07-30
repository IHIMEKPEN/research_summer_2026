"""Echo State Network (ESN) dynamic engine.

Upsamples sparse VLA action tokens (~2 Hz) into smooth joint commands (100 Hz+).

Why ESN for this loop:
- Extremely fast inference (one sparse matrix-vector multiply + readout)
- Built-in temporal memory / dynamics for smooth trajectories
- Trainable with ridge regression on collected demos
- Deterministic low-latency suitable for real-time control
"""

from __future__ import annotations
from s2r.robot import G1_DOF

from pathlib import Path
from typing import Any

import numpy as np

from s2r.core.node import Node
from s2r.core.schemas import ActionToken, JointCommand, Topic
from s2r.core.zmq_bus import Publisher, Subscriber
from s2r.utils.timing import LatencyTracker, RateMeter


class EchoStateNetwork:
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        reservoir_size: int = 300,
        spectral_radius: float = 0.9,
        sparsity: float = 0.1,
        input_scale: float = 0.5,
        leaking_rate: float = 0.3,
        seed: int = 42,
    ) -> None:
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.input_scale = input_scale
        self.leaking_rate = leaking_rate
        rng = np.random.default_rng(seed)

        self.W_in = (rng.uniform(-1, 1, size=(reservoir_size, n_inputs)) * input_scale).astype(np.float64)
        W = rng.uniform(-1, 1, size=(reservoir_size, reservoir_size))
        mask = rng.random((reservoir_size, reservoir_size)) > sparsity
        W[mask] = 0.0
        # Scale to desired spectral radius
        eig = np.linalg.eigvals(W)
        radius = np.max(np.abs(eig))
        if radius > 1e-8:
            W *= spectral_radius / radius
        self.W = W.astype(np.float64)
        self.W_out = np.zeros((n_outputs, reservoir_size + n_inputs), dtype=np.float64)
        self.state = np.zeros(reservoir_size, dtype=np.float64)
        self._trained = False

    def reset(self) -> None:
        self.state[:] = 0.0

    def update(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=np.float64).reshape(-1)
        pre = self.W_in @ u + self.W @ self.state
        x_tilde = np.tanh(pre)
        a = self.leaking_rate
        self.state = (1.0 - a) * self.state + a * x_tilde
        ext = np.concatenate([self.state, u])
        y = self.W_out @ ext
        return y

    def collect_states(self, inputs: np.ndarray, washout: int = 20) -> tuple[np.ndarray, np.ndarray]:
        self.reset()
        xs = []
        us = []
        for i, u in enumerate(inputs):
            _ = self.update(u)
            if i >= washout:
                xs.append(self.state.copy())
                us.append(np.asarray(u, dtype=np.float64).reshape(-1))
        X = np.concatenate([np.asarray(xs), np.asarray(us)], axis=1)
        return X, np.asarray(us)

    def fit_ridge(self, inputs: np.ndarray, targets: np.ndarray, washout: int = 20, reg: float = 1e-6) -> float:
        """Train readout with ridge regression. inputs/targets: [T, D]."""
        self.reset()
        X_rows = []
        Y_rows = []
        for i, (u, y) in enumerate(zip(inputs, targets)):
            _ = self.update(u)
            if i >= washout:
                ext = np.concatenate([self.state, np.asarray(u, dtype=np.float64).reshape(-1)])
                X_rows.append(ext)
                Y_rows.append(np.asarray(y, dtype=np.float64).reshape(-1))
        X = np.asarray(X_rows)
        Y = np.asarray(Y_rows)
        # W_out = Y X^T (X X^T + r I)^-1
        xtx = X.T @ X
        xtx.flat[:: xtx.shape[0] + 1] += reg
        self.W_out = (Y.T @ X) @ np.linalg.inv(xtx)
        self._trained = True
        pred = X @ self.W_out.T
        mse = float(np.mean((pred - Y) ** 2))
        return mse

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            W_in=self.W_in,
            W=self.W,
            W_out=self.W_out,
            state=self.state,
            n_inputs=self.n_inputs,
            n_outputs=self.n_outputs,
            reservoir_size=self.reservoir_size,
            spectral_radius=self.spectral_radius,
            sparsity=self.sparsity,
            input_scale=self.input_scale,
            leaking_rate=self.leaking_rate,
            trained=np.array([1 if self._trained else 0]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "EchoStateNetwork":
        data = np.load(path, allow_pickle=False)
        obj = cls(
            n_inputs=int(data["n_inputs"]),
            n_outputs=int(data["n_outputs"]),
            reservoir_size=int(data["reservoir_size"]),
            spectral_radius=float(data["spectral_radius"]),
            sparsity=float(data["sparsity"]),
            input_scale=float(data["input_scale"]),
            leaking_rate=float(data["leaking_rate"]),
        )
        obj.W_in = data["W_in"]
        obj.W = data["W"]
        obj.W_out = data["W_out"]
        obj.state = data["state"]
        obj._trained = bool(data["trained"][0])
        return obj


class ESNUpsampleNode(Node):
    """Consumes action tokens and emits high-rate joint commands."""

    name = "esn"

    def setup(self) -> None:
        z = self.zmq_cfg
        esn_cfg = self.cfg.get("esn", {})
        robot_cfg = self.cfg.get("robot", {})
        self.n_joints = int(robot_cfg.get("n_joints", G1_DOF))
        self.target_hz = float(esn_cfg.get("target_hz", 100))
        self.model_path = esn_cfg.get("model_path", "data/models/esn_upsample.npz")

        if Path(self.model_path).exists():
            self.esn = EchoStateNetwork.load(self.model_path)
        else:
            self.esn = EchoStateNetwork(
                n_inputs=self.n_joints,
                n_outputs=self.n_joints,
                reservoir_size=int(esn_cfg.get("reservoir_size", 300)),
                spectral_radius=float(esn_cfg.get("spectral_radius", 0.9)),
                sparsity=float(esn_cfg.get("sparsity", 0.1)),
                input_scale=float(esn_cfg.get("input_scale", 0.5)),
                leaking_rate=float(esn_cfg.get("leaking_rate", 0.3)),
            )
            # Identity-ish bootstrap readout so pipeline is usable before training
            d = self.n_joints
            # Map input part of extended state directly to output
            self.esn.W_out[:, -d:] = np.eye(d)
            self.esn._trained = False

        self.sub = Subscriber(
            z["action_token_pub"],
            topics=[Topic.ACTION_TOKEN],
            zmq_cfg=z,
            conflate=True,
        )
        self.state_sub = Subscriber(
            z["state_pub"],
            topics=[Topic.STATE],
            zmq_cfg=z,
            conflate=True,
        )
        self.decision_sub = Subscriber(
            z["decision_pub"],
            topics=[Topic.DECISION],
            zmq_cfg=z,
            conflate=True,
        )
        self.pub = Publisher(z["joint_cmd_pub"], zmq_cfg=z, source=self.name, conflate=True)
        self.metrics_pub = Publisher(z["gui_pub"], zmq_cfg=z, bind=False, source=self.name)

        self._token: np.ndarray = np.zeros(self.n_joints, dtype=np.float64)
        self._q: np.ndarray = np.zeros(self.n_joints, dtype=np.float64)
        self._allow = True
        self._token_ts = 0.0
        self._rate = RateMeter()
        self._lat = LatencyTracker()
        self._upsample_factor = self.target_hz / float(self.cfg.get("vla", {}).get("rate_hz", 2.0))

    def step(self) -> None:
        # Pull latest sparse token / gates / state (non-blocking)
        env = self.sub.recv(timeout_ms=0)
        if env is not None:
            tok = ActionToken.model_validate(env.payload)
            self._token = np.asarray(tok.action, dtype=np.float64)
            self._token_ts = env.ts
            self._lat.add((time_now() - env.ts) * 1000.0)

        st = self.state_sub.recv(timeout_ms=0)
        if st is not None:
            jp = st.payload.get("joint_pos")
            if jp is not None:
                self._q = np.asarray(jp, dtype=np.float64)

        dec = self.decision_sub.recv(timeout_ms=0)
        if dec is not None:
            self._allow = bool(dec.payload.get("allow_motion", True))

        # Drive reservoir with current token target; blend toward token from current q
        # Input encodes desired action residual relative to current joints
        u = self._token
        y = self.esn.update(u)
        # Soft blend for untrained bootstrap smoothness
        alpha = 0.35 if not self.esn._trained else 1.0
        cmd = (1.0 - alpha) * self._q + alpha * y
        if not self._allow:
            cmd = self._q  # hold

        # Clamp to limits
        limits = self.cfg.get("robot", {}).get("joint_limits", {})
        if limits:
            lo = np.asarray(limits.get("min", [-np.pi] * self.n_joints), dtype=np.float64)
            hi = np.asarray(limits.get("max", [np.pi] * self.n_joints), dtype=np.float64)
            cmd = np.clip(cmd, lo, hi)

        payload = JointCommand(
            q=cmd.tolist(),
            dq=((cmd - self._q) * self.target_hz).tolist(),
            source="esn",
            upsample_factor=self._upsample_factor,
        ).model_dump()
        payload["engine"] = "esn"
        payload["mode"] = "reservoir"
        self.pub.publish(Topic.JOINT_CMD, payload)
        self._q = cmd
        self._rate.tick()

        if self._ticks % int(max(1, self.target_hz)) == 0:
            self.metrics_pub.publish(
                Topic.METRICS,
                {
                    "node": self.name,
                    "latency_ms": self._lat.mean_ms,
                    "hz": self._rate.hz,
                    "queue_depth": 0,
                    "extras": {
                        "engine": "esn",
                        "mode": "reservoir",
                        "trained": self.esn._trained,
                        "p95_ms": self._lat.p95_ms,
                        "token_age_ms": (time_now() - self._token_ts) * 1000.0 if self._token_ts else 0.0,
                    },
                },
            )

    def teardown(self) -> None:
        self.sub.close()
        self.state_sub.close()
        self.decision_sub.close()
        self.pub.close()
        self.metrics_pub.close()

    def run(self, hz: float | None = None) -> None:  # noqa: D401
        if hz is None:
            hz = float(self.cfg.get("esn", {}).get("target_hz", 100))
        super().run(hz=hz)


def time_now() -> float:
    import time

    return time.time()
