"""Control bridge without ESN (ablation baselines).

Modes for research comparison against the ESN dynamic engine:
- raw:   publish joint_cmd only when a new VLA token arrives (~2 Hz) — no buffer
- zoh:   zero-order hold: repeat last token at target_hz (buffer, no dynamics)
- linear: linear interpolate between consecutive tokens at target_hz

Use this node instead of `esn` when `control_engine: passthrough`.
"""

from __future__ import annotations
from s2r.robot import G1_DOF

import os
import time

import numpy as np

from s2r.core.node import Node
from s2r.core.schemas import ActionToken, JointCommand, Topic
from s2r.core.zmq_bus import Publisher, Subscriber
from s2r.utils.timing import LatencyTracker, RateMeter


class PassthroughControlNode(Node):
    name = "passthrough"

    def setup(self) -> None:
        z = self.zmq_cfg
        pc = self.cfg.get("passthrough", {})
        robot = self.cfg.get("robot", {})
        # Env override lets orchestrator propagate ablation mode to child processes
        self.mode = str(os.environ.get("S2R_PASSTHROUGH_MODE") or pc.get("mode", "zoh")).lower()
        self.n_joints = int(robot.get("n_joints", G1_DOF))
        self.target_hz = float(pc.get("target_hz", self.cfg.get("esn", {}).get("target_hz", 100)))
        self.vla_hz = float(self.cfg.get("vla", {}).get("rate_hz", 2.0))

        self.sub = Subscriber(z["action_token_pub"], topics=[Topic.ACTION_TOKEN], zmq_cfg=z, conflate=True)
        self.state_sub = Subscriber(z["state_pub"], topics=[Topic.STATE], zmq_cfg=z, conflate=True)
        self.decision_sub = Subscriber(z["decision_pub"], topics=[Topic.DECISION], zmq_cfg=z, conflate=True)
        self.pub = Publisher(z["joint_cmd_pub"], zmq_cfg=z, source=self.name, conflate=True)
        self.metrics_pub = Publisher(z["gui_pub"], zmq_cfg=z, bind=False, source=self.name)

        self._token = np.zeros(self.n_joints, dtype=np.float64)
        self._prev_token = np.zeros(self.n_joints, dtype=np.float64)
        self._q = np.zeros(self.n_joints, dtype=np.float64)
        self._allow = True
        self._token_ts = 0.0
        self._prev_token_ts = 0.0
        self._have_token = False
        self._rate = RateMeter()
        self._lat = LatencyTracker()
        self._upsample_factor = 1.0 if self.mode == "raw" else self.target_hz / max(self.vla_hz, 1e-6)

    def _limits_clip(self, cmd: np.ndarray) -> np.ndarray:
        limits = self.cfg.get("robot", {}).get("joint_limits", {})
        if not limits:
            return cmd
        lo = np.asarray(limits.get("min", [-np.pi] * self.n_joints), dtype=np.float64)
        hi = np.asarray(limits.get("max", [np.pi] * self.n_joints), dtype=np.float64)
        return np.clip(cmd, lo, hi)

    def _compute_cmd(self) -> np.ndarray | None:
        if not self._have_token:
            return None
        if self.mode == "raw":
            # Caller publishes only on new token
            return self._token.copy()
        if self.mode == "linear" and self._prev_token_ts > 0:
            span = max(1e-3, 1.0 / max(self.vla_hz, 1e-6))
            alpha = min(1.0, max(0.0, (time.time() - self._token_ts) / span))
            # interpolate from previous -> current across the token period
            # using age since current token as progress into NEXT interval is awkward;
            # better: progress from prev_ts to token_ts, then hold current.
            denom = max(self._token_ts - self._prev_token_ts, 1e-3)
            a = min(1.0, max(0.0, (time.time() - self._prev_token_ts) / denom))
            if a >= 1.0:
                return self._token.copy()
            return (1.0 - a) * self._prev_token + a * self._token
        # zoh default
        return self._token.copy()

    def step(self) -> None:
        new_token = False
        env = self.sub.recv(timeout_ms=0)
        if env is not None:
            tok = ActionToken.model_validate(env.payload)
            self._prev_token = self._token.copy()
            self._prev_token_ts = self._token_ts if self._token_ts else env.ts
            self._token = np.asarray(tok.action, dtype=np.float64)
            self._token_ts = env.ts
            self._have_token = True
            new_token = True
            self._lat.add((time.time() - env.ts) * 1000.0)

        st = self.state_sub.recv(timeout_ms=0)
        if st is not None and st.payload.get("joint_pos") is not None:
            self._q = np.asarray(st.payload["joint_pos"], dtype=np.float64)

        dec = self.decision_sub.recv(timeout_ms=0)
        if dec is not None:
            self._allow = bool(dec.payload.get("allow_motion", True))

        if self.mode == "raw" and not new_token:
            return

        cmd = self._compute_cmd()
        if cmd is None:
            return
        if not self._allow:
            cmd = self._q
        cmd = self._limits_clip(cmd)

        payload = JointCommand(
            q=cmd.tolist(),
            dq=((cmd - self._q) * (self.target_hz if self.mode != "raw" else self.vla_hz)).tolist(),
            source=f"passthrough_{self.mode}",
            upsample_factor=self._upsample_factor,
        ).model_dump()
        payload["engine"] = "passthrough"
        payload["mode"] = self.mode
        self.pub.publish(Topic.JOINT_CMD, payload)
        self._q = cmd
        self._rate.tick()

        period = 1 if self.mode == "raw" else int(max(1, self.target_hz))
        if self._ticks % period == 0:
            self.metrics_pub.publish(
                Topic.METRICS,
                {
                    "node": self.name,
                    "latency_ms": self._lat.mean_ms,
                    "hz": self._rate.hz,
                    "queue_depth": 0,
                    "extras": {
                        "engine": "passthrough",
                        "mode": self.mode,
                        "p95_ms": self._lat.p95_ms,
                        "token_age_ms": (time.time() - self._token_ts) * 1000.0 if self._token_ts else 0.0,
                    },
                },
            )

    def teardown(self) -> None:
        self.sub.close()
        self.state_sub.close()
        self.decision_sub.close()
        self.pub.close()
        self.metrics_pub.close()

    def run(self, hz: float | None = None) -> None:
        if hz is None:
            if self.cfg.get("passthrough", {}).get("mode", "zoh") == "raw":
                # spin fast enough to catch tokens; publish gated in step()
                hz = float(self.cfg.get("vla", {}).get("rate_hz", 2.0)) * 10.0
            else:
                hz = float(self.cfg.get("passthrough", {}).get("target_hz", 100))
        super().run(hz=hz)
