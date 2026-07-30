"""Simulation bridge node.

Forwards high-rate joint commands into a sim backend adapter and can mirror
state. Default adapter is a no-op sink (state_publisher already integrates).
Replace `SimBackend` with MuJoCo / Isaac / PyBullet hooks.
"""

from __future__ import annotations

from s2r.core.node import Node
from s2r.core.schemas import Topic
from s2r.core.zmq_bus import Publisher, Subscriber
from s2r.utils.timing import RateMeter


class SimBackend:
    def __init__(self) -> None:
        self.last_q: list[float] = []

    def apply_command(self, q: list[float]) -> None:
        self.last_q = list(q)

    def close(self) -> None:
        pass


class SimBridgeNode(Node):
    name = "sim_bridge"

    def setup(self) -> None:
        z = self.zmq_cfg
        self.hz = float(self.cfg.get("robot", {}).get("control_hz", 100))
        self.backend = SimBackend()
        self.sub = Subscriber(z["joint_cmd_pub"], topics=[Topic.JOINT_CMD], zmq_cfg=z, conflate=True)
        self.gui_pub = Publisher(z["gui_pub"], zmq_cfg=z, bind=False, source=self.name)
        self._rate = RateMeter()

    def step(self) -> None:
        env = self.sub.recv(timeout_ms=5)
        if env is None:
            return
        q = env.payload.get("q", [])
        self.backend.apply_command(q)
        self._rate.tick()
        if self._ticks % int(max(1, self.hz)) == 0:
            self.gui_pub.publish(
                Topic.METRICS,
                {
                    "node": self.name,
                    "latency_ms": max(0.0, (self._ticks and 0) or 0.0),
                    "hz": self._rate.hz,
                    "queue_depth": 0,
                    "extras": {"last_q_norm": float(sum(x * x for x in q) ** 0.5) if q else 0.0},
                },
            )

    def teardown(self) -> None:
        self.backend.close()
        self.sub.close()
        self.gui_pub.close()

    def run(self, hz: float | None = None) -> None:
        if hz is None:
            hz = float(self.cfg.get("robot", {}).get("control_hz", 100))
        super().run(hz=hz)
