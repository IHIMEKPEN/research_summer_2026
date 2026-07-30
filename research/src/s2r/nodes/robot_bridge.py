"""Real robot bridge node.

Reads hardware state / writes joint commands. Default is a safe stub that
echoes commands and warns that no hardware driver is attached.
"""

from __future__ import annotations

from rich.console import Console

from s2r.core.node import Node
from s2r.core.schemas import Topic
from s2r.core.zmq_bus import Publisher, Subscriber
from s2r.utils.timing import RateMeter

console = Console()


class RobotBridgeNode(Node):
    name = "robot_bridge"

    def setup(self) -> None:
        z = self.zmq_cfg
        self.hz = float(self.cfg.get("robot", {}).get("control_hz", 100))
        self.sub = Subscriber(z["joint_cmd_pub"], topics=[Topic.JOINT_CMD], zmq_cfg=z, conflate=True)
        self.state_pub = Publisher(z["state_pub"], zmq_cfg=z, bind=False, source=self.name, conflate=True)
        self.gui_pub = Publisher(z["gui_pub"], zmq_cfg=z, bind=False, source=self.name)
        self._rate = RateMeter()
        self._warned = False

    def write_hardware(self, q: list[float]) -> None:
        if not self._warned:
            console.print(
                "[yellow]robot_bridge[/]: no hardware driver attached; "
                "commands are accepted but not sent to a real robot."
            )
            self._warned = True
        # Integrate your driver here (CAN/EtherCAT/ROS2 bridge/etc.)

    def step(self) -> None:
        env = self.sub.recv(timeout_ms=5)
        if env is None:
            return
        q = env.payload.get("q", [])
        self.write_hardware(q)
        self._rate.tick()
        if self._ticks % int(max(1, self.hz)) == 0:
            self.gui_pub.publish(
                Topic.METRICS,
                {
                    "node": self.name,
                    "latency_ms": 0.0,
                    "hz": self._rate.hz,
                    "queue_depth": 0,
                    "extras": {"driver": "stub"},
                },
            )

    def teardown(self) -> None:
        self.sub.close()
        self.state_pub.close()
        self.gui_pub.close()

    def run(self, hz: float | None = None) -> None:
        if hz is None:
            hz = float(self.cfg.get("robot", {}).get("control_hz", 100))
        super().run(hz=hz)
