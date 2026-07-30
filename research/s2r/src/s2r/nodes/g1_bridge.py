"""Unitree G1 Edu robot bridge.

Uses `unitree_sdk2py` when installed; otherwise runs a safe stub that still
participates in the ZMQ control loop for bring-up without hardware.

High-level mode (recommended for Edu locomotion):
  - LocoClient for walk/stand
  - Arm SDK / joint targets for manipulation

Low-level mode (advanced):
  - Release motion services, stream LowCmd @ control_hz
  - Requires careful PD gains and CRC (see docs/UNITREE_G1_EDU.md)
"""

from __future__ import annotations

import time
from typing import Any

from rich.console import Console

from s2r.core.node import Node
from s2r.core.schemas import Topic
from s2r.core.zmq_bus import Publisher, Subscriber
from s2r.utils.timing import RateMeter

console = Console()


class G1BridgeNode(Node):
    name = "g1_bridge"

    def setup(self) -> None:
        z = self.zmq_cfg
        g1 = self.cfg.get("g1", {})
        self.hz = float(g1.get("control_hz", self.cfg.get("robot", {}).get("control_hz", 100)))
        self.iface = str(g1.get("iface", "eth0"))
        self.mode = str(g1.get("mode", "high_level"))  # high_level | low_level | stub
        self.network = str(g1.get("network", "192.168.123.164"))
        self.mock = bool(g1.get("mock", True))
        self._client: Any = None
        self._warned = False
        self._last_q: list[float] = []
        self._rate = RateMeter()

        self.cmd_sub = Subscriber(z["joint_cmd_pub"], topics=[Topic.JOINT_CMD], zmq_cfg=z, conflate=True)
        self.dec_sub = Subscriber(z["decision_pub"], topics=[Topic.DECISION], zmq_cfg=z, conflate=True)
        self.mission_sub = Subscriber(z["mission_pub"], topics=[Topic.MISSION], zmq_cfg=z, conflate=True)
        self.state_pub = Publisher(z["state_pub"], zmq_cfg=z, bind=False, source=self.name, conflate=True)
        self.gui_pub = Publisher(z["gui_pub"], zmq_cfg=z, bind=False, source=self.name)

        self._intent = "explore"
        self._allow = True
        if not self.mock:
            self._connect()

    def _connect(self) -> None:
        try:
            # Official Python SDK package name
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize  # type: ignore

            ChannelFactoryInitialize(0, self.iface)
            if self.mode == "high_level":
                try:
                    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient  # type: ignore

                    self._client = LocoClient()
                    self._client.SetTimeout(10.0)
                    self._client.Init()
                    console.print(f"[green]g1_bridge[/] LocoClient connected iface={self.iface}")
                except Exception as e:
                    console.print(f"[yellow]g1_bridge[/] LocoClient unavailable ({e}); using stub sends")
                    self.mock = True
            else:
                console.print(
                    "[yellow]g1_bridge[/] low_level mode selected — "
                    "implement LowCmd streaming per Unitree docs before enabling on hardware."
                )
                self.mock = True
        except Exception as e:
            console.print(f"[yellow]g1_bridge[/] unitree_sdk2py not available ({e}); stub mode")
            self.mock = True

    def _apply_high_level(self, q: list[float]) -> None:
        """Map intents + residual arm targets to G1 loco/arm APIs."""
        if self._client is None or not self._allow:
            return
        # Example loco primitives driven by mission intent
        try:
            if self._intent in {"explore", "return_to_user", "return"}:
                # Gentle forward / turn — tune on hardware
                vx = 0.15 if self._intent != "return_to_user" else 0.12
                self._client.Move(vx, 0.0, 0.1 if int(time.time()) % 2 == 0 else -0.1)
            elif self._intent in {"approach_table", "approach", "locate_pen"}:
                self._client.Move(0.08, 0.0, 0.0)
            elif self._intent in {"grasp_pen", "handoff", "hold"}:
                self._client.Move(0.0, 0.0, 0.0)
            # Arm joint targeting depends on installed Arm SDK helpers / custom retargeting.
            # Keep q available for a site-specific arm controller hook:
            self._last_q = q
        except Exception as e:
            if not self._warned:
                console.print(f"[red]g1_bridge[/] command failed: {e}")
                self._warned = True

    def step(self) -> None:
        dec = self.dec_sub.recv(timeout_ms=0)
        if dec is not None:
            self._intent = str(dec.payload.get("intent", self._intent))
            self._allow = bool(dec.payload.get("allow_motion", True))

        _ = self.mission_sub.recv(timeout_ms=0)

        env = self.cmd_sub.recv(timeout_ms=5)
        if env is None:
            return
        q = list(env.payload.get("q", []))
        self._last_q = q

        if self.mock:
            if not self._warned:
                console.print(
                    "[yellow]g1_bridge[/] stub mode — commands accepted, not sent to Unitree hardware. "
                    "Set g1.mock:false and install unitree_sdk2py on the robot PC."
                )
                self._warned = True
        else:
            self._apply_high_level(q)

        self._rate.tick()
        if self._ticks % int(max(1, self.hz)) == 0:
            self.gui_pub.publish(
                Topic.METRICS,
                {
                    "node": self.name,
                    "latency_ms": 0.0,
                    "hz": self._rate.hz,
                    "queue_depth": 0,
                    "extras": {
                        "mock": self.mock,
                        "mode": self.mode,
                        "intent": self._intent,
                        "iface": self.iface,
                        "network": self.network,
                    },
                },
            )

    def teardown(self) -> None:
        if self._client is not None and not self.mock:
            try:
                self._client.Move(0.0, 0.0, 0.0)
            except Exception:
                pass
        self.cmd_sub.close()
        self.dec_sub.close()
        self.mission_sub.close()
        self.state_pub.close()
        self.gui_pub.close()

    def run(self, hz: float | None = None) -> None:
        if hz is None:
            hz = float(self.cfg.get("g1", {}).get("control_hz", 100))
        super().run(hz=hz)
