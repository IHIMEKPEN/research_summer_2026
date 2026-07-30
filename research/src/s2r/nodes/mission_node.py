"""Mission manager for long-horizon tasks (e.g. bring_pen).

Tracks phase transitions, exposes a timeline to the GUI, and feeds the
current instruction/phase into reasoning + VLA.
"""

from __future__ import annotations

import time
from typing import Any

from s2r.core.node import Node
from s2r.core.schemas import Topic
from s2r.core.zmq_bus import Publisher, Subscriber


PHASES = [
    "idle",
    "explore",
    "locate",
    "approach",
    "grasp",
    "return",
    "handoff",
    "done",
]


class MissionNode(Node):
    name = "mission"

    def setup(self) -> None:
        z = self.zmq_cfg
        m = self.cfg.get("mission", {})
        self.hz = float(m.get("rate_hz", 5.0))
        self.instruction = str(m.get("instruction", "Go bring me a pen from the table in the lab"))
        self.task = str(m.get("task", "bring_pen"))
        self.phase = "explore"
        self.timeline: list[dict[str, Any]] = []
        self._holding = False
        self._last_intent = ""

        self.perc_sub = Subscriber(z["perception_pub"], topics=[Topic.PERCEPTION], zmq_cfg=z, conflate=True)
        self.dec_sub = Subscriber(z["decision_pub"], topics=[Topic.DECISION], zmq_cfg=z, conflate=True)
        self.pub = Publisher(z["mission_pub"], zmq_cfg=z, source=self.name, conflate=True)
        self.gui_pub = Publisher(z["gui_pub"], zmq_cfg=z, bind=False, source=self.name)

    def _advance(self, intent: str, perception: dict[str, Any]) -> None:
        mapping = {
            "explore": "explore",
            "locate_pen": "locate",
            "approach_table": "approach",
            "grasp_pen": "grasp",
            "return_to_user": "return",
            "handoff": "handoff",
            "hold": self.phase,
            "idle": "idle",
        }
        new_phase = mapping.get(intent, self.phase)
        if perception.get("holding_pen"):
            self._holding = True
        if self._holding and new_phase in {"explore", "locate", "approach"}:
            new_phase = "return"
        if intent == "handoff":
            new_phase = "handoff"
        if self.phase == "handoff" and intent in {"idle", "hold"}:
            new_phase = "done"
        if new_phase != self.phase:
            self.timeline.append(
                {
                    "ts": time.time(),
                    "from": self.phase,
                    "to": new_phase,
                    "intent": intent,
                    "reason": perception.get("caption", ""),
                }
            )
            self.phase = new_phase
        self._last_intent = intent

    def step(self) -> None:
        perc: dict[str, Any] = {}
        env = self.perc_sub.recv(timeout_ms=0)
        if env is not None:
            perc = env.payload
            self._holding = bool(perc.get("holding_pen", self._holding))

        dec = self.dec_sub.recv(timeout_ms=0)
        if dec is not None:
            self._advance(str(dec.payload.get("intent", self.phase)), perc)

        payload = {
            "task": self.task,
            "instruction": self.instruction,
            "phase": self.phase,
            "holding_pen": self._holding,
            "last_intent": self._last_intent,
            "timeline": self.timeline[-20:],
            "ts": time.time(),
        }
        self.cfg.setdefault("_runtime", {})["mission"] = payload
        self.pub.publish(Topic.MISSION, payload)
        self.gui_pub.publish(Topic.MISSION, payload)

    def teardown(self) -> None:
        self.perc_sub.close()
        self.dec_sub.close()
        self.pub.close()
        self.gui_pub.close()

    def run(self, hz: float | None = None) -> None:
        if hz is None:
            hz = float(self.cfg.get("mission", {}).get("rate_hz", 5.0))
        super().run(hz=hz)
