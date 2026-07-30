"""Vision-Language-Action (VLA) node producing sparse action tokens (~2 Hz).

Default mock policy is mission-conditioned (bring_pen phases). Replace `infer()`
with OpenVLA / π0 / GR00T / custom checkpoint when ready.
"""

from __future__ import annotations
from s2r.robot import G1_DOF

import math
import time

import numpy as np

from s2r.core.node import Node
from s2r.core.schemas import ActionToken, Topic
from s2r.core.zmq_bus import Publisher, Subscriber
from s2r.utils.timing import RateMeter


class VLANode(Node):
    name = "vla"

    def setup(self) -> None:
        z = self.zmq_cfg
        vla = self.cfg.get("vla", {})
        self.rate_hz = float(vla.get("rate_hz", 2.0))
        self.action_dim = int(vla.get("action_dim", G1_DOF))
        self.mock = bool(vla.get("mock", True))
        self._t0 = time.time()
        self._goal = "explore"
        self._phase = "explore"
        self._objects: list[str] = []
        self._q = np.zeros(self.action_dim, dtype=np.float64)

        self.state_sub = Subscriber(z["state_pub"], topics=[Topic.STATE], zmq_cfg=z, conflate=True)
        self.decision_sub = Subscriber(z["decision_pub"], topics=[Topic.DECISION], zmq_cfg=z, conflate=True)
        self.perc_sub = Subscriber(z["perception_pub"], topics=[Topic.PERCEPTION], zmq_cfg=z, conflate=True)
        self.mission_sub = Subscriber(z["mission_pub"], topics=[Topic.MISSION], zmq_cfg=z, conflate=True)
        self.pub = Publisher(z["action_token_pub"], zmq_cfg=z, source=self.name, conflate=True)
        self.gui_pub = Publisher(z["gui_pub"], zmq_cfg=z, bind=False, source=self.name)
        self._rate = RateMeter()
        self._allow = True

    def infer(self, joint_pos: np.ndarray, goal: str) -> ActionToken:
        if not self.mock:
            raise NotImplementedError("Plug your VLA model into VLANode.infer()")

        t = time.time() - self._t0
        phase = 2.0 * math.pi * 0.05 * t
        # Intent-conditioned joint-space targets for demo / sim
        bias = {
            "explore": 0.35,
            "locate_pen": 0.25,
            "approach_table": 0.2,
            "grasp_pen": 0.1,
            "return_to_user": 0.3,
            "handoff": 0.05,
            "hold": 0.0,
        }.get(goal, 0.2)
        action = []
        for i in range(self.action_dim):
            amp = bias + 0.05 * i
            # Grasp closes "hand" dims toward a reach pose
            if goal == "grasp_pen" and i >= max(0, self.action_dim - 2):
                action.append(0.9)
            elif goal == "handoff" and i == 0:
                action.append(0.15 * math.sin(phase))
            else:
                action.append(float(amp * math.sin(phase + 0.4 * i)))
        conf = 0.9 if self._objects else 0.55
        if "pen" in self._objects:
            conf = 0.92
        latent = [
            math.sin(phase),
            math.cos(phase),
            float(np.linalg.norm(joint_pos)),
            float(len(self._objects)),
        ]
        return ActionToken(action=action, confidence=conf, goal=goal, latent=latent)

    def step(self) -> None:
        st = self.state_sub.recv(timeout_ms=0)
        if st is not None:
            self._q = np.asarray(st.payload.get("joint_pos", self._q), dtype=np.float64)

        dec = self.decision_sub.recv(timeout_ms=0)
        if dec is not None:
            self._allow = bool(dec.payload.get("allow_motion", True))
            intent = dec.payload.get("intent")
            if intent:
                self._goal = str(intent)

        perc = self.perc_sub.recv(timeout_ms=0)
        if perc is not None:
            self._objects = list(perc.payload.get("objects_of_interest", []))

        miss = self.mission_sub.recv(timeout_ms=0)
        if miss is not None:
            self._phase = str(miss.payload.get("phase", self._phase))

        if not self._allow:
            tok = ActionToken(action=self._q.tolist(), confidence=0.0, goal="hold", latent=[])
        else:
            tok = self.infer(self._q, self._goal)

        payload = tok.model_dump()
        payload["phase"] = self._phase
        payload["objects"] = self._objects
        self.pub.publish(Topic.ACTION_TOKEN, payload)
        self.gui_pub.publish(Topic.ACTION_TOKEN, payload)
        self._rate.tick()

    def teardown(self) -> None:
        self.state_sub.close()
        self.decision_sub.close()
        self.perc_sub.close()
        self.mission_sub.close()
        self.pub.close()
        self.gui_pub.close()

    def run(self, hz: float | None = None) -> None:
        if hz is None:
            hz = float(self.cfg.get("vla", {}).get("rate_hz", 2.0))
        super().run(hz=hz)
