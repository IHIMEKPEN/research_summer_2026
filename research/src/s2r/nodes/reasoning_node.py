"""Reasoning / decision node backed by open-source Qwen (or rule fallback)."""

from __future__ import annotations
from s2r.robot import G1_DOF

import time

import numpy as np

from s2r.core.node import Node
from s2r.core.schemas import Decision, Topic
from s2r.core.zmq_bus import Publisher, Subscriber
from s2r.models.base import PerceptionFrame, Detection
from s2r.models.registry import build_reasoner
from s2r.utils.timing import RateMeter


class ReasoningNode(Node):
    name = "reasoning"

    def setup(self) -> None:
        z = self.zmq_cfg
        rcfg = self.cfg.get("reasoning", {})
        self.rate_hz = float(rcfg.get("rate_hz", 5.0))
        self.reasoner = build_reasoner(self.cfg)
        self.instruction = str(self.cfg.get("mission", {}).get("instruction", "bring me a pen"))
        self._q = np.zeros(int(self.cfg.get("robot", {}).get("n_joints", G1_DOF)), dtype=np.float64)
        self._perception = PerceptionFrame()
        self._phase = "explore"
        self._holding = False

        self.state_sub = Subscriber(z["state_pub"], topics=[Topic.STATE], zmq_cfg=z, conflate=True)
        self.token_sub = Subscriber(z["action_token_pub"], topics=[Topic.ACTION_TOKEN], zmq_cfg=z, conflate=True)
        self.perc_sub = Subscriber(z["perception_pub"], topics=[Topic.PERCEPTION], zmq_cfg=z, conflate=True)
        self.mission_sub = Subscriber(z["mission_pub"], topics=[Topic.MISSION], zmq_cfg=z, conflate=True)
        self.pub = Publisher(z["decision_pub"], zmq_cfg=z, source=self.name, conflate=True)
        self.gui_pub = Publisher(z["gui_pub"], zmq_cfg=z, bind=False, source=self.name)
        self._rate = RateMeter()

    def _near_limit(self) -> bool:
        limits = self.cfg.get("robot", {}).get("joint_limits", {})
        lo = np.asarray(limits.get("min", [-3.0] * len(self._q)), dtype=np.float64)
        hi = np.asarray(limits.get("max", [3.0] * len(self._q)), dtype=np.float64)
        margin = np.minimum(self._q - lo, hi - self._q)
        return bool(np.any(margin < 0.15))

    def step(self) -> None:
        st = self.state_sub.recv(timeout_ms=0)
        if st is not None:
            self._q = np.asarray(st.payload.get("joint_pos", self._q), dtype=np.float64)

        perc = self.perc_sub.recv(timeout_ms=0)
        if perc is not None:
            p = perc.payload
            self._holding = bool(p.get("holding_pen", False))
            dets = []
            for d in p.get("detections", []):
                if isinstance(d, dict):
                    dets.append(
                        Detection(
                            label=str(d.get("label", "object")),
                            confidence=float(d.get("confidence", 0.0)),
                            xyxy=list(d.get("xyxy", [0, 0, 1, 1])),
                            track_id=d.get("track_id"),
                        )
                    )
                else:
                    dets.append(d)
            self._perception = PerceptionFrame(
                detections=dets,
                caption=str(p.get("caption", "")),
                objects_of_interest=list(p.get("objects_of_interest", [])),
                scene_tags=list(p.get("scene_tags", [])),
            )

        miss = self.mission_sub.recv(timeout_ms=0)
        if miss is not None:
            self._phase = str(miss.payload.get("phase", self._phase))
            self.instruction = str(miss.payload.get("instruction", self.instruction))

        # Drain unused token sub to avoid socket buildup
        _ = self.token_sub.recv(timeout_ms=0)

        out = self.reasoner.plan(
            instruction=self.instruction,
            perception=self._perception,
            state={
                "near_limit": self._near_limit(),
                "holding_pen": self._holding,
                "joint_norm": float(np.linalg.norm(self._q)),
            },
            mission_phase=self._phase,
        )
        decision = Decision(
            intent=out.intent,
            risk=out.risk,
            allow_motion=out.allow_motion and not (self._near_limit() and out.risk > 0.6),
            reason=out.reason,
            tags=list(out.tags) + [f"backend:{out.backend}"],
        )
        payload = decision.model_dump()
        payload["latency_ms"] = out.latency_ms
        payload["backend"] = out.backend
        payload["phase"] = self._phase
        self.pub.publish(Topic.DECISION, payload)
        self.gui_pub.publish(Topic.DECISION, payload)
        self.gui_pub.publish(
            Topic.METRICS,
            {
                "node": self.name,
                "latency_ms": out.latency_ms,
                "hz": self._rate.hz,
                "queue_depth": 0,
                "extras": {"backend": out.backend, "intent": out.intent, "phase": self._phase},
            },
        )
        self._rate.tick()

    def teardown(self) -> None:
        self.state_sub.close()
        self.token_sub.close()
        self.perc_sub.close()
        self.mission_sub.close()
        self.pub.close()
        self.gui_pub.close()

    def run(self, hz: float | None = None) -> None:
        if hz is None:
            hz = float(self.cfg.get("reasoning", {}).get("rate_hz", 5.0))
        super().run(hz=hz)
