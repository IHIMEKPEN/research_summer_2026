"""Perception node: YOLO detector + optional Qwen2.5-VL scene understanding."""

from __future__ import annotations

import time
from typing import Any

from s2r.core.node import Node
from s2r.core.schemas import Topic
from s2r.core.zmq_bus import Publisher
from s2r.models.registry import build_detector, build_vlm
from s2r.models.yolo_detector import encode_image_stub
from s2r.utils.timing import RateMeter


class VisionNode(Node):
    name = "vision"

    def setup(self) -> None:
        z = self.zmq_cfg
        vcfg = self.cfg.get("vision", {})
        self.hz = float(vcfg.get("rate_hz", 5.0))
        self.use_vlm = bool(vcfg.get("use_vlm", True))
        self.vlm_every = int(vcfg.get("vlm_every_n", max(1, int(self.hz))))
        self.instruction = str(self.cfg.get("mission", {}).get("instruction", "bring me a pen"))
        self.detector = build_detector(self.cfg)
        self.vlm = build_vlm(self.cfg) if self.use_vlm else None
        self.pub = Publisher(z["perception_pub"], zmq_cfg=z, source=self.name, conflate=True)
        self.gui_pub = Publisher(z["gui_pub"], zmq_cfg=z, bind=False, source=self.name)
        self._rate = RateMeter()
        self._holding = False

    def _frame(self) -> Any:
        runtime = self.cfg.get("_runtime", {})
        frame = runtime.get("latest_frame")
        if frame is None:
            return encode_image_stub()
        return frame

    def step(self) -> None:
        img = self._frame()
        det = self.detector.infer(img, prompt=self.instruction)
        caption = det.caption
        objects = list(det.objects_of_interest)
        tags = list(det.scene_tags)
        vlm_latency = 0.0

        if self.vlm is not None and (self._ticks % max(1, self.vlm_every) == 0):
            vlm = self.vlm.infer(img, prompt=self.instruction)
            caption = vlm.caption or caption
            for o in vlm.objects_of_interest:
                if o not in objects:
                    objects.append(o)
            tags.extend(vlm.scene_tags)
            vlm_latency = vlm.latency_ms

        if "grasp zone" in (caption or "") or "graspable" in tags:
            self._holding = True

        payload = {
            "caption": caption,
            "objects_of_interest": objects,
            "detections": [d.__dict__ for d in det.detections],
            "scene_tags": tags,
            "holding_pen": self._holding,
            "detector_backend": det.backend,
            "latency_ms": det.latency_ms,
            "vlm_latency_ms": vlm_latency,
            "ts": time.time(),
        }
        self.cfg.setdefault("_runtime", {})["latest_perception"] = payload
        self.pub.publish(Topic.PERCEPTION, payload)
        self.gui_pub.publish(Topic.PERCEPTION, payload)
        self.gui_pub.publish(
            Topic.METRICS,
            {
                "node": self.name,
                "latency_ms": det.latency_ms + vlm_latency,
                "hz": self._rate.hz,
                "queue_depth": 0,
                "extras": {
                    "detector": det.backend,
                    "objects": objects,
                    "vlm_latency_ms": vlm_latency,
                },
            },
        )
        self._rate.tick()

    def teardown(self) -> None:
        self.pub.close()
        self.gui_pub.close()

    def run(self, hz: float | None = None) -> None:
        if hz is None:
            hz = float(self.cfg.get("vision", {}).get("rate_hz", 5.0))
        super().run(hz=hz)
