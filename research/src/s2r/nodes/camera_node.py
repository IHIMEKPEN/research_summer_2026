"""Camera publisher node (OpenCV / mock frames)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from s2r.core.node import Node
from s2r.core.schemas import Topic
from s2r.core.zmq_bus import Publisher
from s2r.models.yolo_detector import encode_image_stub


class CameraNode(Node):
    name = "camera"

    def setup(self) -> None:
        z = self.zmq_cfg
        c = self.cfg.get("camera", {})
        self.hz = float(c.get("rate_hz", 15.0))
        self.source = str(c.get("source", "mock"))
        self.width = int(c.get("width", 640))
        self.height = int(c.get("height", 480))
        self._cap = None
        self._frame_id = 0
        if self.source != "mock":
            self._open_capture()
        self.pub = Publisher(z["camera_pub"], zmq_cfg=z, source=self.name, conflate=True)
        self.gui_pub = Publisher(z["gui_pub"], zmq_cfg=z, bind=False, source=self.name)

    def _open_capture(self) -> None:
        try:
            import cv2

            src: Any = int(self.source) if str(self.source).isdigit() else self.source
            self._cap = cv2.VideoCapture(src)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if not self._cap.isOpened():
                self.source = "mock"
                self._cap = None
        except Exception:
            self.source = "mock"
            self._cap = None

    def _read(self) -> np.ndarray:
        if self._cap is not None:
            ok, frame = self._cap.read()
            if ok:
                return frame
        return encode_image_stub(self.height, self.width)

    def step(self) -> None:
        frame = self._read()
        self._frame_id += 1
        self.cfg.setdefault("_runtime", {})["latest_frame"] = frame
        payload = {
            "frame_id": self._frame_id,
            "source": self.source,
            "shape": list(frame.shape),
            "ts": time.time(),
            "fingerprint": [float(x) for x in frame.reshape(-1, 3).mean(axis=0)],
        }
        self.pub.publish(Topic.CAMERA, payload)
        if self._ticks % int(max(1, self.hz)) == 0:
            self.gui_pub.publish(
                Topic.METRICS,
                {
                    "node": self.name,
                    "latency_ms": 0.0,
                    "hz": self.hz,
                    "queue_depth": 0,
                    "extras": payload,
                },
            )

    def teardown(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self.pub.close()
        self.gui_pub.close()

    def run(self, hz: float | None = None) -> None:
        if hz is None:
            hz = float(self.cfg.get("camera", {}).get("rate_hz", 15.0))
        super().run(hz=hz)
