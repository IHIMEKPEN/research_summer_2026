"""Open-source object detection via Ultralytics YOLO (YOLOv8/YOLO11).

Falls back to a deterministic mock detector when ultralytics/torch are absent
or `backend: mock` is configured.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from s2r.models.base import Detection, PerceptionFrame, VisionBackend


# Labels useful for lab fetch tasks
DEFAULT_LABELS = ["pen", "pencil", "bottle", "cup", "book", "person", "chair", "table", "laptop"]


class YOLODetector(VisionBackend):
    name = "yolo"

    def __init__(
        self,
        model_id: str = "yolov8n.pt",
        device: str = "cuda",
        conf: float = 0.35,
        mock: bool = False,
        target_labels: list[str] | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.conf = conf
        self.mock = mock
        self.target_labels = [x.lower() for x in (target_labels or DEFAULT_LABELS)]
        self._model = None
        if not mock:
            self._try_load()

    def _try_load(self) -> None:
        try:
            from ultralytics import YOLO  # type: ignore

            self._model = YOLO(self.model_id)
            self.mock = False
            self.name = f"yolo:{self.model_id}"
        except Exception:
            self.mock = True
            self.name = "yolo:mock"

    def infer(self, image: Any, prompt: str = "") -> PerceptionFrame:
        t0 = time.perf_counter()
        if self.mock or self._model is None:
            frame = self._mock_infer(image, prompt)
        else:
            frame = self._real_infer(image)
        frame.latency_ms = (time.perf_counter() - t0) * 1000.0
        frame.backend = self.name
        return frame

    def _real_infer(self, image: Any) -> PerceptionFrame:
        results = self._model.predict(image, conf=self.conf, device=self.device, verbose=False)
        dets: list[Detection] = []
        h = w = 1
        if hasattr(image, "shape"):
            h, w = int(image.shape[0]), int(image.shape[1])
        for r in results:
            names = r.names
            if r.boxes is None:
                continue
            for b in r.boxes:
                cls_id = int(b.cls.item())
                label = str(names.get(cls_id, cls_id)).lower()
                conf = float(b.conf.item())
                x1, y1, x2, y2 = [float(x) for x in b.xyxy[0].tolist()]
                dets.append(
                    Detection(
                        label=label,
                        confidence=conf,
                        xyxy=[x1 / w, y1 / h, x2 / w, y2 / h],
                    )
                )
        interesting = [d.label for d in dets if d.label in self.target_labels]
        return PerceptionFrame(
            detections=dets,
            caption=f"detected {len(dets)} objects",
            objects_of_interest=interesting,
            scene_tags=["yolo"],
        )

    def _mock_infer(self, image: Any, prompt: str) -> PerceptionFrame:
        # Synthetic lab scene cycling for pipeline demos
        phase = int(time.time() / 3.0) % 5
        dets: list[Detection] = []
        objects: list[str] = []
        if phase <= 1:
            dets = [Detection("table", 0.7, [0.2, 0.4, 0.8, 0.9])]
            caption = "lab table visible, no pen yet"
        elif phase == 2:
            dets = [
                Detection("table", 0.8, [0.2, 0.4, 0.8, 0.9]),
                Detection("pen", 0.86, [0.45, 0.55, 0.55, 0.72]),
            ]
            objects = ["pen"]
            caption = "pen detected on the table"
        elif phase == 3:
            dets = [Detection("pen", 0.9, [0.48, 0.48, 0.52, 0.6])]
            objects = ["pen"]
            caption = "pen close — grasp zone"
        else:
            dets = [Detection("person", 0.88, [0.35, 0.2, 0.65, 0.95])]
            objects = ["person"]
            caption = "person (requester) in view for handoff"
        if prompt and "pen" in prompt.lower() and "pen" not in objects and phase >= 2:
            objects.append("pen")
        return PerceptionFrame(
            detections=dets,
            caption=caption,
            objects_of_interest=objects,
            scene_tags=["mock_lab", f"phase_{phase}"],
        )


def encode_image_stub(h: int = 480, w: int = 640) -> np.ndarray:
    """Create a placeholder RGB frame when no camera is attached."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = (30, 40, 55)
    return img
