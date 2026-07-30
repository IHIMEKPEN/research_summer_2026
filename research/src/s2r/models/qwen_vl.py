"""Qwen2.5-VL open-source vision-language backend for scene understanding.

Can run:
  - local transformers (`Qwen/Qwen2.5-VL-3B-Instruct`)
  - OpenAI-compatible HTTP (vLLM on Jetson Thor / V100 host)
  - mock captioner when weights are unavailable
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any

import numpy as np

from s2r.models.base import Detection, PerceptionFrame, VisionBackend


class QwenVLBackend(VisionBackend):
    name = "qwen_vl"

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: str = "cuda",
        mock: bool = False,
        api_base: str | None = None,
        api_key: str = "EMPTY",
        max_new_tokens: int = 128,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.mock = mock
        self.api_base = api_base
        self.api_key = api_key
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None
        if not mock and api_base is None:
            self._try_load()
        self.name = f"qwen_vl:{model_id.split('/')[-1]}" + (":mock" if self.mock else "")

    def _try_load(self) -> None:
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

            self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            dtype = torch.float16 if str(self.device).startswith("cuda") else torch.float32
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                device_map="auto" if str(self.device).startswith("cuda") else None,
                trust_remote_code=True,
            )
            self.mock = False
        except Exception:
            self.mock = True

    def infer(self, image: Any, prompt: str = "") -> PerceptionFrame:
        t0 = time.perf_counter()
        q = prompt or (
            "Describe the lab scene for a humanoid robot fetching a pen. "
            "List objects (pen, table, person) and say if a pen is graspable."
        )
        if self.mock:
            frame = self._mock(image, q)
        elif self.api_base:
            frame = self._api(image, q)
        else:
            frame = self._local(image, q)
        frame.latency_ms = (time.perf_counter() - t0) * 1000.0
        frame.backend = self.name
        return frame

    def _local(self, image: Any, prompt: str) -> PerceptionFrame:
        assert self._model is not None and self._processor is not None
        # Prefer PIL if available
        try:
            from PIL import Image

            if isinstance(image, np.ndarray):
                image = Image.fromarray(image[..., ::-1] if image.shape[-1] == 3 else image)
        except Exception:
            pass
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt + " Reply with JSON keys: caption, objects, graspable_pen."},
                ],
            }
        ]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], images=[image], return_tensors="pt", padding=True)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        ids = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        out = self._processor.batch_decode(
            [ids[0][inputs["input_ids"].shape[-1] :]], skip_special_tokens=True
        )[0]
        return self._from_text(out)

    def _api(self, image: Any, prompt: str) -> PerceptionFrame:
        import urllib.request

        b64 = self._b64_image(image)
        body = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt + " Reply JSON: caption, objects, graspable_pen."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                }
            ],
            "max_tokens": self.max_new_tokens,
        }
        req = urllib.request.Request(
            self.api_base.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        text = payload["choices"][0]["message"]["content"]
        return self._from_text(text)

    def _from_text(self, text: str) -> PerceptionFrame:
        objects: list[str] = []
        caption = text.strip()[:240]
        graspable = False
        try:
            import re

            m = re.search(r"\{.*\}", text, flags=re.DOTALL)
            data = json.loads(m.group(0) if m else text)
            caption = str(data.get("caption", caption))
            objects = [str(x).lower() for x in data.get("objects", [])]
            graspable = bool(data.get("graspable_pen", False))
        except Exception:
            low = text.lower()
            for lab in ("pen", "table", "person", "chair", "bottle"):
                if lab in low:
                    objects.append(lab)
            graspable = "grasp" in low and "pen" in low
        dets = [Detection(o, 0.7, [0.3, 0.3, 0.7, 0.7]) for o in objects]
        if graspable and "pen" not in objects:
            objects.append("pen")
        return PerceptionFrame(
            detections=dets,
            caption=caption,
            objects_of_interest=objects,
            scene_tags=["qwen_vl"] + (["graspable"] if graspable else []),
        )

    def _mock(self, image: Any, prompt: str) -> PerceptionFrame:
        # Reuse YOLO mock phasing for consistency
        from s2r.models.yolo_detector import YOLODetector

        return YOLODetector(mock=True).infer(image, prompt)

    @staticmethod
    def _b64_image(image: Any) -> str:
        try:
            import cv2

            if isinstance(image, np.ndarray):
                ok, buf = cv2.imencode(".jpg", image)
                if ok:
                    return base64.b64encode(buf.tobytes()).decode("ascii")
        except Exception:
            pass
        # tiny placeholder jpeg header-ish bytes
        return base64.b64encode(np.zeros((8, 8, 3), dtype=np.uint8).tobytes()).decode("ascii")
