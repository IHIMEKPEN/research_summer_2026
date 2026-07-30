"""Qwen open-source LLM reasoner for mission planning / decisions.

Preferred stack:
  - Qwen2.5-3B/7B-Instruct (local transformers or OpenAI-compatible HTTP)
  - Optional vLLM / Ollama / llama.cpp server via `api_base`

Mock mode provides a deterministic finite-state planner for the pen-fetch demo.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from s2r.models.base import PerceptionFrame, PlanStep, ReasonerBackend, ReasonerOutput


SYSTEM_PROMPT = """You are a robotics mission reasoner for a Unitree G1 humanoid.
Given a human instruction, perception summary, and robot state, output STRICT JSON:
{
  "intent": "explore|approach_table|locate_pen|grasp_pen|return_to_user|handoff|hold|idle",
  "reason": "short explanation",
  "risk": 0.0-1.0,
  "allow_motion": true/false,
  "tags": ["..."],
  "steps": [{"intent":"...","instruction":"...","risk":0.1,"allow_motion":true}]
}
Prefer safety. If pen is held and person visible -> handoff. If pen visible but far -> approach.
If no pen and exploring -> explore. Never invent sensors you do not have.
"""


class QwenReasoner(ReasonerBackend):
    name = "qwen_reasoner"

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = "cuda",
        mock: bool = False,
        api_base: str | None = None,
        api_key: str = "EMPTY",
        max_new_tokens: int = 256,
        temperature: float = 0.2,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.mock = mock
        self.api_base = api_base
        self.api_key = api_key
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._model = None
        self._tokenizer = None
        self._phase = "explore"
        if not mock and api_base is None:
            self._try_load_local()
        self.name = f"qwen:{model_id.split('/')[-1]}" + (":mock" if self.mock else "")

    def _try_load_local(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            dtype = torch.float16 if str(self.device).startswith("cuda") else torch.float32
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                device_map="auto" if str(self.device).startswith("cuda") else None,
                trust_remote_code=True,
            )
            if not str(self.device).startswith("cuda"):
                self._model.to(self.device)
            self.mock = False
        except Exception:
            self.mock = True

    def plan(
        self,
        instruction: str,
        perception: PerceptionFrame | None,
        state: dict[str, Any] | None = None,
        mission_phase: str = "",
    ) -> ReasonerOutput:
        t0 = time.perf_counter()
        if self.mock:
            out = self._mock_plan(instruction, perception, state, mission_phase)
        elif self.api_base:
            out = self._api_plan(instruction, perception, state, mission_phase)
        else:
            out = self._local_plan(instruction, perception, state, mission_phase)
        out.latency_ms = (time.perf_counter() - t0) * 1000.0
        out.backend = self.name
        return out

    def _context(
        self,
        instruction: str,
        perception: PerceptionFrame | None,
        state: dict[str, Any] | None,
        mission_phase: str,
    ) -> str:
        perc = {
            "caption": getattr(perception, "caption", ""),
            "objects": getattr(perception, "objects_of_interest", []),
            "detections": [
                {"label": d.label, "confidence": d.confidence} for d in getattr(perception, "detections", [])
            ],
        }
        return json.dumps(
            {
                "instruction": instruction,
                "mission_phase": mission_phase or self._phase,
                "perception": perc,
                "state": state or {},
            },
            ensure_ascii=True,
        )

    def _parse(self, text: str, fallback: ReasonerOutput) -> ReasonerOutput:
        try:
            m = re.search(r"\{.*\}", text, flags=re.DOTALL)
            data = json.loads(m.group(0) if m else text)
            steps = [
                PlanStep(
                    intent=s.get("intent", data.get("intent", "hold")),
                    instruction=s.get("instruction", ""),
                    risk=float(s.get("risk", data.get("risk", 0.2))),
                    allow_motion=bool(s.get("allow_motion", True)),
                    tags=list(s.get("tags", [])),
                )
                for s in data.get("steps", [])
            ]
            return ReasonerOutput(
                intent=str(data.get("intent", fallback.intent)),
                reason=str(data.get("reason", "")),
                risk=float(data.get("risk", 0.2)),
                allow_motion=bool(data.get("allow_motion", True)),
                steps=steps,
                tags=list(data.get("tags", [])),
                raw_text=text,
            )
        except Exception:
            fallback.raw_text = text
            return fallback

    def _local_plan(self, instruction, perception, state, mission_phase) -> ReasonerOutput:
        assert self._model is not None and self._tokenizer is not None
        user = self._context(instruction, perception, state, mission_phase)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        out_ids = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=max(self.temperature, 1e-5),
        )
        text = self._tokenizer.decode(out_ids[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        fb = self._mock_plan(instruction, perception, state, mission_phase)
        return self._parse(text, fb)

    def _api_plan(self, instruction, perception, state, mission_phase) -> ReasonerOutput:
        import urllib.request

        user = self._context(instruction, perception, state, mission_phase)
        body = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
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
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        text = payload["choices"][0]["message"]["content"]
        fb = self._mock_plan(instruction, perception, state, mission_phase)
        return self._parse(text, fb)

    def _mock_plan(self, instruction, perception, state, mission_phase) -> ReasonerOutput:
        objs = set(getattr(perception, "objects_of_interest", []) or [])
        caption = getattr(perception, "caption", "") or ""
        holding = bool((state or {}).get("holding_pen", False))
        phase = mission_phase or self._phase

        if holding and "person" in objs:
            intent, reason, phase = "handoff", "requester visible; deliver pen", "handoff"
        elif holding:
            intent, reason, phase = "return_to_user", "pen acquired; return to requester", "return"
        elif "pen" in objs and "grasp" in caption:
            intent, reason, phase = "grasp_pen", "pen in grasp zone", "grasp"
        elif "pen" in objs:
            intent, reason, phase = "approach_table", "pen seen; approach", "approach"
        elif "table" in {d.label for d in getattr(perception, "detections", [])}:
            intent, reason, phase = "locate_pen", "table found; search for pen", "locate"
        else:
            intent, reason, phase = "explore", "search lab for table/pen", "explore"

        # Safety: high joint risk from caller can gate motion
        risk = 0.15
        if (state or {}).get("near_limit"):
            risk = 0.7
        allow = risk < 0.85
        self._phase = phase
        return ReasonerOutput(
            intent=intent,
            reason=reason,
            risk=risk,
            allow_motion=allow,
            steps=[
                PlanStep(intent=intent, instruction=instruction or "bring pen", risk=risk, allow_motion=allow)
            ],
            tags=["qwen_mock", phase, "bring_pen"],
            raw_text="",
        )
