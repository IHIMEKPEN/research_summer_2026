"""Model adapter interfaces with mock fallbacks for offline bring-up."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Detection:
    label: str
    confidence: float
    xyxy: list[float]  # x1,y1,x2,y2 normalized 0..1
    track_id: int | None = None


@dataclass
class PerceptionFrame:
    detections: list[Detection] = field(default_factory=list)
    caption: str = ""
    objects_of_interest: list[str] = field(default_factory=list)
    scene_tags: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    backend: str = "mock"


@dataclass
class PlanStep:
    intent: str
    instruction: str
    risk: float = 0.1
    allow_motion: bool = True
    tags: list[str] = field(default_factory=list)


@dataclass
class ReasonerOutput:
    intent: str
    reason: str
    risk: float
    allow_motion: bool
    steps: list[PlanStep] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    raw_text: str = ""
    latency_ms: float = 0.0
    backend: str = "mock"


class VisionBackend(ABC):
    name: str = "vision"

    @abstractmethod
    def infer(self, image: Any, prompt: str = "") -> PerceptionFrame:
        raise NotImplementedError


class ReasonerBackend(ABC):
    name: str = "reasoner"

    @abstractmethod
    def plan(
        self,
        instruction: str,
        perception: PerceptionFrame | None,
        state: dict[str, Any] | None = None,
        mission_phase: str = "",
    ) -> ReasonerOutput:
        raise NotImplementedError
