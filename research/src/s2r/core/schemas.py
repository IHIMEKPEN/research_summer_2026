"""Typed message schemas for the S2R ZMQ bus."""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class Topic(str, Enum):
    STATE = "state"
    ACTION_TOKEN = "action_token"
    JOINT_CMD = "joint_cmd"
    DECISION = "decision"
    MAP = "map"
    METRICS = "metrics"
    DATA = "data"
    GUI = "gui"
    PERCEPTION = "perception"
    MISSION = "mission"
    CAMERA = "camera"


class Envelope(BaseModel):
    """Universal wire envelope. Payload stays compact for low latency."""

    topic: Topic
    ts: float
    seq: int = 0
    source: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="python")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Envelope":
        return cls.model_validate(data)


class RobotState(BaseModel):
    joint_pos: list[float]
    joint_vel: list[float] = Field(default_factory=list)
    ee_pos: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    ee_quat: list[float] = Field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    mode: str = "sim"

    def as_numpy(self) -> np.ndarray:
        return np.asarray(self.joint_pos, dtype=np.float64)


class ActionToken(BaseModel):
    """Sparse VLA action token (~2 Hz)."""

    action: list[float]
    confidence: float = 1.0
    goal: str = ""
    latent: list[float] = Field(default_factory=list)


class JointCommand(BaseModel):
    """High-rate joint command (>=100 Hz)."""

    q: list[float]
    dq: list[float] = Field(default_factory=list)
    source: str = "esn"
    upsample_factor: float = 50.0


class Decision(BaseModel):
    intent: str
    risk: float = 0.0
    allow_motion: bool = True
    reason: str = ""
    tags: list[str] = Field(default_factory=list)


class MapFrame(BaseModel):
    """Lightweight occupancy / landmark snapshot for GUI mapping."""

    frame_id: str = "map"
    robot_xy: list[float] = Field(default_factory=lambda: [0.0, 0.0])
    robot_yaw: float = 0.0
    landmarks: list[list[float]] = Field(default_factory=list)
    grid: list[list[float]] = Field(default_factory=list)  # small downsampled grid
    resolution: float = 0.1


class Metrics(BaseModel):
    node: str
    latency_ms: float = 0.0
    hz: float = 0.0
    queue_depth: int = 0
    extras: dict[str, Any] = Field(default_factory=dict)
