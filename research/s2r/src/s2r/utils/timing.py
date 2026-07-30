"""Lightweight latency / rate helpers."""

from __future__ import annotations

import time
from collections import deque


class RateMeter:
    def __init__(self, window: int = 100) -> None:
        self._ts: deque[float] = deque(maxlen=window)

    def tick(self) -> None:
        self._ts.append(time.perf_counter())

    @property
    def hz(self) -> float:
        if len(self._ts) < 2:
            return 0.0
        dt = self._ts[-1] - self._ts[0]
        if dt <= 0:
            return 0.0
        return (len(self._ts) - 1) / dt


class LatencyTracker:
    def __init__(self, window: int = 200) -> None:
        self._samples: deque[float] = deque(maxlen=window)

    def add(self, latency_ms: float) -> None:
        self._samples.append(latency_ms)

    @property
    def mean_ms(self) -> float:
        if not self._samples:
            return 0.0
        return sum(self._samples) / len(self._samples)

    @property
    def p95_ms(self) -> float:
        if not self._samples:
            return 0.0
        arr = sorted(self._samples)
        idx = int(0.95 * (len(arr) - 1))
        return arr[idx]
