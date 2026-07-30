"""Base Node abstraction for scalable pipeline workers."""

from __future__ import annotations

import signal
import time
from abc import ABC, abstractmethod
from typing import Any

from rich.console import Console

from s2r.core.config import load_config


console = Console()


class Node(ABC):
    """Lifecycle: setup -> run loop -> teardown.

    Each node is an independent process-friendly unit communicating over ZMQ.
    """

    name: str = "node"

    def __init__(self, config_path: str | None = None, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg if cfg is not None else load_config(config_path)
        self._running = False
        self._seq = 0
        self._t0 = time.perf_counter()
        self._ticks = 0

    @property
    def zmq_cfg(self) -> dict[str, Any]:
        return self.cfg.get("zmq", {})

    def setup(self) -> None:
        """Allocate sockets / models."""

    @abstractmethod
    def step(self) -> None:
        """One iteration of the node loop."""

    def teardown(self) -> None:
        """Release resources."""

    def rate_sleep(self, hz: float, started: float) -> None:
        if hz <= 0:
            return
        target = 1.0 / hz
        elapsed = time.perf_counter() - started
        delay = target - elapsed
        if delay > 0:
            time.sleep(delay)

    def measured_hz(self) -> float:
        dt = time.perf_counter() - self._t0
        if dt <= 0:
            return 0.0
        return self._ticks / dt

    def run(self, hz: float | None = None) -> None:
        self._install_signals()
        self.setup()
        self._running = True
        console.print(f"[bold green]▶[/] starting node [cyan]{self.name}[/]")
        try:
            while self._running:
                t = time.perf_counter()
                self.step()
                self._ticks += 1
                if hz:
                    self.rate_sleep(hz, t)
        except KeyboardInterrupt:
            console.print(f"[yellow]■[/] interrupt node [cyan]{self.name}[/]")
        finally:
            self.teardown()
            console.print(f"[bold red]■[/] stopped node [cyan]{self.name}[/]")

    def stop(self) -> None:
        self._running = False

    def _install_signals(self) -> None:
        def _handler(signum, frame):  # noqa: ARG001
            self.stop()

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
