"""Low-latency ZMQ helpers tuned for robotics control loops.

Design notes for ultra-low latency:
- Single-frame messages (CONFLATE is incompatible with multipart)
- CONFLATE=1 on state topics so subscribers always get the newest sample
- LINGER=0 to avoid shutdown stalls
- Bounded HWM to prevent unbounded memory growth under backpressure
- msgpack payloads (compact binary)
- PUB/SUB for fan-out; PUSH/PULL for collection sinks
"""

from __future__ import annotations

import time
from typing import Any, Iterable

import zmq

from s2r.core.serialize import pack, unpack
from s2r.core.schemas import Envelope, Topic


def _apply_low_latency(sock: zmq.Socket, cfg: dict[str, Any], conflate: bool = False) -> None:
    sock.setsockopt(zmq.LINGER, int(cfg.get("linger_ms", 0)))
    sock.setsockopt(zmq.SNDHWM, int(cfg.get("sndhwm", 1000)))
    sock.setsockopt(zmq.RCVHWM, int(cfg.get("rcvhwm", 1000)))
    if conflate and cfg.get("conflate", True):
        sock.setsockopt(zmq.CONFLATE, 1)


class Publisher:
    def __init__(
        self,
        endpoint: str,
        zmq_cfg: dict[str, Any] | None = None,
        bind: bool = True,
        conflate: bool = False,
        source: str = "",
    ) -> None:
        self.endpoint = endpoint
        self.source = source
        self._seq = 0
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PUB)
        _apply_low_latency(self.sock, zmq_cfg or {}, conflate=conflate)
        if bind:
            self.sock.bind(endpoint)
        else:
            self.sock.connect(endpoint)
        # PUB/SUB slow-joiner mitigation
        time.sleep(0.05)

    def publish(self, topic: Topic | str, payload: dict[str, Any], ts: float | None = None) -> Envelope:
        self._seq += 1
        env = Envelope(
            topic=Topic(topic) if not isinstance(topic, Topic) else topic,
            ts=time.time() if ts is None else ts,
            seq=self._seq,
            source=self.source,
            payload=payload,
        )
        try:
            self.sock.send(pack(env.to_dict()), flags=zmq.NOBLOCK)
        except zmq.Again:
            # Drop under backpressure — prefer freshness over backlog
            pass
        return env

    def close(self) -> None:
        self.sock.close(0)


class Subscriber:
    def __init__(
        self,
        endpoint: str,
        topics: Iterable[str | Topic] = (),
        zmq_cfg: dict[str, Any] | None = None,
        conflate: bool = False,
        bind: bool = False,
    ) -> None:
        self.endpoint = endpoint
        self._topic_filter = {
            (t.value if isinstance(t, Topic) else str(t)) for t in topics
        }
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)
        _apply_low_latency(self.sock, zmq_cfg or {}, conflate=conflate)
        if bind:
            self.sock.bind(endpoint)
        else:
            self.sock.connect(endpoint)
        # Subscribe to all frames; filter by envelope.topic in Python.
        # (Required because CONFLATE + multipart topic frames are unreliable.)
        self.sock.setsockopt_string(zmq.SUBSCRIBE, "")

    def _accept(self, env: Envelope) -> bool:
        if not self._topic_filter:
            return True
        return env.topic.value in self._topic_filter

    def recv(self, timeout_ms: int | None = None) -> Envelope | None:
        if timeout_ms is not None:
            if not self.sock.poll(timeout_ms, zmq.POLLIN):
                return None
        try:
            body = self.sock.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            return None
        env = Envelope.from_dict(unpack(body))
        if not self._accept(env):
            return None
        return env

    def recv_blocking(self) -> Envelope:
        while True:
            body = self.sock.recv()
            env = Envelope.from_dict(unpack(body))
            if self._accept(env):
                return env

    def close(self) -> None:
        self.sock.close(0)


class PushSocket:
    def __init__(self, endpoint: str, zmq_cfg: dict[str, Any] | None = None, bind: bool = False) -> None:
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PUSH)
        _apply_low_latency(self.sock, zmq_cfg or {}, conflate=False)
        if bind:
            self.sock.bind(endpoint)
        else:
            self.sock.connect(endpoint)

    def send(self, payload: dict[str, Any]) -> None:
        try:
            self.sock.send(pack(payload), flags=zmq.NOBLOCK)
        except zmq.Again:
            pass

    def close(self) -> None:
        self.sock.close(0)


class PullSocket:
    def __init__(self, endpoint: str, zmq_cfg: dict[str, Any] | None = None, bind: bool = True) -> None:
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PULL)
        _apply_low_latency(self.sock, zmq_cfg or {}, conflate=False)
        if bind:
            self.sock.bind(endpoint)
        else:
            self.sock.connect(endpoint)

    def recv(self, timeout_ms: int | None = None) -> dict[str, Any] | None:
        if timeout_ms is not None and not self.sock.poll(timeout_ms, zmq.POLLIN):
            return None
        try:
            return unpack(self.sock.recv(flags=zmq.NOBLOCK))
        except zmq.Again:
            return None

    def close(self) -> None:
        self.sock.close(0)
