"""FastAPI monitoring GUI for mapping, states, decisions, and metrics."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from s2r.core.node import Node
from s2r.core.schemas import Topic
from s2r.core.zmq_bus import Subscriber


STATIC_DIR = Path(__file__).parent / "static"


class TelemetryStore:
    def __init__(self, history: int = 300) -> None:
        self.history = history
        self.lock = threading.Lock()
        self.latest: dict[str, Any] = {}
        self.series: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=history))
        self.metrics: dict[str, dict[str, Any]] = {}

    def update(self, topic: str, payload: dict[str, Any], ts: float, source: str) -> None:
        row = {"ts": ts, "source": source, "payload": payload}
        with self.lock:
            self.latest[topic] = row
            self.series[topic].append(row)
            if topic == Topic.METRICS.value:
                node = payload.get("node") or source or "unknown"
                self.metrics[node] = payload

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "latest": {k: v for k, v in self.latest.items()},
                "metrics": dict(self.metrics),
                "series": {k: list(v)[-100:] for k, v in self.series.items()},
                "server_ts": time.time(),
            }


class GUINode(Node):
    """Binds the GUI bus (SUB) and serves a live dashboard."""

    name = "gui"

    def setup(self) -> None:
        z = self.zmq_cfg
        g = self.cfg.get("gui", {})
        self.host = str(g.get("host", "0.0.0.0"))
        self.port = int(g.get("port", 8080))
        self.store = TelemetryStore(history=int(g.get("history", 300)))

        # Fan-in: GUI binds SUB; node publishers connect to gui_pub
        self.sub = Subscriber(
            z["gui_pub"],
            topics=(),
            zmq_cfg=z,
            conflate=False,
            bind=True,
        )
        self.extra_subs = [
            Subscriber(z["state_pub"], topics=[Topic.STATE], zmq_cfg=z, conflate=True),
            Subscriber(z["action_token_pub"], topics=[Topic.ACTION_TOKEN], zmq_cfg=z, conflate=True),
            Subscriber(z["joint_cmd_pub"], topics=[Topic.JOINT_CMD], zmq_cfg=z, conflate=True),
            Subscriber(z["decision_pub"], topics=[Topic.DECISION], zmq_cfg=z, conflate=True),
            Subscriber(z["map_pub"], topics=[Topic.MAP], zmq_cfg=z, conflate=True),
            Subscriber(z["perception_pub"], topics=[Topic.PERCEPTION], zmq_cfg=z, conflate=True),
            Subscriber(z["mission_pub"], topics=[Topic.MISSION], zmq_cfg=z, conflate=True),
        ]

        self.app = FastAPI(title="S2R Monitor")
        if STATIC_DIR.exists():
            self.app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
        self._register_routes()
        self._server_thread = threading.Thread(target=self._run_server, daemon=True)
        self._server_thread.start()

    def _register_routes(self) -> None:
        store = self.store

        @self.app.get("/", response_class=HTMLResponse)
        async def index() -> HTMLResponse:
            html_path = STATIC_DIR / "index.html"
            return HTMLResponse(html_path.read_text(encoding="utf-8"))

        @self.app.get("/api/snapshot")
        async def snapshot() -> dict[str, Any]:
            return store.snapshot()

        @self.app.websocket("/ws")
        async def ws_endpoint(ws: WebSocket) -> None:
            await ws.accept()
            try:
                while True:
                    await ws.send_json(store.snapshot())
                    await asyncio_sleep(0.1)
            except WebSocketDisconnect:
                return
            except Exception:
                return

    def _run_server(self) -> None:
        uvicorn.run(self.app, host=self.host, port=self.port, log_level="warning")

    def _ingest(self, env: Any) -> None:
        self.store.update(env.topic.value, env.payload, env.ts, env.source)

    def step(self) -> None:
        got = False
        for _ in range(64):
            env = self.sub.recv(timeout_ms=0)
            if env is None:
                break
            got = True
            self._ingest(env)
        for sub in self.extra_subs:
            env = sub.recv(timeout_ms=0)
            if env is not None:
                got = True
                self._ingest(env)
        if not got:
            time.sleep(0.001)

    def teardown(self) -> None:
        self.sub.close()
        for s in self.extra_subs:
            s.close()


async def asyncio_sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)


def serve_gui(config_path: str | None = None) -> None:
    GUINode(config_path=config_path).run()
