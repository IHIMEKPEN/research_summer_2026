"""Data collection sink for real-to-sim training corpora.

Aggregates state / action tokens / joint commands / decisions into episode
shards (JSONL) suitable for ESN training and VLA fine-tune export.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from s2r.core.node import Node
from s2r.core.schemas import Topic
from s2r.core.zmq_bus import Publisher, Subscriber


class DataCollectorNode(Node):
    name = "data_collector"

    def setup(self) -> None:
        z = self.zmq_cfg
        dcfg = self.cfg.get("data_collection", {})
        self.enabled = bool(dcfg.get("enabled", True))
        self.out_dir = Path(dcfg.get("out_dir", "data/raw"))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.flush_every = int(dcfg.get("flush_every", 50))
        self._buffer: list[dict[str, Any]] = []
        self._episode = time.strftime("%Y%m%d_%H%M%S")
        self._path = self.out_dir / f"episode_{self._episode}.jsonl"
        self._fh = self._path.open("a", encoding="utf-8")

        # Subscribe to primary topics via dedicated endpoints + gui bus
        self.subs = [
            Subscriber(z["state_pub"], topics=[Topic.STATE], zmq_cfg=z, conflate=False),
            Subscriber(z["action_token_pub"], topics=[Topic.ACTION_TOKEN], zmq_cfg=z, conflate=False),
            Subscriber(z["joint_cmd_pub"], topics=[Topic.JOINT_CMD], zmq_cfg=z, conflate=False),
            Subscriber(z["decision_pub"], topics=[Topic.DECISION], zmq_cfg=z, conflate=False),
            Subscriber(z["perception_pub"], topics=[Topic.PERCEPTION], zmq_cfg=z, conflate=False),
            Subscriber(z["mission_pub"], topics=[Topic.MISSION], zmq_cfg=z, conflate=False),
        ]
        self.gui_pub = Publisher(z["gui_pub"], zmq_cfg=z, bind=False, source=self.name)
        self._count = 0

    def _write(self, row: dict[str, Any]) -> None:
        self._buffer.append(row)
        if len(self._buffer) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        for row in self._buffer:
            self._fh.write(json.dumps(row, separators=(",", ":")) + "\n")
        self._fh.flush()
        self._buffer.clear()

    def step(self) -> None:
        if not self.enabled:
            time.sleep(0.05)
            return
        got = False
        for sub in self.subs:
            while True:
                env = sub.recv(timeout_ms=0)
                if env is None:
                    break
                got = True
                self._count += 1
                self._write(
                    {
                        "ts": env.ts,
                        "topic": env.topic.value,
                        "source": env.source,
                        "seq": env.seq,
                        "payload": env.payload,
                    }
                )
        if not got:
            time.sleep(0.001)
        if self._ticks % 100 == 0:
            self.gui_pub.publish(
                Topic.METRICS,
                {
                    "node": self.name,
                    "latency_ms": 0.0,
                    "hz": 0.0,
                    "queue_depth": len(self._buffer),
                    "extras": {"rows": self._count, "file": str(self._path)},
                },
            )

    def teardown(self) -> None:
        self.flush()
        self._fh.close()
        for s in self.subs:
            s.close()
        self.gui_pub.close()
