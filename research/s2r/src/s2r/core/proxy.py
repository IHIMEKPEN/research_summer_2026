"""ZMQ XPUB/XSUB proxy for fan-in/fan-out scalability.

Use this when many nodes must publish onto one logical bus (e.g. GUI telemetry)
without fighting over a single PUB bind.
"""

from __future__ import annotations

import zmq


def run_forwarder(frontend_bind: str, backend_bind: str) -> None:
    """Blocking proxy: publishers connect to frontend; subscribers to backend.

    frontend: XSUB bind  (e.g. tcp://127.0.0.1:5561)
    backend:  XPUB bind  (e.g. tcp://127.0.0.1:5562)
    """
    ctx = zmq.Context.instance()
    xsub = ctx.socket(zmq.XSUB)
    xpub = ctx.socket(zmq.XPUB)
    xsub.bind(frontend_bind)
    xpub.bind(backend_bind)
    zmq.proxy(xsub, xpub)
