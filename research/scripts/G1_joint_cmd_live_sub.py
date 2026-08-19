#!/usr/bin/env python3
"""Live-robot ZMQ subscriber for oracle / joint_cmd streams.

Run this ON the G1 (or any machine that should receive cmds).
Pairs with G1_oracle_demo_stage0.py --live (PUB on port 5557).

Camera stays on :5555 (G1_camera_v2). Control uses :5557 so they do not collide.

  # Terminal A — Mac or G1 (publisher):
  python G1_oracle_demo_stage0.py --live --synthetic --seconds 20

  # Terminal B — G1 (this script):
  # if publisher is on the same G1:
  python G1_joint_cmd_live_sub.py --host 127.0.0.1 --port 5557
  # if publisher is on your Mac, use the Mac LAN IP:
  python G1_joint_cmd_live_sub.py --host 192.168.x.x --port 5557

Default: LOG ONLY (no Unitree motion). Add --enable-motion later only with e-stop ready.
"""

from __future__ import annotations

import argparse
import json
import time

import zmq


def main() -> int:
    p = argparse.ArgumentParser(description="SUB joint_cmd from oracle demo PUB (live ZMQ)")
    p.add_argument("--host", default="127.0.0.1", help="Host where oracle PUB is bound")
    p.add_argument("--port", type=int, default=5557, help="joint_cmd port (default 5557)")
    p.add_argument("--topic", default="joint_cmd", help="ZMQ topic filter")
    p.add_argument("--timeout", type=int, default=3000)
    p.add_argument(
        "--enable-motion",
        action="store_true",
        help="RESERVED: would apply cmds via Unitree SDK (OFF — not wired for wipe yet)",
    )
    args = p.parse_args()

    if args.enable_motion:
        print("[ERROR] --enable-motion is not implemented for 29-DoF wipe yet.")
        print("        Use log-only mode to verify ZMQ. Exiting.")
        return 2

    addr = f"tcp://{args.host}:{args.port}"
    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.RCVHWM, 2)
    sub.setsockopt(zmq.RCVTIMEO, args.timeout)
    sub.setsockopt_string(zmq.SUBSCRIBE, args.topic)
    sub.connect(addr)

    print(f"[INFO] SUB {addr} topic={args.topic!r}")
    print("[INFO] LOG ONLY — no hardware motion")
    print("[INFO] Ctrl+C to stop")

    n = 0
    t0 = time.perf_counter()
    last_log = t0
    try:
        while True:
            try:
                parts = sub.recv_multipart()
            except zmq.Again:
                print(f"[WARN] no joint_cmd in {args.timeout}ms — is oracle PUB running?")
                continue

            # Accept [topic, json] or single json frame
            raw = parts[-1]
            try:
                msg = json.loads(raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw)
            except Exception:
                print(f"[WARN] bad payload parts={len(parts)} bytes={len(raw)}")
                continue

            payload = msg.get("payload", msg)
            q = payload.get("q") or payload.get("action")
            n += 1
            now = time.perf_counter()
            if now - last_log >= 1.0:
                hz = n / max(1e-6, now - t0)
                qn = 0.0
                if isinstance(q, list) and q:
                    qn = sum(float(x) * float(x) for x in q) ** 0.5
                print(
                    f"[INFO] Receiving joint_cmd at {hz:.1f} Hz | frames={n} "
                    f"| source={payload.get('source', '?')} ||q||={qn:.3f}"
                )
                last_log = now
    except KeyboardInterrupt:
        print("\n[INFO] stopped")
    finally:
        elapsed = max(1e-6, time.perf_counter() - t0)
        print(f"[INFO] done frames={n} mean_hz={n / elapsed:.1f}")
        sub.close(0)
    return 0 if n > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
