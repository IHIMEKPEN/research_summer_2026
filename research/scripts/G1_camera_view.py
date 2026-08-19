#!/usr/bin/env python3
"""Live G1 head_camera viewer (no YOLO, no motion).

Expects G1_camera_v2.py publishing multipart:
  [topic, jpeg_bytes, depth_u16 640x480] on tcp://*:5555 topic head_camera

  # terminal 1 (G1):
  python G1_camera_v2.py --fps 10

  # terminal 2 (Mac, with scripts/.venv):
  source scripts/.venv/bin/activate
  python scripts/G1_camera_view.py --host 10.54.182.34

  # on G1 itself:
  python scripts/G1_camera_view.py --host 127.0.0.1

Press q to quit, s to save a snapshot.
"""

from __future__ import annotations

import argparse
import datetime as dt
import time

import cv2
import numpy as np
import zmq


def main() -> int:
    p = argparse.ArgumentParser(description="Live head_camera stream (RGB + depth)")
    p.add_argument("--host", default="127.0.0.1", help="Publisher host / G1 IP")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--topic", default="head_camera")
    p.add_argument("--timeout", type=int, default=5000, help="Recv timeout ms")
    p.add_argument("--no-depth", action="store_true", help="Show RGB only")
    args = p.parse_args()

    addr = f"tcp://{args.host}:{args.port}"
    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    # Do NOT set CONFLATE with multipart [topic, jpeg, depth] — it drops
    # intermediate frames and triggers: Assertion failed: !_more (fq.cpp)
    sub.setsockopt(zmq.RCVHWM, 2)
    sub.setsockopt(zmq.RCVTIMEO, args.timeout)
    sub.setsockopt_string(zmq.SUBSCRIBE, args.topic)
    sub.connect(addr)

    print(f"[INFO] SUB {addr} topic={args.topic!r}")
    print("[INFO] Press q to quit, s to snapshot")

    n = 0
    t0 = time.perf_counter()
    last_log = t0
    window = "G1 head_camera"

    try:
        while True:
            try:
                parts = sub.recv_multipart()
            except zmq.Again:
                print(f"[WARN] no frame in {args.timeout}ms — is G1_camera_v2.py running?")
                continue

            if len(parts) < 2:
                print(f"[WARN] malformed message (parts={len(parts)})")
                continue

            frame = cv2.imdecode(np.frombuffer(parts[1], dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                print("[WARN] JPEG decode failed")
                continue

            n += 1
            now = time.perf_counter()
            if now - last_log >= 2.0:
                print(f"[INFO] Receiving at {n / (now - t0):.1f} fps | frames={n} | shape={frame.shape}")
                last_log = now

            if args.no_depth or len(parts) < 3:
                disp = frame
            else:
                depth = np.frombuffer(parts[2], dtype=np.uint16).reshape((480, 640))
                heat = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.1), cv2.COLORMAP_JET)
                disp = np.hstack((frame, heat))

            cv2.imshow(window, disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                path = f"snapshot_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(path, frame)
                print(f"[INFO] saved {path}")
    except KeyboardInterrupt:
        print("\n[INFO] stopped")
    finally:
        elapsed = max(1e-6, time.perf_counter() - t0)
        print(f"[INFO] done frames={n} mean_fps={n / elapsed:.1f}")
        sub.close(0)
        cv2.destroyAllWindows()

    return 0 if n > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
