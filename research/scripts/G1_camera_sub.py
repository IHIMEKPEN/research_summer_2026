#!/usr/bin/env python3
"""Subscribe to G1_camera_v2.py ZMQ PUB and display / time the live feed.

Publisher wire format (confirmed via teammate G1_yolo_world_follow_v2.py):
  multipart: [topic:str, jpeg_bytes, depth_u16_bytes 640x480]

Publisher logs:
  Streaming on tcp://*:5555
  Topic: 'head_camera'

On the G1 (publisher already running):
  python G1_camera_v2.py --fps 10
  python scripts/G1_camera_sub.py                  # same box → default 127.0.0.1

  # laptop on robot WiFi:
  python scripts/G1_camera_sub.py --host 10.54.182.34

Press q to quit. Headless: --no-show --seconds 10 --out results/sense_check/head_camera_sample.jpg

Perception only — does not send loco / arm commands.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import zmq


def decode_parts(
    parts: list[bytes],
    depth_hw: Tuple[int, int] = (480, 640),
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (bgr_frame, depth_u16) from multipart [topic, jpeg, depth]."""
    if len(parts) < 2:
        return None, None

    frame = cv2.imdecode(np.frombuffer(parts[1], dtype=np.uint8), cv2.IMREAD_COLOR)
    depth = None
    if len(parts) >= 3:
        h, w = depth_hw
        expected = h * w * 2
        raw = parts[2]
        if len(raw) >= expected:
            depth = np.frombuffer(raw[:expected], dtype=np.uint16).reshape((h, w))
    return frame, depth


def depth_heatmap(depth: np.ndarray) -> np.ndarray:
    depth_8u = cv2.convertScaleAbs(depth, alpha=0.1)
    return cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)


def main() -> int:
    p = argparse.ArgumentParser(description="ZMQ SUB for G1_camera_v2 head_camera stream")
    p.add_argument("--host", default="127.0.0.1", help="Publisher host (G1 IP or 127.0.0.1)")
    p.add_argument("--port", type=int, default=5555, help="Publisher port (default 5555)")
    p.add_argument("--topic", default="head_camera", help="ZMQ topic filter")
    p.add_argument("--no-show", action="store_true", help="Do not open an OpenCV window")
    p.add_argument("--no-depth", action="store_true", help="Do not show depth heatmap")
    p.add_argument("--seconds", type=float, default=0.0, help="Stop after N seconds (0 = until q/Ctrl+C)")
    p.add_argument("--out", default="", help="Optional path to save one sample JPEG")
    p.add_argument("--rcv-timeout-ms", type=int, default=2000)
    args = p.parse_args()

    endpoint = f"tcp://{args.host}:{args.port}"
    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    # CONFLATE is incompatible with multipart [topic, jpeg, depth]
    sub.setsockopt(zmq.RCVHWM, 2)
    sub.setsockopt(zmq.RCVTIMEO, args.rcv_timeout_ms)
    sub.connect(endpoint)
    sub.setsockopt_string(zmq.SUBSCRIBE, args.topic)

    print(f"[INFO] SUB {endpoint} topic={args.topic!r}")
    print("[INFO] Expect multipart [topic, jpeg, depth_u16]")
    print("[INFO] Press q in the window (or Ctrl+C) to stop")

    n = 0
    t0 = time.perf_counter()
    last_log = t0
    sample_path = Path(args.out) if args.out else None
    window = "head_camera"
    show = not args.no_show

    try:
        while True:
            if args.seconds > 0 and (time.perf_counter() - t0) >= args.seconds:
                break
            try:
                parts = sub.recv_multipart()
            except zmq.Again:
                print("[WARN] waiting for frames...")
                continue

            frame, depth = decode_parts(parts)
            if frame is None:
                print(f"[WARN] could not decode jpeg (parts={len(parts)})")
                continue

            n += 1
            now = time.perf_counter()
            if sample_path is not None and n == 1:
                sample_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(sample_path), frame)
                print(f"[INFO] wrote sample → {sample_path}")

            if now - last_log >= 2.0:
                hz = n / max(1e-6, now - t0)
                depth_note = f" depth={depth.shape}" if depth is not None else " depth=none"
                print(f"[INFO] Receiving at {hz:.1f} fps | frames: {n} | rgb={frame.shape}{depth_note}")
                last_log = now

            if show:
                if depth is not None and not args.no_depth:
                    disp = np.hstack((frame, depth_heatmap(depth)))
                else:
                    disp = frame
                cv2.imshow(window, disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\n[INFO] stopped")
    finally:
        elapsed = max(1e-6, time.perf_counter() - t0)
        print(f"[INFO] done: frames={n} mean_fps={n / elapsed:.1f}")
        sub.close(0)
        if show:
            cv2.destroyAllWindows()

    return 0 if n > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
