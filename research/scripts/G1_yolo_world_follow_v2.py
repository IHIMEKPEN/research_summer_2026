#!/usr/bin/env python3
"""G1 head_camera ZMQ subscriber: RGB + depth + YOLO-World person track.

Teammate script (G1_yolo_world_follow_v2), cleaned for Stage −1.

Wire format from G1_camera_v2.py:
  multipart [topic, jpeg_bytes, depth_u16 640x480]

Publisher (terminal 1 on G1):
  python G1_camera_v2.py --fps 10

Viewer / follow (terminal 2):
  # Stage −1 sense only (default — no loco / arm commands):
  python scripts/G1_yolo_world_follow_v2.py --host 127.0.0.1

  # From laptop:
  python scripts/G1_yolo_world_follow_v2.py --host 10.54.182.34

  # Explicit motion (e-stop ready; clear workspace):
  python scripts/G1_yolo_world_follow_v2.py --host 127.0.0.1 --enable-motion --iface enP2p1s0

Press q to quit, s to snapshot.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import signal
import threading
import time

import cv2
import numpy as np
import zmq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="G1 head_camera ZMQ + YOLO-World viewer / follower")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Publisher host (127.0.0.1 on G1, or robot IP from laptop)",
    )
    parser.add_argument("--port", type=int, default=5555, help="ZMQ port (default: 5555)")
    parser.add_argument("--topic", type=str, default="head_camera", help="ZMQ topic (default: head_camera)")
    parser.add_argument("--timeout", type=int, default=5000, help="Receive timeout in ms")
    parser.add_argument("--save", action="store_true", help="Save every frame to disk")
    parser.add_argument("--save-dir", type=str, default="./g1_frames", help="Directory for --save frames")
    parser.add_argument("--no-display", action="store_true", help="Headless (no OpenCV window)")
    parser.add_argument(
        "--enable-motion",
        action="store_true",
        help="Allow LocoClient Move + hug arm action (OFF by default for Stage −1)",
    )
    parser.add_argument(
        "--iface",
        type=str,
        default="enP2p1s0",
        help="DDS network interface for unitree_sdk2py (only used with --enable-motion)",
    )
    parser.add_argument(
        "--yolo-weights",
        type=str,
        default="yolov8s-world.pt",
        help="Ultralytics YOLO-World weights",
    )
    parser.add_argument("--classes", type=str, default="person", help="Comma-separated YOLO-World classes")
    return parser.parse_args()


def g1_follower(obj_center, img_center, depth_center, sport_client) -> None:
    cx_diff = img_center[0] - obj_center[0]
    if cx_diff < -200:
        sport_client.Move(0, 0, -0.3)
    elif cx_diff > 200:
        sport_client.Move(0, 0, 0.3)
    elif depth_center > 1000.0:
        sport_client.Move(0.3, 0, 0)


def main(args: argparse.Namespace) -> None:
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.RCVTIMEO, args.timeout)
    # CONFLATE breaks multipart [topic, jpeg, depth] → fq.cpp !_more abort
    socket.setsockopt(zmq.RCVHWM, 2)
    socket.setsockopt_string(zmq.SUBSCRIBE, args.topic)

    addr = f"tcp://{args.host}:{args.port}"
    socket.connect(addr)
    print(f"[INFO] Connecting to {addr}")
    print(f"[INFO] Subscribed to topic: '{args.topic}'")
    print(f"[INFO] Motion enabled: {args.enable_motion}")
    print("[INFO] Press 'q' to quit, 's' to snapshot")
    print()

    from ultralytics import YOLO

    yolo_model = YOLO(args.yolo_weights)
    yolo_model.set_classes([c.strip() for c in args.classes.split(",") if c.strip()])
    img_center = (640 // 2, 480 // 2)

    sport_client = None
    arm_action_client = None
    if args.enable_motion:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient, action_map
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

        print(f"[INFO] Init Unitree DDS on iface={args.iface}")
        ChannelFactoryInitialize(0, args.iface)
        sport_client = LocoClient()
        sport_client.SetTimeout(10.0)
        sport_client.Init()
        arm_action_client = G1ArmActionClient()
        arm_action_client.SetTimeout(10.0)
        arm_action_client.Init()
        action_map_ref = action_map
    else:
        action_map_ref = {}

    if args.save:
        os.makedirs(args.save_dir, exist_ok=True)

    running = True

    def handle_signal(sig, frame):
        nonlocal running
        print("\n[INFO] Shutting down...")
        running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    frame_count = 0
    start_time = time.time()
    last_report = start_time
    last_frame_time = None
    latencies: list[float] = []
    follower_thread: threading.Thread | None = None
    window_name = "G1 Head Camera"

    while running:
        try:
            parts = socket.recv_multipart()
            recv_time = time.time()

            if len(parts) < 3:
                print(f"[WARN] Malformed message (need 3 parts, got {len(parts)})")
                continue

            jpeg_bytes = parts[1]
            depth_bytes = parts[2]

            frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                print("[WARN] Failed to decode JPEG")
                continue

            depth_frame_np = np.frombuffer(depth_bytes, dtype=np.uint16).reshape((480, 640))
            depth_8u = cv2.convertScaleAbs(depth_frame_np, alpha=0.1)
            heatmap = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)

            yolo_preds = yolo_model.track(frame, persist=True, max_det=1, verbose=False)
            bbox_preds = yolo_preds[0].boxes.xywh.cpu().numpy().reshape(-1)
            frame = yolo_preds[0].plot()
            cv2.circle(frame, img_center, radius=5, color=(0, 0, 255), thickness=-1)

            if len(bbox_preds) > 0:
                obj_center = (int(bbox_preds[0]), int(bbox_preds[1]))
                cv2.circle(frame, obj_center, radius=5, color=(0, 0, 255), thickness=-1)
                cv2.circle(heatmap, obj_center, radius=5, color=(0, 0, 255), thickness=-1)
                y = min(max(obj_center[1], 0), depth_frame_np.shape[0] - 1)
                x = min(max(obj_center[0], 0), depth_frame_np.shape[1] - 1)
                depth_center = float(depth_frame_np[y, x])
                cv2.putText(
                    frame,
                    f"{depth_center:.0f}",
                    obj_center,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

                if args.enable_motion and sport_client is not None and arm_action_client is not None:
                    if frame_count % 300 == 0 and depth_center < 1000:
                        threading.Thread(
                            target=arm_action_client.ExecuteAction,
                            args=(action_map_ref["hug"],),
                            daemon=True,
                        ).start()
                    elif follower_thread is None or not follower_thread.is_alive():
                        follower_thread = threading.Thread(
                            target=g1_follower,
                            args=(obj_center, img_center, depth_center, sport_client),
                            daemon=True,
                        )
                        follower_thread.start()

            frame_count += 1

            if last_frame_time is not None:
                latencies.append(recv_time - last_frame_time)
                if len(latencies) > 30:
                    latencies.pop(0)
            last_frame_time = recv_time

            if not args.no_display:
                avg_fps = 1.0 / (sum(latencies) / len(latencies)) if latencies else 0.0
                h, w = frame.shape[:2]
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (280, 60), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"Frames: {frame_count}  {w}x{h}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    1,
                )

                disp = np.hstack((frame, heatmap))
                cv2.imshow(window_name, disp)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[INFO] Quit key pressed")
                    break
                if key == ord("s"):
                    path = f"snapshot_{dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                    cv2.imwrite(path, frame)
                    print(f"[INFO] Snapshot saved: {path}")

            if args.save:
                path = os.path.join(
                    args.save_dir,
                    f"frame_{frame_count:06d}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg",
                )
                cv2.imwrite(path, frame)

            now = time.time()
            if now - last_report >= 5.0:
                avg_fps = 1.0 / (sum(latencies) / len(latencies)) if latencies else 0.0
                print(f"[INFO] Receiving at {avg_fps:.1f} fps | frames: {frame_count}")
                last_report = now

        except zmq.Again:
            print(f"[WARN] No frame in {args.timeout}ms — is G1_camera_v2.py running?")
            print(f"       Expected: tcp://{args.host}:{args.port}")
            continue
        except Exception as e:
            print(f"[ERROR] {e}")
            break

    cv2.destroyAllWindows()
    socket.close()
    context.term()
    print(f"[INFO] Received {frame_count} frames in {time.time() - start_time:.1f}s")
    print("[INFO] Client stopped.")


if __name__ == "__main__":
    main(parse_args())
