#!/usr/bin/env python3
"""Stage 0 — wipe ORACLE demo over ZMQ (no live VLA).

Local dry-run (laptop only):
  python G1_oracle_demo_stage0.py --synthetic --seconds 12

Live-robot ZMQ test (does NOT fight G1_camera_v2 on :5555):
  # Terminal A — PUB (Mac or G1). Binds *:5557 for joint_cmd, *:5556 for tokens.
  python G1_oracle_demo_stage0.py --live --synthetic --seconds 30

  # Terminal B — SUB on the G1 (log-only; no motion):
  python G1_joint_cmd_live_sub.py --host <PUB_IP> --port 5557
  # If both on G1: --host 127.0.0.1

Camera = tcp://G1:5555 topic head_camera
Control = tcp://*:5557 topic joint_cmd

This does NOT move Unitree arms yet (no 29-DoF wipe SDK path).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import zmq

G1_DOF = 29
TOPIC_TOKEN = "action_token"
TOPIC_CMD = "joint_cmd"
TOPIC_STATE = "state"


def synthetic_wipe(t: float, n: int = G1_DOF) -> np.ndarray:
    """Simple left-arm-ish wipe waveform for Stage 0 wiring tests."""
    q = np.zeros(n, dtype=np.float64)
    phase = 2.0 * math.pi * 0.08 * t
    q[15] = 0.25 * math.sin(phase)
    q[16] = 0.35 * math.sin(phase * 0.5)
    q[17] = 0.2 + 0.15 * math.sin(phase)
    q[18] = -0.4 + 0.2 * math.cos(phase)
    q[19] = 0.1 * math.sin(phase * 2)
    return q


def load_episode_tokens(episode: int, dataset_id: str) -> tuple[np.ndarray, np.ndarray]:
    import sys

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from step2_esn_cuda_ridge import load_episode_trajectory_numpy

    return load_episode_trajectory_numpy(episode, dataset_id=dataset_id)


def _pub_json(sock: zmq.Socket, topic: str, payload: dict) -> None:
    sock.send_string(topic, zmq.SNDMORE)
    sock.send_json(payload)


def main() -> int:
    p = argparse.ArgumentParser(description="Stage 0 oracle wipe demo over ZMQ")
    p.add_argument("--episode", type=int, default=160)
    p.add_argument("--dataset", default="unitreerobotics/G1_Dex1_Wipe_Table")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("--token-hz", type=float, default=2.0)
    p.add_argument("--control-hz", type=float, default=100.0)
    p.add_argument(
        "--live",
        action="store_true",
        help="Bind on all interfaces for robot LAN; skip state on :5555 (camera port)",
    )
    p.add_argument("--bind-host", default="", help="Override bind host (default 127.0.0.1 or * if --live)")
    p.add_argument("--token-port", type=int, default=5556)
    p.add_argument("--cmd-port", type=int, default=5557, help="joint_cmd port (live default 5557)")
    p.add_argument(
        "--state-port",
        type=int,
        default=5555,
        help="state port (local dry-run only; ignored with --live)",
    )
    p.add_argument("--out", default="results/stage0_oracle_demo/latest.jsonl")
    args = p.parse_args()

    host = args.bind_host or ("*" if args.live else "127.0.0.1")
    token_addr = f"tcp://{host}:{args.token_port}"
    cmd_addr = f"tcp://{host}:{args.cmd_port}"
    state_addr = f"tcp://{host}:{args.state_port}"
    publish_state = not args.live

    out_path = Path(args.out)
    research_root = Path(__file__).resolve().parents[1]
    if not out_path.is_absolute():
        out_path = (research_root / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gt = None
    if not args.synthetic:
        try:
            print(f"[INFO] Loading HF episode {args.episode} from {args.dataset} …")
            gt, _vla = load_episode_tokens(args.episode, args.dataset)
            print(f"[INFO] loaded T={len(gt)} @ ~{args.control_hz} Hz")
        except Exception as e:
            print(f"[WARN] HF load failed ({e}); falling back to --synthetic")
            args.synthetic = True

    ctx = zmq.Context.instance()
    token_pub = ctx.socket(zmq.PUB)
    cmd_pub = ctx.socket(zmq.PUB)
    state_pub = None
    token_pub.bind(token_addr)
    cmd_pub.bind(cmd_addr)
    print(f"[INFO] PUB action_token → {token_addr}")
    print(f"[INFO] PUB joint_cmd    → {cmd_addr}")
    if publish_state:
        state_pub = ctx.socket(zmq.PUB)
        state_pub.bind(state_addr)
        print(f"[INFO] PUB state       → {state_addr}")
    else:
        print("[INFO] --live: not binding :5555 (leave it for G1_camera_v2 head_camera)")

    time.sleep(0.4)
    print("[INFO] Oracle demo PUB running — NO hardware motion from this script")
    if args.live:
        print(f"[INFO] On G1 run: python G1_joint_cmd_live_sub.py --host <this_machine_ip> --port {args.cmd_port}")
    print("[INFO] Ctrl+C to stop")

    dt = 1.0 / args.control_hz
    token_period = 1.0 / args.token_hz
    t0 = time.perf_counter()
    last_token_t = -1e9
    n_tok = n_cmd = 0
    q = np.zeros(G1_DOF, dtype=np.float64)
    q_star = q.copy()
    idx = 0

    try:
        with out_path.open("w", encoding="utf-8") as log:
            while True:
                now = time.perf_counter()
                elapsed = now - t0
                if elapsed >= args.seconds:
                    break

                if args.synthetic:
                    if now - last_token_t >= token_period:
                        q_star = synthetic_wipe(elapsed)
                        last_token_t = now
                        n_tok += 1
                        payload = {
                            "ts": time.time(),
                            "topic": TOPIC_TOKEN,
                            "payload": {
                                "action": q_star.tolist(),
                                "confidence": 1.0,
                                "goal": "oracle_wipe_synthetic",
                            },
                        }
                        _pub_json(token_pub, TOPIC_TOKEN, payload)
                        log.write(json.dumps(payload) + "\n")
                    q = 0.85 * q + 0.15 * q_star
                else:
                    assert gt is not None
                    idx = min(idx, len(gt) - 1)
                    q = gt[idx].astype(np.float64)
                    hold = max(1, int(round(args.control_hz / args.token_hz)))
                    if idx % hold == 0:
                        q_star = q.copy()
                        n_tok += 1
                        payload = {
                            "ts": time.time(),
                            "topic": TOPIC_TOKEN,
                            "payload": {
                                "action": q_star.tolist(),
                                "confidence": 1.0,
                                "goal": f"oracle_wipe_ep{args.episode}",
                            },
                        }
                        _pub_json(token_pub, TOPIC_TOKEN, payload)
                        log.write(json.dumps(payload) + "\n")
                    idx += 1

                cmd = {
                    "ts": time.time(),
                    "topic": TOPIC_CMD,
                    "payload": {
                        "q": q.tolist(),
                        "source": "oracle_stage0",
                        "upsample_factor": args.control_hz / args.token_hz,
                    },
                }
                _pub_json(cmd_pub, TOPIC_CMD, cmd)
                if state_pub is not None:
                    state = {
                        "ts": time.time(),
                        "topic": TOPIC_STATE,
                        "payload": {
                            "joint_pos": q.tolist(),
                            "joint_vel": [0.0] * G1_DOF,
                            "mode": "mock",
                        },
                    }
                    _pub_json(state_pub, TOPIC_STATE, state)
                if n_cmd % 50 == 0:
                    log.write(json.dumps(cmd) + "\n")
                n_cmd += 1

                if n_cmd % int(args.control_hz) == 0:
                    print(
                        f"[INFO] t={elapsed:.1f}s tokens={n_tok} cmds={n_cmd} "
                        f"||q||={float(np.linalg.norm(q)):.3f}"
                    )

                target = t0 + n_cmd * dt
                sleep = target - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)
    except KeyboardInterrupt:
        print("\n[INFO] stopped by user")
    finally:
        token_pub.close(0)
        cmd_pub.close(0)
        if state_pub is not None:
            state_pub.close(0)

    summary = {
        "stage": 0,
        "live_zmq": bool(args.live),
        "mode": "synthetic" if args.synthetic else f"episode_{args.episode}",
        "token_addr": token_addr,
        "cmd_addr": cmd_addr,
        "seconds": args.seconds,
        "n_tokens": n_tok,
        "n_cmds": n_cmd,
        "token_hz_approx": n_tok / max(1e-6, args.seconds),
        "cmd_hz_approx": n_cmd / max(1e-6, args.seconds),
        "hardware_motion": False,
        "log": str(out_path),
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[INFO] wrote {summary_path}")
    print(json.dumps(summary, indent=2))
    print("[INFO] Demo PUB complete. No Unitree motion was commanded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
