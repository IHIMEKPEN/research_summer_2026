#!/usr/bin/env python3
"""Today's goal: slow arm go-to-init (demo start pose) over ZMQ — no full wipe.

Present robot: G1, 5-finger hands, arms hanging down, table + dark cloth in view.
Demo init: wipe ep.160 first frame (arms already up over table in sim video).

  # Terminal A (Mac or G1) — PUB slow ramp to init:
  python G1_arm_goto_init.py --preset arms_ready --live --seconds 25

  # Or load real Dex1 ep.160 frame-0 joints (needs `datasets`):
  python G1_arm_goto_init.py --episode 160 --live --seconds 25

  # Terminal B (G1) — LOG ONLY:
  python G1_joint_cmd_live_sub.py --host <PUB_IP> --port 5557

E-stop: remote L1+A (damping). Tether on. One person on the remote.
Does NOT enable Unitree arm SDK motion yet (log / rate proof + cmd stream).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import zmq

G1_DOF = 29
LEFT = slice(15, 22)
RIGHT = slice(22, 29)
LEGS = slice(0, 12)
WAIST = slice(12, 15)


def hang_down_q() -> np.ndarray:
    """Approximate stand with arms down (start). Legs/waist ~0."""
    return np.zeros(G1_DOF, dtype=np.float64)


def arms_ready_preset() -> np.ndarray:
    """Approximate 'arms up / table-ready' when HF ep.160 is unavailable.

    Tuned as a *safe visual target* for go-to-init practice — not exact Dex1 IK.
    Left slightly forward (wipe side); right ready. Legs frozen at 0.
    """
    q = hang_down_q()
    # left arm: pitch up, slight roll/yaw, elbow bent
    q[15:22] = np.array([0.55, 0.25, -0.15, 0.90, 0.0, 0.20, 0.0], dtype=np.float64)
    # right arm: raised ready, less extended
    q[22:29] = np.array([0.45, -0.20, 0.10, 0.70, 0.0, 0.15, 0.0], dtype=np.float64)
    return q


def load_episode_init(episode: int, dataset_id: str) -> np.ndarray:
    import sys

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from vla_ee_bridge import load_wipe_table_init_joints

    return load_wipe_table_init_joints(episode, dataset_id=dataset_id).astype(np.float64)


def main() -> int:
    p = argparse.ArgumentParser(description="Slow arm go-to-init over ZMQ (today's success metric)")
    p.add_argument("--episode", type=int, default=None, help="Use frame-0 joints from wipe episode")
    p.add_argument("--dataset", default="unitreerobotics/G1_Dex1_Wipe_Table")
    p.add_argument("--preset", choices=("arms_ready",), default="arms_ready")
    p.add_argument("--seconds", type=float, default=20.0, help="Ramp duration (slow)")
    p.add_argument("--control-hz", type=float, default=50.0, help="Cmd rate (keep modest today)")
    p.add_argument("--live", action="store_true", help="Bind tcp://*:cmd-port for robot LAN")
    p.add_argument("--cmd-port", type=int, default=5557)
    p.add_argument("--bind-host", default="")
    p.add_argument("--hold-seconds", type=float, default=3.0, help="Hold final pose after ramp")
    p.add_argument(
        "--out",
        default="results/arm_goto_init/latest.jsonl",
        help="JSONL log under research/",
    )
    p.add_argument(
        "--instruction",
        default="Go to that table and wipe the dirt off the table using the cloth on the table.",
        help="Logged mission text (not sent to VLA today)",
    )
    args = p.parse_args()

    host = args.bind_host or ("*" if args.live else "127.0.0.1")
    cmd_addr = f"tcp://{host}:{args.cmd_port}"

    q0 = hang_down_q()
    goal_src = "preset:arms_ready"
    if args.episode is not None:
        try:
            q1 = load_episode_init(args.episode, args.dataset)
            # Freeze legs to current hang (do not replay demo locomotion today)
            q1 = q1.copy()
            q1[LEGS] = q0[LEGS]
            q1[WAIST] = q0[WAIST]
            goal_src = f"episode_{args.episode}_frame0_arms_only"
            print(f"[INFO] Loaded demo init from episode {args.episode} (arms only; legs frozen)")
        except Exception as e:
            print(f"[WARN] episode init load failed ({e}); using --preset {args.preset}")
            q1 = arms_ready_preset()
            goal_src = f"preset:{args.preset}"
    else:
        q1 = arms_ready_preset()

    research_root = Path(__file__).resolve().parents[1]
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (research_root / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Persist goal for the team
    goal_path = out_path.with_name("demo_init_goal.json")
    goal_path.write_text(
        json.dumps(
            {
                "present_robot": {
                    "hands": "5-finger (not Dex1)",
                    "start_pose": "arms_hanging_down",
                    "workspace": "white_table_dark_cloth_black_ring_dirt",
                    "camera": "head_downward_rgb_depth_port_5555",
                },
                "demo_init": {
                    "source": goal_src,
                    "sim_reference": "results/step4_mujoco_evaluation/table_wipe_ep160_oracle_esn.mp4 (arms already up)",
                    "q_goal_29d": q1.tolist(),
                    "legs_waist_frozen": True,
                    "fingers_not_commanded": True,
                },
                "instruction": args.instruction,
                "e_stop": "Remote L1+A (damping). Tether on. Ctrl+C scripts.",
                "today_success": "slow arm motion toward init + verified e-stop",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[INFO] wrote goal card → {goal_path}")

    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.bind(cmd_addr)
    print(f"[INFO] PUB joint_cmd → {cmd_addr}")
    print(f"[INFO] instruction: {args.instruction}")
    print("[INFO] E-STOP: remote L1+A (damping). Keep tether on.")
    print("[INFO] NO Unitree SDK motion from this script — SUB logs cmds only unless you wire arms later.")
    if args.live:
        print(f"[INFO] On G1: python G1_joint_cmd_live_sub.py --host <this_ip> --port {args.cmd_port}")
    time.sleep(0.4)

    dt = 1.0 / args.control_hz
    n_ramp = max(1, int(args.seconds * args.control_hz))
    n_hold = max(0, int(args.hold_seconds * args.control_hz))
    t0 = time.perf_counter()
    n = 0

    try:
        with out_path.open("w", encoding="utf-8") as log:
            for i in range(n_ramp + n_hold):
                alpha = min(1.0, i / max(1, n_ramp - 1))
                q = (1.0 - alpha) * q0 + alpha * q1
                # hard freeze legs/waist every tick
                q[LEGS] = q0[LEGS]
                q[WAIST] = q0[WAIST]
                msg = {
                    "ts": time.time(),
                    "topic": "joint_cmd",
                    "payload": {
                        "q": q.tolist(),
                        "source": "arm_goto_init",
                        "alpha": alpha,
                        "goal_src": goal_src,
                        "phase": "ramp" if i < n_ramp else "hold",
                    },
                }
                pub.send_string("joint_cmd", zmq.SNDMORE)
                pub.send_json(msg)
                if n % 25 == 0:
                    log.write(json.dumps(msg) + "\n")
                n += 1
                if n % int(args.control_hz) == 0:
                    print(
                        f"[INFO] t={time.perf_counter()-t0:.1f}s alpha={alpha:.2f} "
                        f"||q_arm||={float(np.linalg.norm(q[LEFT])+np.linalg.norm(q[RIGHT])):.3f}"
                    )
                target = t0 + n * dt
                sleep = target - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)
    except KeyboardInterrupt:
        print("\n[INFO] stopped (treat as soft e-stop for this PUB)")
    finally:
        pub.close(0)

    summary = {
        "cmds": n,
        "goal_src": goal_src,
        "cmd_addr": cmd_addr,
        "hardware_motion": False,
        "log": str(out_path),
        "goal_card": str(goal_path),
    }
    out_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("[INFO] Ramp done. Success today = arms commanded toward init + e-stop known.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
