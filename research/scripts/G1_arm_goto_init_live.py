#!/usr/bin/env python3
"""LIVE arm go-to-init on Unitree G1 via rt/arm_sdk (real motion).

Run ON the G1 (needs unitree_sdk2py + DDS iface). Open space — away from table.

Safety (required):
  - Overhead tether ON
  - Remote: Locked Standing = L2+Up | E-stop = L2+B  (your lab sticker)
  - NOT Running Mode (R2+A) — do not enable walking
  - Clear floor; one person on the remote

  python G1_arm_goto_init_live.py --iface enP2p1s0 --enable-motion --seconds 20

Without --enable-motion this only prints the plan (dry-run).
Ctrl+C ramps arm_sdk weight down and exits.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np


# Motor indices matching unitree_sdk2py g1_arm7 example / observation.body layout
class J:
    WaistYaw = 12
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20
    LeftWristYaw = 21
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27
    RightWristYaw = 28
    Weight = 29  # arm_sdk enable weight (not a real joint)


ARM_JOINTS = [
    J.LeftShoulderPitch,
    J.LeftShoulderRoll,
    J.LeftShoulderYaw,
    J.LeftElbow,
    J.LeftWristRoll,
    J.LeftWristPitch,
    J.LeftWristYaw,
    J.RightShoulderPitch,
    J.RightShoulderRoll,
    J.RightShoulderYaw,
    J.RightElbow,
    J.RightWristRoll,
    J.RightWristPitch,
    J.RightWristYaw,
]


def arms_ready_targets() -> dict[int, float]:
    """Mild table-ready arm targets (rad). Legs untouched. Waist not driven."""
    return {
        J.LeftShoulderPitch: 0.55,
        J.LeftShoulderRoll: 0.25,
        J.LeftShoulderYaw: -0.15,
        J.LeftElbow: 0.90,
        J.LeftWristRoll: 0.0,
        J.LeftWristPitch: 0.20,
        J.LeftWristYaw: 0.0,
        J.RightShoulderPitch: 0.45,
        J.RightShoulderRoll: -0.20,
        J.RightShoulderYaw: 0.10,
        J.RightElbow: 0.70,
        J.RightWristRoll: 0.0,
        J.RightWristPitch: 0.15,
        J.RightWristYaw: 0.0,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="G1 live arm go-to-init via arm_sdk")
    p.add_argument("--iface", default="enP2p1s0", help="DDS network interface on G1")
    p.add_argument("--seconds", type=float, default=20.0, help="Ramp duration (slow)")
    p.add_argument("--hold-seconds", type=float, default=3.0)
    p.add_argument("--kp", type=float, default=40.0, help="PD kp (official demo uses 60; start lower)")
    p.add_argument("--kd", type=float, default=1.5)
    p.add_argument("--control-dt", type=float, default=0.02)
    p.add_argument("--max-dq", type=float, default=0.35, help="Max joint speed rad/s during ramp")
    p.add_argument(
        "--enable-motion",
        action="store_true",
        help="Actually publish arm_sdk cmds (required for real motion)",
    )
    p.add_argument("--yes", action="store_true", help="Skip interactive Enter confirm")
    args = p.parse_args()

    targets = arms_ready_targets()
    print("=== G1 ARM GOTO INIT (LIVE) ===")
    print("E-stop: L2+B | Mode: Locked Standing L2+Up | NO Running Mode")
    print(f"iface={args.iface} ramp={args.seconds}s kp={args.kp} max_dq={args.max_dq}")
    print("Target (arms only):")
    for j, v in targets.items():
        print(f"  motor[{j}] -> {v:.3f} rad")

    if not args.enable_motion:
        print("\n[DRY-RUN] Pass --enable-motion to move. Exiting.")
        return 0

    if not args.yes:
        print("\nWARNING: Robot will move arms. Clear space. Tether on. Remote ready.")
        try:
            input("Press Enter to continue (Ctrl+C abort)… ")
        except KeyboardInterrupt:
            print("\naborted")
            return 1

    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC
    except Exception as e:
        print(f"[ERROR] unitree_sdk2py import failed: {e}")
        print("Install/run this ON the G1 with the SDK.")
        return 1

    ChannelFactoryInitialize(0, args.iface)

    pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
    pub.Init()
    low_cmd = unitree_hg_msg_dds__LowCmd_()
    crc = CRC()
    low_state = {"msg": None}

    def on_state(msg: LowState_):
        low_state["msg"] = msg

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(on_state, 10)

    print("[INFO] waiting for rt/lowstate…")
    t_wait = time.time()
    while low_state["msg"] is None and time.time() - t_wait < 10.0:
        time.sleep(0.05)
    if low_state["msg"] is None:
        print("[ERROR] no lowstate — check iface / robot power / DDS")
        return 1

    st0 = low_state["msg"]
    q0 = {j: float(st0.motor_state[j].q) for j in ARM_JOINTS}
    print("[INFO] current arm q:")
    for j in ARM_JOINTS:
        print(f"  motor[{j}] = {q0[j]:.3f}")

    # Enable arm_sdk with weight=1
    dt = args.control_dt
    max_delta = args.max_dq * dt
    n_ramp = max(1, int(args.seconds / dt))
    n_hold = max(0, int(args.hold_seconds / dt))
    print(f"[INFO] ramping {n_ramp} steps @ {1.0/dt:.0f} Hz — Ctrl+C or L2+B to stop")

    def write_arms(q_cmd: dict[int, float], weight: float) -> None:
        low_cmd.motor_cmd[J.Weight].q = float(weight)
        for j in ARM_JOINTS:
            low_cmd.motor_cmd[j].q = float(q_cmd[j])
            low_cmd.motor_cmd[j].dq = 0.0
            low_cmd.motor_cmd[j].kp = float(args.kp)
            low_cmd.motor_cmd[j].kd = float(args.kd)
            low_cmd.motor_cmd[j].tau = 0.0
        low_cmd.crc = crc.Crc(low_cmd)
        pub.Write(low_cmd)

    def release_weight(steps: int = 50) -> None:
        st = low_state["msg"]
        q_now = {j: float(st.motor_state[j].q) for j in ARM_JOINTS} if st else q0
        for k in range(steps):
            w = 1.0 - (k + 1) / steps
            write_arms(q_now, w)
            time.sleep(dt)
        write_arms(q_now, 0.0)

    try:
        for i in range(n_ramp + n_hold):
            alpha = min(1.0, i / max(1, n_ramp - 1))
            st = low_state["msg"]
            q_cur = {j: float(st.motor_state[j].q) for j in ARM_JOINTS}
            q_cmd = {}
            for j in ARM_JOINTS:
                goal = (1.0 - alpha) * q0[j] + alpha * targets[j]
                # rate-limit vs current measured
                delta = float(np.clip(goal - q_cur[j], -max_delta, max_delta))
                q_cmd[j] = q_cur[j] + delta
            write_arms(q_cmd, 1.0)
            if i % int(1.0 / dt) == 0:
                err = sum(abs(q_cmd[j] - targets[j]) for j in ARM_JOINTS)
                print(f"[INFO] t={i*dt:.1f}s alpha={alpha:.2f} sum|err|={err:.3f}")
            time.sleep(dt)
        print("[INFO] hold complete — releasing arm_sdk weight")
        release_weight()
        print("[INFO] done (weight=0). If arms sag, robot may need Locked Standing hold from remote.")
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C — releasing arm_sdk weight")
        try:
            release_weight()
        except Exception:
            pass
        return 130

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
