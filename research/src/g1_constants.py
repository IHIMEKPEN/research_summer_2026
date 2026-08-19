"""Unitree G1 constants shared by Steps 1–4 and the S2R deploy stack."""

from __future__ import annotations

from typing import List

# Controllable actuated joints (Dex1 wipe / 29-DoF body; Dex1-1 fingers are separate).
G1_DOF = 29
CONTROL_HZ = 100.0
VLA_HZ = 2.0

# observation.body layout (G1_Dex1_Wipe_Table / step2_esn_cuda_ridge)
LEG_WAIST_SLICE = slice(0, 15)
LEFT_ARM_SLICE = slice(15, 22)
RIGHT_ARM_SLICE = slice(22, 29)

# Conservative default position limits (rad). Per-joint URDF limits can replace these.
_G1_LIMIT = 3.1416


def g1_joint_limit_min(n: int = G1_DOF) -> List[float]:
    return [-_G1_LIMIT] * int(n)


def g1_joint_limit_max(n: int = G1_DOF) -> List[float]:
    return [_G1_LIMIT] * int(n)
