"""Unitree G1 29-DoF defaults for the S2R deploy stack.

Aligned with ``research/src`` wipe-table / CUDA ESN pipeline (G1_DOF = 29).
"""

from __future__ import annotations

from typing import List

G1_DOF = 29
CONTROL_HZ = 100.0
VLA_HZ = 2.0

LEG_WAIST_SLICE = slice(0, 15)
LEFT_ARM_SLICE = slice(15, 22)
RIGHT_ARM_SLICE = slice(22, 29)

_G1_LIMIT = 3.1416


def g1_joint_limit_min(n: int = G1_DOF) -> List[float]:
    return [-_G1_LIMIT] * int(n)


def g1_joint_limit_max(n: int = G1_DOF) -> List[float]:
    return [_G1_LIMIT] * int(n)
