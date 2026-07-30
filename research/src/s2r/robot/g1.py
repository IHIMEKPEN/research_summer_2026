"""Unitree G1 29-DoF defaults for the S2R deploy stack.

Single source of truth is ``src.g1_constants`` (Steps 1–4 + Step 5).
"""

from __future__ import annotations

from typing import List

try:
    from src.g1_constants import (
        CONTROL_HZ,
        G1_DOF,
        LEFT_ARM_SLICE,
        LEG_WAIST_SLICE,
        RIGHT_ARM_SLICE,
        VLA_HZ,
        g1_joint_limit_max,
        g1_joint_limit_min,
    )
except ImportError:
    try:
        from g1_constants import (  # type: ignore  # PYTHONPATH=src
            CONTROL_HZ,
            G1_DOF,
            LEFT_ARM_SLICE,
            LEG_WAIST_SLICE,
            RIGHT_ARM_SLICE,
            VLA_HZ,
            g1_joint_limit_max,
            g1_joint_limit_min,
        )
    except ImportError:  # offline fallback
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

__all__ = [
    "G1_DOF",
    "CONTROL_HZ",
    "VLA_HZ",
    "LEG_WAIST_SLICE",
    "LEFT_ARM_SLICE",
    "RIGHT_ARM_SLICE",
    "g1_joint_limit_min",
    "g1_joint_limit_max",
]
