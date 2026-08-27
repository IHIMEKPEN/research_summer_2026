"""Hierarchical contact layer between motion intent and joint PD.

Intended stack (revised research direction):

    task-specific ESN  ->  desired arm trajectory
          ->  ContactImpedanceController (this module)
          ->  joint PD / Unitree WBC
          ->  robot

The ESN emits motion intent only. This layer owns table contact, rate limits,
workspace clamps, frozen legs, and joint safety. It does **not** require images,
language, or live UnifoLM.

Reuses the live-wipe stabilizers in ``src.vla_ee_bridge`` so sim and deploy
share the same rate-limit / leg-freeze / joint-limit semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from wipe_esn_experiment import G1_DOF

LEG_SLICE = slice(0, 12)


@dataclass
class ContactImpedanceConfig:
    max_arm_dq: float = 0.06          # rad per 10 ms tick
    max_waist_dq: float = 0.04
    max_leg_dq: float = 0.02
    table_z: float = 0.80
    press_offset_m: float = 0.0       # geometric prior; >0 presses into table frame
    freeze_legs: bool = True
    workspace_xyz_lo: tuple = (0.05, -0.60, 0.72)
    workspace_xyz_hi: tuple = (0.70, 0.20, 1.20)


class ContactImpedanceController:
    """Rate-limit and safety-project a 29-D joint intent relative to current q."""

    def __init__(self, cfg: ContactImpedanceConfig | None = None, model=None):
        self.cfg = cfg or ContactImpedanceConfig()
        self.model = model
        self._q_ref_legs: np.ndarray | None = None
        self._prev_cmd: np.ndarray | None = None

    def reset(self, q0: np.ndarray):
        q0 = np.asarray(q0, dtype=np.float32).reshape(G1_DOF)
        self._q_ref_legs = q0[LEG_SLICE].copy()
        self._prev_cmd = q0.copy()

    def project(self, q_intent: np.ndarray, q_current: np.ndarray) -> np.ndarray:
        from src.vla_ee_bridge import stabilize_joint_command

        intent = np.asarray(q_intent, dtype=np.float32).reshape(G1_DOF)
        current = np.asarray(q_current, dtype=np.float32).reshape(G1_DOF)
        freeze = None
        if self.cfg.freeze_legs:
            if self._q_ref_legs is None:
                self._q_ref_legs = current[LEG_SLICE].copy()
            freeze = current.copy()
            freeze[LEG_SLICE] = self._q_ref_legs
        rate_from = self._prev_cmd if self._prev_cmd is not None else current
        out = stabilize_joint_command(
            intent,
            current_29d=current,
            freeze_legs_to=freeze,
            max_dq_leg=self.cfg.max_leg_dq,
            max_dq_waist=self.cfg.max_waist_dq,
            max_dq_arm=self.cfg.max_arm_dq,
            model=self.model,
            rate_from=rate_from,
        )
        self._prev_cmd = out.copy()
        return out

    def apply_press_table_flag(self, cloth_ctrl, enabled: Optional[bool] = None) -> None:
        """Toggle geometric table-press prior on the mocap cloth controller."""
        if enabled is None:
            enabled = bool(self.cfg.press_offset_m > 0.0)
        if hasattr(cloth_ctrl, "press_to_table"):
            cloth_ctrl.press_to_table = bool(enabled)
