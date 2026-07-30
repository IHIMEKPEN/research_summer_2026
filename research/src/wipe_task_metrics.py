"""
Numerical benchmark metrics for G1 wipe-table MuJoCo evaluation (Step 4).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.mujoco_wipe_scene import (
    GRIPPER_GRASP_THRESHOLD,
    TABLE_TOP_Z,
    WipeClothController,
)

@dataclass
class WipeTaskMetrics:
    """Aggregated wipe-task benchmark (motion + interaction)."""

    joint_rmse: float = 0.0
    joint_mse: float = 0.0
    right_ee_rmse_m: float = 0.0
    max_cloth_jump_m: float = 0.0
    mean_cloth_jump_m: float = 0.0
    grasp_proximity_error_m: float = 0.0
    grasp_success: bool = False
    false_attach_frames: int = 0
    grasp_frames: int = 0
    wipe_path_length_m: float = 0.0
    table_contact_ratio: float = 0.0
    wipe_phase_frames: int = 0
    steps: int = 0

    def to_dict(self) -> Dict[str, float | int | bool]:
        return asdict(self)


@dataclass
class WipeTaskMetricsRecorder:
    """Online accumulator for per-step wipe evaluation signals."""

    grasp_proximity_m: float = 0.14
    table_contact_z_tol: float = 0.02
    control_hz: float = 100.0

    _joint_sq_err: List[float] = field(default_factory=list)
    _cloth_positions: List[np.ndarray] = field(default_factory=list)
    _right_hand_positions: List[np.ndarray] = field(default_factory=list)
    _right_ee_targets: List[np.ndarray] = field(default_factory=list)
    _grasped_flags: List[bool] = field(default_factory=list)
    _right_gripper: List[float] = field(default_factory=list)
    _first_grasp_proximity: Optional[float] = None
    _prev_attached: bool = False
    _false_attach: int = 0

    def record_step(
        self,
        *,
        joint_err_sq_mean: float,
        cloth_pos: np.ndarray,
        right_hand_pos: np.ndarray,
        right_gripper: float,
        cloth_ctrl: WipeClothController,
        right_ee_target: Optional[np.ndarray] = None,
    ) -> None:
        self._joint_sq_err.append(float(joint_err_sq_mean))
        cloth_pos = np.asarray(cloth_pos, dtype=np.float64).reshape(3)
        hand_pos = np.asarray(right_hand_pos, dtype=np.float64).reshape(3)
        self._cloth_positions.append(cloth_pos.copy())
        self._right_hand_positions.append(hand_pos.copy())
        self._right_gripper.append(float(right_gripper))
        if right_ee_target is not None:
            self._right_ee_targets.append(
                np.asarray(right_ee_target, dtype=np.float64).reshape(3).copy()
            )

        dist = float(np.linalg.norm(hand_pos - cloth_pos))
        gripper_closed = float(right_gripper) < GRIPPER_GRASP_THRESHOLD
        grasped = cloth_ctrl.is_attached

        if (
            not np.isnan(cloth_ctrl.last_attach_distance_m)
            and self._first_grasp_proximity is None
        ):
            self._first_grasp_proximity = float(cloth_ctrl.last_attach_distance_m)

        if grasped and not self._prev_attached and self._first_grasp_proximity is None:
            self._first_grasp_proximity = dist

        if gripper_closed and not grasped and dist > self.grasp_proximity_m:
            pass  # failed grasp attempt (gripper closed but too far)

        if grasped and dist > self.grasp_proximity_m * 2.5:
            self._false_attach += 1

        self._prev_attached = grasped
        self._grasped_flags.append(grasped)

    def finalize(self) -> WipeTaskMetrics:
        metrics = WipeTaskMetrics()
        n = len(self._joint_sq_err)
        if n == 0:
            return metrics

        metrics.steps = n
        metrics.joint_mse = float(np.mean(self._joint_sq_err))
        metrics.joint_rmse = float(metrics.joint_mse ** 0.5)
        metrics.grasp_frames = int(sum(self._grasped_flags))
        metrics.false_attach_frames = self._false_attach

        cloth = np.stack(self._cloth_positions, axis=0)
        hand = np.stack(self._right_hand_positions, axis=0)

        if self._right_ee_targets:
            tgt = np.stack(self._right_ee_targets, axis=0)
            m = min(len(hand), len(tgt))
            if m > 0:
                metrics.right_ee_rmse_m = float(
                    np.sqrt(np.mean(np.sum((hand[:m] - tgt[:m]) ** 2, axis=1)))
                )
        elif n > 1:
            # Fallback: EE tracking proxy vs episode-mean hand position when no GT target.
            metrics.right_ee_rmse_m = float(
                np.sqrt(np.mean(np.sum((hand - hand.mean(axis=0)) ** 2, axis=1)))
            )

        if n > 1:
            jumps = np.linalg.norm(np.diff(cloth, axis=0), axis=1)
            metrics.max_cloth_jump_m = float(jumps.max())
            metrics.mean_cloth_jump_m = float(jumps.mean())

        if self._first_grasp_proximity is not None:
            metrics.grasp_proximity_error_m = self._first_grasp_proximity
            metrics.grasp_success = self._first_grasp_proximity < self.grasp_proximity_m

        grasped_mask = np.asarray(self._grasped_flags, dtype=bool)
        if grasped_mask.any():
            metrics.wipe_phase_frames = int(grasped_mask.sum())
            xy = cloth[grasped_mask, :2]
            if xy.shape[0] > 1:
                metrics.wipe_path_length_m = float(
                    np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1))
                )
            on_table = np.abs(cloth[grasped_mask, 2] - TABLE_TOP_Z) < self.table_contact_z_tol
            metrics.table_contact_ratio = float(on_table.mean())

        return metrics
