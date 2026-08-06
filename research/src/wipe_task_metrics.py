"""
Numerical benchmark metrics for G1 wipe-table MuJoCo evaluation (Step 4).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.mujoco_wipe_scene import (
    CLOTH_HALF_EXTENTS,
    CLOTH_HALF_THICKNESS,
    GRASP_PROXIMITY_M,
    TABLE_CONTACT_Z_TOL,
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
    wipe_coverage_m2: float = 0.0
    wipe_phase_frames: int = 0
    task_success: bool = False
    steps: int = 0

    def to_dict(self) -> Dict[str, float | int | bool]:
        return asdict(self)


@dataclass
class WipeTaskMetricsRecorder:
    """Online accumulator for per-step wipe evaluation signals."""

    grasp_proximity_m: float = GRASP_PROXIMITY_M
    table_top_z: float = TABLE_TOP_Z
    table_contact_z_tol: float = TABLE_CONTACT_Z_TOL
    control_hz: float = 100.0
    coverage_cell_m: float = 0.02
    # Task-success gates (oracle wipe quality, not live VLA closed-loop).
    min_wipe_path_m: float = 0.30
    min_table_contact_ratio: float = 0.15
    min_wipe_coverage_m2: float = 0.008

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
        grasped = cloth_ctrl.is_attached

        if (
            not np.isnan(cloth_ctrl.last_attach_distance_m)
            and self._first_grasp_proximity is None
        ):
            self._first_grasp_proximity = float(cloth_ctrl.last_attach_distance_m)

        if grasped and not self._prev_attached and self._first_grasp_proximity is None:
            self._first_grasp_proximity = dist

        if grasped and dist > self.grasp_proximity_m * 2.5:
            self._false_attach += 1

        self._prev_attached = grasped
        self._grasped_flags.append(grasped)

    def _in_table_contact(self, cloth_z: np.ndarray) -> np.ndarray:
        """Cloth underside within ``tol`` above this episode's table plane."""
        underside = np.asarray(cloth_z, dtype=np.float64) - CLOTH_HALF_THICKNESS
        gap = underside - float(self.table_top_z)
        return (gap >= -0.002) & (gap <= float(self.table_contact_z_tol))

    def _footprint_cells(self, xy: np.ndarray) -> Set[Tuple[int, int]]:
        """Axis-aligned cloth footprint cells (ignore yaw — conservative wipe area)."""
        cell = float(self.coverage_cell_m)
        hx, hy = float(CLOTH_HALF_EXTENTS[0]), float(CLOTH_HALF_EXTENTS[1])
        cells: Set[Tuple[int, int]] = set()
        for x, y in np.asarray(xy, dtype=np.float64).reshape(-1, 2):
            i0 = int(np.floor((x - hx) / cell))
            i1 = int(np.floor((x + hx) / cell))
            j0 = int(np.floor((y - hy) / cell))
            j1 = int(np.floor((y + hy) / cell))
            for i in range(i0, i1 + 1):
                for j in range(j0, j1 + 1):
                    cells.add((i, j))
        return cells

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
            contact = self._in_table_contact(cloth[grasped_mask, 2])
            metrics.table_contact_ratio = float(contact.mean())
            if contact.any():
                cells = self._footprint_cells(xy[contact])
                metrics.wipe_coverage_m2 = float(
                    len(cells) * self.coverage_cell_m * self.coverage_cell_m
                )

        metrics.task_success = bool(
            metrics.grasp_success
            and metrics.wipe_path_length_m >= self.min_wipe_path_m
            and metrics.table_contact_ratio >= self.min_table_contact_ratio
            and metrics.wipe_coverage_m2 >= self.min_wipe_coverage_m2
        )
        return metrics
