"""
MuJoCo wipe-table scene extras: mocap cloth, Dex1 gripper-driven grasp, table stains.

Used by Step 4 full MuJoCo evaluation (not Step 3 dual-process integration).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Tuple

import mujoco
import numpy as np

from src.g1_dex1 import (
    DEX1_CLOTH_HAND_OFFSET,
    hide_non_dex1_hand_geoms,
    materialize_g1_dex1_mjcf,
    resolve_wipe_hand_body,
)
from src.vla_ee_bridge import resolve_robot_mjcf

CLOTH_BODY_NAME = "wipe_cloth"
CLOTH_GEOM_NAME = "wipe_cloth_geom"
# Fallback if a model has no Dex1 palm yet (should not happen after overlay).
RIGHT_HAND_BODY = "right_dex1_base_link"
LEFT_HAND_BODY = "left_dex1_base_link"
RIGHT_WRIST_FALLBACK = "right_wrist_yaw_link"
LEFT_WRIST_FALLBACK = "left_wrist_yaw_link"

# Dex1 gripper width (rad): ~4.5 open, ~0.45 closed on G1_Dex1_Wipe_Table.
GRIPPER_OPEN_TYPICAL = 4.5
GRIPPER_CLOSED_TYPICAL = 0.45
GRIPPER_GRASP_THRESHOLD = 2.0

# Table geometry is aligned to the G1_Dex1_Wipe_Table wipe workspace
# (held-out hand-attach XY ≈ [0.23, 0.47] × [-0.46, 0.04], wipe z ≈ 0.83–0.94).
# Body origin is the table-top geom center; surface z = body_z + half_z.
TABLE_BODY_POS = np.array([0.345, -0.200, 0.805], dtype=np.float64)
TABLE_TOP_HALF_EXTENTS = np.array([0.18, 0.28, 0.025], dtype=np.float64)
TABLE_TOP_Z = float(TABLE_BODY_POS[2] + TABLE_TOP_HALF_EXTENTS[2])  # 0.830

# Cloth box half-extents (MuJoCo geom size); rest pose sits on the table surface.
CLOTH_HALF_EXTENTS = np.array([0.12, 0.08, 0.004], dtype=np.float64)
CLOTH_HALF_THICKNESS = float(CLOTH_HALF_EXTENTS[2])
CLOTH_TABLE_POS = np.array(
    [TABLE_BODY_POS[0], TABLE_BODY_POS[1], TABLE_TOP_Z + CLOTH_HALF_THICKNESS],
    dtype=np.float64,
)
CLOTH_HAND_OFFSET = np.array([0.085, -0.004, -0.045], dtype=np.float64)

GRASP_PROXIMITY_M = 0.14
ATTACH_BLEND_STEPS = 12
RELEASE_BLEND_STEPS = 10
# Contact: cloth underside within this gap above the table surface (m).
TABLE_CONTACT_Z_TOL = 0.04


class ClothState(Enum):
    ON_TABLE = auto()
    ATTACHING = auto()
    HELD = auto()
    RELEASING = auto()


def build_wipe_table_scene_model(
    robot_mjcf: Path,
    *,
    interactive_cloth: bool = True,
) -> mujoco.MjModel:
    """Compile G1 + table; cloth is a mocap body when ``interactive_cloth``."""
    robot_mjcf = materialize_g1_dex1_mjcf(robot_mjcf.resolve())
    hx, hy, hz = CLOTH_HALF_EXTENTS
    tx, ty, tz = TABLE_BODY_POS
    thx, thy, thz = TABLE_TOP_HALF_EXTENTS
    # Legs hang below the table-top center; length ≈ body_z so feet land near z=0.
    leg_half = max(0.05, float(tz) * 0.5)
    leg_z = -leg_half
    if interactive_cloth:
        cloth_xml = f"""
    <body name="{CLOTH_BODY_NAME}" mocap="true" pos="{CLOTH_TABLE_POS[0]} {CLOTH_TABLE_POS[1]} {CLOTH_TABLE_POS[2]}">
      <geom name="{CLOTH_GEOM_NAME}" type="box" size="{hx} {hy} {hz}"
            rgba="0.92 0.88 0.55 1" friction="0.9 0.3 0.01" mass="0.05"/>
    </body>"""
    else:
        cloth_xml = f"""
    <geom name="wipe_cloth" type="box" pos="{CLOTH_TABLE_POS[0]} {CLOTH_TABLE_POS[1]} {CLOTH_TABLE_POS[2]}"
          size="{hx} {hy} {hz}" rgba="0.92 0.88 0.55 1"/>"""

    scene_xml = f"""
<mujoco model="g1_wipe_table_scene">
  <include file="{robot_mjcf.name}"/>
  <worldbody>
    <body name="wipe_table" pos="{tx} {ty} {tz}">
      <geom name="table_top" type="box" size="{thx} {thy} {thz}" rgba="0.55 0.38 0.22 1"/>
      <geom name="table_leg_fl" type="cylinder" pos="{0.75*thx} {0.75*thy} {leg_z}" size="0.02 {leg_half}" rgba="0.35 0.35 0.35 1"/>
      <geom name="table_leg_fr" type="cylinder" pos="{0.75*thx} {-0.75*thy} {leg_z}" size="0.02 {leg_half}" rgba="0.35 0.35 0.35 1"/>
      <geom name="table_leg_bl" type="cylinder" pos="{-0.75*thx} {0.75*thy} {leg_z}" size="0.02 {leg_half}" rgba="0.35 0.35 0.35 1"/>
      <geom name="table_leg_br" type="cylinder" pos="{-0.75*thx} {-0.75*thy} {leg_z}" size="0.02 {leg_half}" rgba="0.35 0.35 0.35 1"/>
    </body>
    {cloth_xml}
  </worldbody>
</mujoco>
"""
    # Write beside the robot MJCF so the <include> relative path resolves; fall
    # back to /tmp if the robot tree is read-only.
    scene_path = robot_mjcf.parent / "_g1_wipe_table_scene_runtime.xml"
    try:
        scene_path.write_text(scene_xml, encoding="utf-8")
    except OSError:
        scene_path = Path("/tmp") / "_g1_wipe_table_scene_runtime.xml"
        # Absolute include so the model still finds the robot MJCF from /tmp.
        scene_xml = scene_xml.replace(
            f'<include file="{robot_mjcf.name}"/>',
            f'<include file="{robot_mjcf}"/>',
        )
        scene_path.write_text(scene_xml, encoding="utf-8")
    try:
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        hide_non_dex1_hand_geoms(model)
        return model
    finally:
        scene_path.unlink(missing_ok=True)
        if robot_mjcf.name.startswith("_g1_dex1_runtime"):
            robot_mjcf.unlink(missing_ok=True)


def _body_id(model: mujoco.MjModel, name: str) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise ValueError(f"Body not found in MJCF: {name}")
    return bid


def _mat_to_quat(rot_flat: np.ndarray) -> np.ndarray:
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(rot_flat, dtype=np.float64).reshape(9))
    return quat


def _slerp_quat(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + alpha * (q1 - q0)
        return out / np.linalg.norm(out)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    s = np.sin(theta)
    w0 = np.sin((1.0 - alpha) * theta) / s
    w1 = np.sin(alpha * theta) / s
    return w0 * q0 + w1 * q1


@dataclass
class WipeClothController:
    """
    Mocap cloth with proximity-gated grasp and blended attach/release.

    Avoids instant teleport: cloth lerps from table to hand over ``attach_blend_steps``.
    """

    model: mujoco.MjModel
    data: mujoco.MjData
    grasp_threshold: float = GRIPPER_GRASP_THRESHOLD
    grasp_proximity_m: float = GRASP_PROXIMITY_M
    attach_blend_steps: int = ATTACH_BLEND_STEPS
    release_blend_steps: int = RELEASE_BLEND_STEPS
    table_pos: np.ndarray = field(default_factory=lambda: CLOTH_TABLE_POS.copy())
    hand_offset: np.ndarray = field(default_factory=lambda: CLOTH_HAND_OFFSET.copy())
    # Live wipe: while held, keep cloth on the table plane (hand drives XY only).
    press_to_table: bool = False
    press_contact_gap_m: float = 0.012
    _cloth_body_id: int = 0
    _cloth_mocap_id: int = -1
    _right_hand_id: int = 0
    _state: ClothState = ClothState.ON_TABLE
    _blend_step: int = 0
    _blend_total: int = 1
    _blend_start_pos: np.ndarray = field(default_factory=lambda: CLOTH_TABLE_POS.copy())
    _blend_start_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    _cloth_pos: np.ndarray = field(default_factory=lambda: CLOTH_TABLE_POS.copy())
    _cloth_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    is_attached: bool = False
    last_attach_distance_m: float = float("nan")

    def __post_init__(self) -> None:
        self._cloth_body_id = _body_id(self.model, CLOTH_BODY_NAME)
        self._cloth_mocap_id = int(self.model.body_mocapid[self._cloth_body_id])
        if self._cloth_mocap_id < 0:
            raise ValueError(f"Body {CLOTH_BODY_NAME} is not a mocap body.")
        hand_name = resolve_wipe_hand_body(self.model, fallback=RIGHT_WRIST_FALLBACK)
        self._right_hand_id = _body_id(self.model, hand_name)
        self.table_pos = np.asarray(self.table_pos, dtype=np.float64)
        if np.allclose(self.hand_offset, CLOTH_HAND_OFFSET) and hand_name == RIGHT_HAND_BODY:
            self.hand_offset = DEX1_CLOTH_HAND_OFFSET.copy()
        self.hand_offset = np.asarray(self.hand_offset, dtype=np.float64)
        self.reset()

    def reset(self) -> None:
        self._state = ClothState.ON_TABLE
        self._blend_step = 0
        self.is_attached = False
        self.last_attach_distance_m = float("nan")
        self._cloth_pos = self.table_pos.copy()
        self._cloth_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self._apply_mocap_pose(self._cloth_pos, self._cloth_quat)

    def set_rest_pose(self, pos: np.ndarray) -> None:
        """Place the cloth rest pose on the table (XY from ``pos``, Z on surface)."""
        xy = np.asarray(pos, dtype=np.float64).reshape(3)
        self.table_pos = np.array(
            [xy[0], xy[1], TABLE_TOP_Z + CLOTH_HALF_THICKNESS],
            dtype=np.float64,
        )
        self.reset()

    @staticmethod
    def rest_pose_from_hand_attach(hand_attach_xyz: np.ndarray) -> np.ndarray:
        """Project a hand-attach pose onto the table surface for cloth rest."""
        p = np.asarray(hand_attach_xyz, dtype=np.float64).reshape(3)
        return np.array([p[0], p[1], TABLE_TOP_Z + CLOTH_HALF_THICKNESS], dtype=np.float64)

    def _apply_mocap_pose(self, pos: np.ndarray, quat: np.ndarray) -> None:
        self.data.mocap_pos[self._cloth_mocap_id] = np.asarray(pos, dtype=np.float64).reshape(3)
        self.data.mocap_quat[self._cloth_mocap_id] = np.asarray(quat, dtype=np.float64).reshape(4)
        self._cloth_pos = np.asarray(pos, dtype=np.float64).reshape(3).copy()
        self._cloth_quat = np.asarray(quat, dtype=np.float64).reshape(4).copy()

    def right_hand_pos(self) -> np.ndarray:
        return self.data.xpos[self._right_hand_id].copy()

    def cloth_position(self) -> np.ndarray:
        return self._cloth_pos.copy()

    def hand_to_cloth_distance(self) -> float:
        target_pos, _ = self._hand_target_pose()
        # With press_to_table the held cloth sits on the plane while the hand may
        # float — use XY distance so proximity grasp/release stays valid.
        if self.press_to_table and (
            self.is_attached or self._state in (ClothState.ATTACHING, ClothState.HELD)
        ):
            return float(np.linalg.norm(target_pos[:2] - self._cloth_pos[:2]))
        return float(np.linalg.norm(target_pos - self._cloth_pos))

    def _hand_target_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        rot = self.data.xmat[self._right_hand_id].reshape(3, 3)
        pos = self.data.xpos[self._right_hand_id] + rot @ self.hand_offset
        quat = _mat_to_quat(self.data.xmat[self._right_hand_id])
        return pos, quat

    def _wipe_held_pose(self, hand_pos: np.ndarray, hand_quat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Hand XY + table-plane Z (and flat quat) so grasped cloth actually wipes."""
        if not self.press_to_table:
            return hand_pos, hand_quat
        z = float(TABLE_TOP_Z) + float(CLOTH_HALF_THICKNESS) + float(self.press_contact_gap_m)
        pos = np.array([hand_pos[0], hand_pos[1], z], dtype=np.float64)
        # Keep cloth flat on the table while wiping (ignore wrist tilt).
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return pos, quat

    def _table_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.table_pos.copy(), np.array([1.0, 0.0, 0.0, 0.0])

    def _start_blend(self, target_state: ClothState, total_steps: int) -> None:
        self._state = target_state
        self._blend_step = 0
        self._blend_start_pos = self._cloth_pos.copy()
        self._blend_start_quat = self._cloth_quat.copy()
        self._blend_total = max(1, total_steps)

    def _tick_blend(self, target_pos: np.ndarray, target_quat: np.ndarray) -> None:
        self._blend_step += 1
        alpha = min(1.0, self._blend_step / self._blend_total)
        pos = (1.0 - alpha) * self._blend_start_pos + alpha * target_pos
        quat = _slerp_quat(self._blend_start_quat, target_quat, alpha)
        self._apply_mocap_pose(pos, quat)

    def wants_grasp(self, right_gripper: float) -> bool:
        closed = float(right_gripper) < self.grasp_threshold
        near = self.hand_to_cloth_distance() < self.grasp_proximity_m
        return closed and near

    def wants_release(self, right_gripper: float) -> bool:
        return float(right_gripper) >= self.grasp_threshold

    def synthetic_gripper_from_proximity(self) -> float:
        """
        Live-VLA proxy for Dex1 width: UnifoLM EE actions have no gripper channel.

        Near cloth → closed; far → open. Lets Step-3 live wipe reuse the same
        attach FSM / metrics as oracle Step 4 without claiming a real Dex1 command.
        """
        if self.hand_to_cloth_distance() < self.grasp_proximity_m:
            return float(GRIPPER_CLOSED_TYPICAL)
        return float(GRIPPER_OPEN_TYPICAL)

    def update(self, right_gripper: float, left_gripper: float) -> bool:
        """Advance cloth state machine; returns whether cloth is held on the hand."""
        del left_gripper
        hand_pos, hand_quat = self._hand_target_pose()
        held_pos, held_quat = self._wipe_held_pose(hand_pos, hand_quat)
        table_pos, table_quat = self._table_pose()

        if self._state == ClothState.ON_TABLE:
            self._apply_mocap_pose(table_pos, table_quat)
            self.is_attached = False
            if self.wants_grasp(right_gripper):
                self.last_attach_distance_m = self.hand_to_cloth_distance()
                self._start_blend(ClothState.ATTACHING, self.attach_blend_steps)

        elif self._state == ClothState.ATTACHING:
            self._tick_blend(held_pos, held_quat)
            if self._blend_step >= self._blend_total:
                self._state = ClothState.HELD
                self.is_attached = True
            elif self.wants_release(right_gripper):
                self._start_blend(ClothState.RELEASING, self.release_blend_steps)

        elif self._state == ClothState.HELD:
            self._apply_mocap_pose(held_pos, held_quat)
            self.is_attached = True
            if self.wants_release(right_gripper):
                self._start_blend(ClothState.RELEASING, self.release_blend_steps)

        elif self._state == ClothState.RELEASING:
            self._tick_blend(table_pos, table_quat)
            if self._blend_step >= self._blend_total:
                self._state = ClothState.ON_TABLE
                self.is_attached = False
            elif self.wants_grasp(right_gripper):
                self._start_blend(ClothState.ATTACHING, self.attach_blend_steps)

        return self.is_attached

    def gripper_openness(self, gripper_width: float) -> float:
        width = float(gripper_width)
        span = GRIPPER_OPEN_TYPICAL - GRIPPER_CLOSED_TYPICAL
        return float(np.clip((width - GRIPPER_CLOSED_TYPICAL) / span, 0.0, 1.0))


def make_wipe_scene_env_model(
    mjcf_path: Optional[Path] = None,
    *,
    interactive_cloth: bool = True,
) -> mujoco.MjModel:
    robot_path = mjcf_path if mjcf_path and mjcf_path.is_file() else resolve_robot_mjcf()
    return build_wipe_table_scene_model(robot_path, interactive_cloth=interactive_cloth)
