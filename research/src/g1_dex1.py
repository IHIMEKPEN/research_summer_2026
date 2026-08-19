"""Unitree G1 Dex1-1 end-effector (matches UnifoLM / G1_Dex1_Wipe_Table).

The wipe VLA and ESN readout were trained on Dex1-1 2-finger demonstrations.
Full-dex models (Inspire RH56DFX, Dex3 ``with_hand``) change palm geometry and
contact, so this stack never loads them. If only a 29-DoF body MJCF is present,
a Dex1-1 overlay is injected (official kinematics, primitive collision geoms).

Hardware: lab G1 currently has Inspire 5-finger hands; Dex1-1 will be purchased
and attached so the real robot matches this sim + dataset.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

# Dataset gripper scalars on G1_Dex1_Wipe_Table (open → closed).
GRIPPER_OPEN_TYPICAL = 4.5
GRIPPER_CLOSED_TYPICAL = 0.45

# Official Dex1-1 prismatic range (m). Larger |slide| opens the jaws.
DEX1_SLIDE_CLOSED = -0.02
DEX1_SLIDE_OPEN = 0.0245

DEX1_FINGER_JOINTS: Tuple[str, ...] = (
    "left_dex1_finger_joint_1",
    "left_dex1_finger_joint_2",
    "right_dex1_finger_joint_1",
    "right_dex1_finger_joint_2",
)
DEX1_PALM_BODIES = {
    "left": "left_dex1_base_link",
    "right": "right_dex1_base_link",
}
RIGHT_DEX1_PALM = DEX1_PALM_BODIES["right"]

# Offset from Dex1 palm origin to a cloth-grasp point between the jaws.
DEX1_CLOTH_HAND_OFFSET = np.array([0.075, 0.0, -0.010], dtype=np.float64)

FULL_DEX_NAME_MARKERS = ("inspire", "with_hand", "dex3", "rh56")
FULL_DEX_XML_MARKERS = (
    "inspire",
    "rubber_hand",
    "left_hand_palm_joint",
    "right_hand_palm_joint",
)


def is_full_dex_path(path: Path) -> bool:
    name = path.name.lower()
    if "dex1" in name:
        return False
    return any(marker in name for marker in FULL_DEX_NAME_MARKERS)


def xml_has_dex1(xml: str) -> bool:
    return "dex1_finger_joint_1" in xml or "left_dex1_base_link" in xml


def _is_full_dex_xml(xml: str, filename: str = "") -> bool:
    if "dex1" in filename.lower() or xml_has_dex1(xml):
        return False
    lowered = xml.lower()
    fname = filename.lower()
    if "inspire" in lowered or "inspire" in fname or "with_hand" in fname:
        return True
    # Actuated multi-finger hands (not the commented-out rubber_hand stubs).
    if re.search(r'<joint name="[^"]*(finger|thumb)[^"]*"', xml, flags=re.I):
        return True
    return False


def dex1_width_to_slide(width: float) -> float:
    """Map dataset gripper scalar → Dex1-1 prismatic joint (m)."""
    span = GRIPPER_OPEN_TYPICAL - GRIPPER_CLOSED_TYPICAL
    openness = float(np.clip((float(width) - GRIPPER_CLOSED_TYPICAL) / span, 0.0, 1.0))
    return DEX1_SLIDE_CLOSED + openness * (DEX1_SLIDE_OPEN - DEX1_SLIDE_CLOSED)


def _dex1_hand_xml(side: str) -> str:
    sign = 1.0 if side == "right" else 1.0
    del sign
    y1, y2 = (-0.012, 0.012)
    return f"""
      <body name="{side}_dex1_base_link" pos="0.0415 0 0">
        <inertial pos="0.044 0 0" mass="0.19138" diaginertia="7.5e-5 5.9e-5 3.9e-5"/>
        <geom name="{side}_dex1_palm" type="box" size="0.032 0.022 0.016"
              pos="0.028 0 0" rgba="0.79 0.82 0.93 1" friction="1.2 0.3 0.01"/>
        <body name="{side}_dex1_finger_link_1">
          <joint name="{side}_dex1_finger_joint_1" type="slide" axis="0 -1 0"
                 range="-0.02 0.0245" damping="5" armature="0.001"/>
          <inertial pos="0.073 -0.011 0.009" mass="0.0868" diaginertia="2.6e-5 1.5e-5 3.6e-5"/>
          <geom name="{side}_dex1_finger_1" type="box" size="0.052 0.007 0.010"
                pos="0.055 {y1} 0.008" rgba="0.41 0.41 0.41 1" friction="1.4 0.3 0.01"/>
        </body>
        <body name="{side}_dex1_finger_link_2">
          <joint name="{side}_dex1_finger_joint_2" type="slide" axis="0 1 0"
                 range="-0.02 0.0245" damping="5" armature="0.001"/>
          <inertial pos="0.073 0.011 -0.009" mass="0.0868" diaginertia="2.6e-5 1.5e-5 3.6e-5"/>
          <geom name="{side}_dex1_finger_2" type="box" size="0.052 0.007 0.010"
                pos="0.055 {y2} -0.008" rgba="0.90 0.92 0.93 1" friction="1.4 0.3 0.01"/>
        </body>
      </body>
"""


def _dex1_actuator_xml() -> str:
    lines = []
    for name in DEX1_FINGER_JOINTS:
        lines.append(
            f'    <motor name="{name}" joint="{name}" gear="20" ctrllimited="true" ctrlrange="-20 20"/>'
        )
    return "\n".join(lines) + "\n"


def _insert_into_body(xml: str, body_name: str, inner: str) -> str:
    token = f'<body name="{body_name}"'
    start = xml.find(token)
    if start < 0:
        token = f"<body name='{body_name}'"
        start = xml.find(token)
    if start < 0:
        raise ValueError(f"MJCF is missing body {body_name!r} (needed for Dex1-1).")
    pos = xml.find(">", start) + 1
    depth = 1
    n = len(xml)
    while pos < n and depth > 0:
        nxt_open = xml.find("<body", pos)
        nxt_close = xml.find("</body>", pos)
        if nxt_close < 0:
            raise ValueError(f"Unclosed body {body_name!r} while attaching Dex1-1.")
        if nxt_open != -1 and nxt_open < nxt_close:
            depth += 1
            pos = nxt_open + 5
        else:
            depth -= 1
            if depth == 0:
                return xml[:nxt_close] + inner + xml[nxt_close:]
            pos = nxt_close + 7
    raise ValueError(f"Failed to attach Dex1-1 under {body_name!r}.")


def _insert_actuators(xml: str, actuator_xml: str) -> str:
    close = xml.rfind("</actuator>")
    if close >= 0:
        return xml[:close] + actuator_xml + xml[close:]
    close_mujoco = xml.rfind("</mujoco>")
    if close_mujoco < 0:
        raise ValueError("MJCF has neither </actuator> nor </mujoco>.")
    return xml[:close_mujoco] + f"  <actuator>\n{actuator_xml}  </actuator>\n" + xml[close_mujoco:]


def ensure_dex1_xml(xml: str, filename: str = "") -> str:
    """Return MJCF text that includes Dex1-1. Refuses full-dex / Inspire models."""
    if _is_full_dex_xml(xml, filename):
        raise ValueError(
            "Refusing full-dex / Inspire G1 MJCF. Wipe UnifoLM and "
            "G1_Dex1_Wipe_Table use Dex1-1 2-finger grippers. Point G1_MJCF at "
            "g1_29dof_mode_15_with_dex1_1 or a 29-DoF body model (overlay attached)."
        )
    if xml_has_dex1(xml):
        return xml
    out = xml
    for side, parent in (("left", "left_wrist_yaw_link"), ("right", "right_wrist_yaw_link")):
        out = _insert_into_body(out, parent, _dex1_hand_xml(side))
    out = _insert_actuators(out, _dex1_actuator_xml())
    return out


def g1_mjcf_candidates() -> Tuple[Path, ...]:
    env = Path(os.environ.get("G1_MJCF", "")).expanduser()
    home = Path.home()
    return (
        env,
        home / "unitree_mujoco" / "unitree_robots" / "g1" / "g1_29dof_mode_15_with_dex1_1.xml",
        Path("/home/aihimekpen/research/unitree_ros/robots/g1_description/g1_29dof_mode_15_with_dex1_1.xml"),
        Path("/home/aihimekpen/research/unitree_ros/robots/g1_description/g1_29dof_mode_15_with_dex1_1.urdf"),
        home / "unitree_mujoco" / "unitree_robots" / "g1" / "g1_29dof.xml",
        Path("/home/aihimekpen/research/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.xml"),
        Path("/home/aihimekpen/research/unitree_ros/robots/g1_description/g1_29dof.xml"),
    )


def materialize_g1_dex1_mjcf(robot_mjcf: Path) -> Path:
    """Write a Dex1-1 robot MJCF beside ``robot_mjcf`` (keeps mesh relative paths)."""
    robot_mjcf = robot_mjcf.resolve()
    if is_full_dex_path(robot_mjcf):
        raise ValueError(
            f"Refusing full-dex MJCF {robot_mjcf.name}. Use Dex1-1 "
            "(g1_29dof_mode_15_with_dex1_1) or a 29-DoF body XML."
        )
    text = robot_mjcf.read_text(encoding="utf-8")
    text = ensure_dex1_xml(text, filename=robot_mjcf.name)
    out = robot_mjcf.parent / "_g1_dex1_runtime.xml"
    try:
        out.write_text(text, encoding="utf-8")
        return out
    except OSError:
        raise PermissionError(
            f"Cannot write Dex1-1 overlay next to {robot_mjcf}. "
            "Copy the G1 description tree to a writable path or set G1_MJCF."
        )


def hide_non_dex1_hand_geoms(model: object) -> None:
    """Make leftover rubber-hand / Inspire meshes invisible and non-colliding."""
    import mujoco

    hide_tokens = ("rubber_hand", "inspire", "rh56")
    for i in range(int(model.ngeom)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        bid = int(model.geom_bodyid[i])
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        blob = f"{name} {bname}".lower()
        if "dex1" in blob:
            continue
        if any(tok in blob for tok in hide_tokens):
            model.geom_rgba[i, 3] = 0.0
            model.geom_contype[i] = 0
            model.geom_conaffinity[i] = 0


def resolve_wipe_hand_body(model: object, fallback: str = "right_wrist_yaw_link") -> str:
    import mujoco

    for name in (RIGHT_DEX1_PALM, fallback):
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0:
            return name
    raise ValueError("No right-hand body for wipe cloth attach.")


@dataclass
class Dex1Binding:
    """Actuator / qpos maps so the 29-DoF ESN never writes into Dex1 joints."""

    present: bool
    body_actuator_ids: np.ndarray
    body_qpos_adr: np.ndarray
    body_dof_adr: np.ndarray
    finger_actuator_ids: Dict[str, Tuple[int, int]]
    finger_qpos_adr: Dict[str, Tuple[int, int]]
    finger_dof_adr: Dict[str, Tuple[int, int]]
    palm_body_id: int

    @classmethod
    def from_model(cls, model: object, body_dof: int = 29) -> "Dex1Binding":
        import mujoco

        present = all(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0
            for name in DEX1_FINGER_JOINTS
        )
        body_act: list[int] = []
        for i in range(int(model.nu)):
            jid = int(model.actuator_trnid[i, 0])
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
            if "dex1" in jname.lower():
                continue
            body_act.append(i)
            if len(body_act) == body_dof:
                break
        if len(body_act) != body_dof:
            raise ValueError(
                f"Expected {body_dof} body actuators besides Dex1, got {len(body_act)} "
                f"(model.nu={model.nu})."
            )

        def _ids(names: Sequence[str]) -> Tuple[int, int]:
            out = []
            for name in names:
                aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                if aid < 0:
                    # Overlay motors use the joint name as actuator name.
                    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                    aid = -1
                    for k in range(int(model.nu)):
                        if int(model.actuator_trnid[k, 0]) == jid:
                            aid = k
                            break
                if aid < 0:
                    raise ValueError(f"Dex1 actuator not found for {name}.")
                out.append(aid)
            return int(out[0]), int(out[1])

        def _qpos(names: Sequence[str]) -> Tuple[int, int]:
            out = []
            for name in names:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jid < 0:
                    raise ValueError(f"Dex1 joint not found: {name}")
                out.append(int(model.jnt_qposadr[jid]))
            return int(out[0]), int(out[1])

        def _dof(names: Sequence[str]) -> Tuple[int, int]:
            out = []
            for name in names:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                out.append(int(model.jnt_dofadr[jid]))
            return int(out[0]), int(out[1])

        body_qpos = np.array(
            [int(model.jnt_qposadr[int(model.actuator_trnid[i, 0])]) for i in body_act],
            dtype=np.int32,
        )
        body_dof_adr = np.array(
            [int(model.jnt_dofadr[int(model.actuator_trnid[i, 0])]) for i in body_act],
            dtype=np.int32,
        )
        finger_act = {
            "left": _ids(DEX1_FINGER_JOINTS[:2]) if present else (-1, -1),
            "right": _ids(DEX1_FINGER_JOINTS[2:]) if present else (-1, -1),
        }
        finger_qpos = {
            "left": _qpos(DEX1_FINGER_JOINTS[:2]) if present else (-1, -1),
            "right": _qpos(DEX1_FINGER_JOINTS[2:]) if present else (-1, -1),
        }
        finger_dof = {
            "left": _dof(DEX1_FINGER_JOINTS[:2]) if present else (-1, -1),
            "right": _dof(DEX1_FINGER_JOINTS[2:]) if present else (-1, -1),
        }
        palm_name = resolve_wipe_hand_body(model)
        palm_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, palm_name))
        return cls(
            present=present,
            body_actuator_ids=np.asarray(body_act, dtype=np.int32),
            body_qpos_adr=body_qpos,
            body_dof_adr=body_dof_adr,
            finger_actuator_ids=finger_act,
            finger_qpos_adr=finger_qpos,
            finger_dof_adr=finger_dof,
            palm_body_id=palm_id,
        )

    def set_gripper_qpos(
        self,
        data: object,
        left_width: Optional[float],
        right_width: Optional[float],
        *,
        zero_vel: bool = True,
    ) -> None:
        if not self.present:
            return
        for side, width in (("left", left_width), ("right", right_width)):
            if width is None:
                continue
            slide = dex1_width_to_slide(width)
            a0, a1 = self.finger_qpos_adr[side]
            data.qpos[a0] = slide
            data.qpos[a1] = slide
            if zero_vel:
                d0, d1 = self.finger_dof_adr[side]
                data.qvel[d0] = 0.0
                data.qvel[d1] = 0.0
