"""Dex1-1 overlay: refuse full-dex MJCF and inject 2-finger grippers."""

from __future__ import annotations

from src.g1_dex1 import (
    DEX1_SLIDE_CLOSED,
    DEX1_SLIDE_OPEN,
    _insert_into_body,
    dex1_width_to_slide,
    ensure_dex1_xml,
    is_full_dex_path,
    xml_has_dex1,
)
from pathlib import Path


def test_full_dex_paths_rejected() -> None:
    assert is_full_dex_path(Path("g1_29dof_with_hand_rev_1_0.xml"))
    assert is_full_dex_path(Path("g1_29dof_rev_1_0_with_inspire_hand_DFQ.xml"))
    assert not is_full_dex_path(Path("g1_29dof_mode_15_with_dex1_1.xml"))
    assert not is_full_dex_path(Path("g1_29dof_rev_1_0.xml"))


def test_dex1_width_mapping_ends() -> None:
    assert abs(dex1_width_to_slide(0.45) - DEX1_SLIDE_CLOSED) < 1e-9
    assert abs(dex1_width_to_slide(4.5) - DEX1_SLIDE_OPEN) < 1e-9
    mid = dex1_width_to_slide(0.5 * (0.45 + 4.5))
    assert DEX1_SLIDE_CLOSED < mid < DEX1_SLIDE_OPEN


def test_overlay_injects_four_prismatic_fingers() -> None:
    xml = """
<mujoco model="stub">
  <worldbody>
    <body name="left_wrist_yaw_link" pos="0 0 1">
      <joint name="left_wrist_yaw_joint" type="hinge"/>
      <geom size="0.02"/>
    </body>
    <body name="right_wrist_yaw_link" pos="0 0 1">
      <joint name="right_wrist_yaw_joint" type="hinge"/>
      <geom size="0.02"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="left_wrist_yaw_joint" joint="left_wrist_yaw_joint"/>
    <motor name="right_wrist_yaw_joint" joint="right_wrist_yaw_joint"/>
  </actuator>
</mujoco>
"""
    out = ensure_dex1_xml(xml, filename="g1_29dof.xml")
    assert xml_has_dex1(out)
    for name in (
        "left_dex1_finger_joint_1",
        "left_dex1_finger_joint_2",
        "right_dex1_finger_joint_1",
        "right_dex1_finger_joint_2",
    ):
        assert name in out
    assert out.count("type=\"slide\"") == 4


def test_overlay_is_idempotent() -> None:
    xml = """
<mujoco>
  <worldbody>
    <body name="left_wrist_yaw_link"><geom size="0.02"/></body>
    <body name="right_wrist_yaw_link"><geom size="0.02"/></body>
  </worldbody>
</mujoco>
"""
    once = ensure_dex1_xml(xml, filename="g1_29dof.xml")
    twice = ensure_dex1_xml(once, filename="g1_29dof.xml")
    assert once == twice


def test_inspire_xml_is_refused() -> None:
    xml = """
<mujoco>
  <worldbody>
    <body name="left_wrist_yaw_link">
      <joint name="left_thumb_joint" type="hinge"/>
    </body>
    <body name="right_wrist_yaw_link"/>
  </worldbody>
</mujoco>
"""
    try:
        ensure_dex1_xml(xml, filename="g1_inspire.xml")
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "Dex1-1" in str(exc)


def test_insert_into_nested_body() -> None:
    xml = """
<body name="parent">
  <body name="child"><geom size="0.01"/></body>
</body>
"""
    out = _insert_into_body(xml, "parent", "<geom name='extra' size='0.02'/>")
    assert "<geom name='extra' size='0.02'/>" in out
    assert out.index("<geom name='extra'") < out.rindex("</body>")
