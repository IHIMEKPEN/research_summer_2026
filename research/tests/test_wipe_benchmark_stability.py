"""Tests for stabilized wipe benchmark accounting and control baselines."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
sys.path.insert(0, str(NOTEBOOKS))
sys.path.insert(0, str(ROOT))

from src.wipe_task_metrics import MAX_CLOTH_JUMP_M, WipeTaskMetricsRecorder
from wipe_esn_experiment import (
    MAX_PLAUSIBLE_WIPE_PATH_M,
    ESN,
    _bound01,
    pack_episodes,
)
from wipe_contact_controller import ContactImpedanceController


class _FakeCloth:
    is_attached = True
    last_attach_distance_m = 0.01


def test_bound01_clips_and_maps_nan():
    assert _bound01(-1.0) == 0.0
    assert _bound01(0.5) == 0.5
    assert _bound01(2.0) == 1.0
    assert _bound01(float("nan")) == 1.0


def test_jump_segments_do_not_inflate_path_or_coverage():
    rec = WipeTaskMetricsRecorder(
        min_wipe_path_m=0.1,
        min_table_contact_ratio=0.1,
        min_wipe_coverage_m2=0.001,
        max_cloth_jump_m=MAX_CLOTH_JUMP_M,
        table_top_z=0.8,
        table_contact_z_tol=0.03,
    )
    cloth = _FakeCloth()
    # Slow wipe along table, then one 50 cm teleport that must be rejected.
    positions = [
        np.array([0.3, -0.2, 0.82]),
        np.array([0.32, -0.2, 0.82]),
        np.array([0.34, -0.2, 0.82]),
        np.array([0.34, -0.2, 0.82]) + np.array([0.5, 0.0, 0.0]),  # jump
        np.array([0.86, -0.2, 0.82]),
    ]
    for pos in positions:
        rec.record_step(
            joint_err_sq_mean=0.0,
            cloth_pos=pos,
            right_hand_pos=pos,
            right_gripper=0.4,
            cloth_ctrl=cloth,
        )
    m = rec.finalize()
    assert m.rejected_jump_segments >= 1
    assert m.max_cloth_jump_m > MAX_CLOTH_JUMP_M
    # Path must exclude the teleport (~0.5 m); only ~0.04 m of slow motion remains.
    assert m.wipe_path_length_m < 0.1
    assert m.wipe_path_length_m < MAX_PLAUSIBLE_WIPE_PATH_M


def test_contact_controller_rate_limits_and_freezes_legs():
    ctl = ContactImpedanceController()
    q0 = np.zeros(29, dtype=np.float32)
    ctl.reset(q0)
    intent = q0.copy()
    intent[0] = 1.0   # leg
    intent[20] = 1.0  # arm
    out = ctl.project(intent, q0)
    assert out[0] == 0.0
    assert abs(out[20]) <= ctl.cfg.max_arm_dq + 1e-6


def test_stationary_and_oracle_paths_are_plausible(tmp_path):
    """MuJoCo integration: stationary and demo replay cannot invent 50 m paths."""
    mjcf = ROOT / "unitree_mujoco/unitree_robots/g1/g1_29dof.xml"
    if not mjcf.is_file():
        pytest.skip("G1 MJCF unavailable")
    pytest.importorskip("mujoco")
    from datasets import load_dataset
    from wipe_control_baselines import run_baseline

    ds = load_dataset("unitreerobotics/G1_Dex1_Wipe_Table")["train"]
    eps = pack_episodes(ds, [0])
    for name in ("stationary", "oracle_linear"):
        r = run_baseline(name, eps[0], mjcf, teacher_weight=0.0, press_table=False)
        assert r["teacher_source"] == "none"
        assert r["L_teacher"] == 0.0
        assert r["wipe_path_length_m"] <= MAX_PLAUSIBLE_WIPE_PATH_M
        # All component losses must be in [0, 1].
        for key in ("L_task", "L_grasp", "L_path", "L_contact", "L_coverage", "L_smooth", "L_limits", "L_teacher"):
            assert 0.0 <= r[key] <= 1.0, (name, key, r[key])
        if name == "stationary":
            # Holding still should not accumulate a huge wipe path.
            assert r["wipe_path_length_m"] < 0.5
            # Base-pin BADQACC warnings must not abort a hold as "unstable".
            assert not r["terminated_unstable"], "stationary aborted by false QACC gate"
        if name == "oracle_linear":
            # Demo replay may wipe several meters across passes, but not tens.
            assert r["wipe_path_length_m"] < MAX_PLAUSIBLE_WIPE_PATH_M
            # If the sim went unstable we must have terminated, not scored a teleport.
            if r["terminated_unstable"]:
                assert r["L_task"] == 1.0
            else:
                assert r["wipe_path_length_m"] > 0.05  # some real wipe motion


def test_esn_rollout_teacher_none_when_weight_zero():
    mjcf = ROOT / "unitree_mujoco/unitree_robots/g1/g1_29dof.xml"
    if not mjcf.is_file():
        pytest.skip("G1 MJCF unavailable")
    from datasets import load_dataset
    from wipe_esn_experiment import rollout

    ds = load_dataset("unitreerobotics/G1_Dex1_Wipe_Table")["train"]
    eps = pack_episodes(ds, [0])
    esn = ESN(n=32, seed=0)
    r = rollout(esn, eps[0], mjcf, teacher_weight=0.0, max_s=1.0)
    assert r["teacher_source"] == "none"
    assert r["L_teacher"] == 0.0
    assert r["anchors"] == 0
