"""Unit tests for multi-task ESN helpers (no GPU / HF required)."""

from __future__ import annotations

import torch

from src.step2_esn_cuda_ridge import extract_joint_state_29d
from src.unifolm_tasks import (
    ESN_SUITE_TASK_IDS,
    esn_checkpoint_basename,
    get_task,
    list_esn_suite_tasks,
)


def test_esn_checkpoint_basenames():
    assert esn_checkpoint_basename("wipe_table") == "esn_cuda_ridge"
    assert esn_checkpoint_basename("stack_block") == "esn_cuda_ridge_stack_block"
    assert esn_checkpoint_basename("g1_clean_table") == "esn_cuda_ridge_clean_table"


def test_esn_suite_excludes_dual():
    ids = [t.id for t in list_esn_suite_tasks()]
    assert "wipe_table" in ids
    assert "dual_clean_table" not in ids
    assert set(ids) == set(ESN_SUITE_TASK_IDS)
    assert len(ids) == 11


def test_extract_joint_state_from_body_only():
    body = list(range(29))
    q = extract_joint_state_29d({"observation.body": body})
    assert q.shape == (29,)
    assert torch.allclose(q, torch.tensor(body, dtype=torch.float32))


def test_extract_joint_state_fuses_arms():
    body = [0.0] * 29
    left = [1.0] * 7
    right = [2.0] * 7
    q = extract_joint_state_29d(
        {
            "observation.body": body,
            "observation.left_arm": left,
            "observation.right_arm": right,
        }
    )
    assert q.shape == (29,)
    assert torch.allclose(q[15:22], torch.ones(7))
    assert torch.allclose(q[22:29], torch.full((7,), 2.0))


def test_get_task_aliases():
    assert get_task("wipe_table").unnorm_key == "g1_wipe_table"
    assert get_task("G1_Dex1_Wipe_Table").id == "wipe_table"
