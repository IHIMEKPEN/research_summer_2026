"""Unit tests for Step 3 control baselines (ZOH / linear)."""

from __future__ import annotations

import numpy as np

from src.step3_control_baselines import (
    online_linear_command,
    upsample_linear,
    upsample_zoh,
)


def test_upsample_zoh_passthrough():
    vla = np.arange(20, dtype=np.float32).reshape(10, 2)
    out = upsample_zoh(vla)
    assert out.shape == vla.shape
    assert np.allclose(out, vla)


def test_upsample_linear_uses_vla_knots_not_midpoints_from_gt():
    # Hold every 5 ticks: knots at 0,5,9 with values 0 and 1 (then last).
    vla = np.zeros((10, 1), dtype=np.float32)
    vla[5:] = 1.0
    out = upsample_linear(vla, vla_hz=2.0, control_hz=10.0)
    assert out.shape == vla.shape
    # Midpoint between knot 0 (t=0, val=0) and knot 1 (t=5, val=1) ≈ 0.5
    assert 0.4 < float(out[2, 0]) < 0.6
    assert float(out[0, 0]) == 0.0
    assert float(out[5, 0]) == 1.0


def test_online_linear_ramps_then_holds():
    prev = np.zeros(3, dtype=np.float32)
    curr = np.ones(3, dtype=np.float32)
    mid = online_linear_command(
        prev_token=prev, curr_token=curr, ticks_since_update=25, hold_ticks=50
    )
    assert np.allclose(mid, 0.5)
    end = online_linear_command(
        prev_token=prev, curr_token=curr, ticks_since_update=100, hold_ticks=50
    )
    assert np.allclose(end, 1.0)
