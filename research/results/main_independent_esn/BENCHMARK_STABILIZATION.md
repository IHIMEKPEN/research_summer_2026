# Benchmark stabilization & hierarchical contact (post independent-ESN run)

**Date:** 2026-08-27  
**Status:** benchmark stabilization shipped; oracle gate **NO-GO** — **do not** spend
compute on SPSA/CEM/CMA-ES. Next: hierarchical contact + grasp/init so oracle ≥90%.

## Why

The frozen-UnifoLM teacher-guided independent ESN run completed with real labels
(`mock=false`) but **0/120** held-out successes. Optimizer choice is not the
bottleneck: MuJoCo QACC NaNs and unbounded path/loss accounting made the
benchmark untrustworthy, and a proprioception-only ESN is being asked to invent
contact physics.

## Immediate coding sequence

1. **Stabilize the benchmark** (this pass) — **tests green**
   - Terminate on **non-finite** q/qvel/qacc/ctrl/cmd/cloth (not on `mjWARN_BADQACC`
     alone: base pinning + stiff PD routinely emit that warning with finite state)
   - Reject >5 cm cloth jumps from path/coverage
   - Bound every loss term to `[0,1]`; report `L_grasp`…`L_teacher` separately
   - Held-out / `teacher_weight=0` → `teacher_source="none"`, `L_teacher=0`
   - Plausible path cap: `MAX_PLAUSIBLE_WIPE_PATH_M = 8.0` (multi-pass demos ~3 m OK;
     teleport-scale scores rejected)
   - Tests: `tests/test_wipe_benchmark_stability.py`

2. **Oracle / control baselines** (`notebooks/run_oracle_benchmark_gate.py`)
   - stationary, oracle ZOH, linear, PD, sparse teacher ZOH
   - Optional: real UnifoLM + `press_table`
   - **Go/no-go:** demo replay stable + plausible path on ≥90% episodes, 0 NaNs
     (NaN = `terminated_unstable` from non-finite state)

3. **Hierarchical contact controller** (`notebooks/wipe_contact_controller.py`)
   ```
   ESN intent → ContactImpedanceController → joint PD / WBC → robot
   ```
   Wraps `src.vla_ee_bridge.stabilize_joint_command` (leg freeze, rate limits,
   joint-limit clip). Geometric `press_table` remains a separate prior on the
   cloth controller until a calibrated force channel exists.
4. Enrich ESN observations (q, qdot, hand pose/vel, grasp, table-relative height,
   contact flag, coverage/phase) — still no images/language/live VLA.

5. Retrain only after gates pass (BC → DAgger real UnifoLM → residual task opt →
   freeze → 40 held-out × ≥5 seeds).

## Paper direction

> Sparse VLA supervision alone does not enable a proprioception-only ESN to learn
> stable contact-rich wiping.

Revised question:

> Can frozen VLA semantic guidance be distilled into a high-rate recurrent motion
> policy when contact regulation remains in a conventional low-level controller?

## Commands

```bash
cd research
export MUJOCO_GL=egl
export PYTHONPATH=.:notebooks:../unifolm-vla/src

# Unit + integration tests
python -m pytest tests/test_wipe_benchmark_stability.py -q

# Go/no-go oracle gate (no press prior)
python notebooks/run_oracle_benchmark_gate.py --episodes 0,1,2,3,4,5,160,161,162

# Same with geometric press_table prior
python notebooks/run_oracle_benchmark_gate.py --episodes 0,1,2,3,4,5,160,161,162 --press-table
```

Artifacts: `results/main_independent_esn/oracle_benchmark_gate/gate_{nopress,press}.json`

## Gate result (2026-08-27, 9 episodes: 0–5, 160–162)

| Condition | focus | gate_rate | NaN terms | mean contact | mean path | verdict |
|-----------|-------|-----------|-----------|--------------|-----------|---------|
| no `press_table` | `oracle_pd` | **0%** | 0 | ~0.037 | ~1.9 m | **NO-GO** |
| `press_table` | `oracle_pd` | **55.6%** (5/9) | 0 | ~0.67 | ~3.6 m | **NO-GO** (<90%) |

Failure modes with press: grasp attach miss (eps 2,5,160 → path 0); one path >8 m (ep 1).
Without press, oracle replay never meets the soft contact≥0.15 diagnostic — contact
physics is prior-dependent. **Do not retrain ESN until the hierarchical contact
layer + attach/init make oracle replay ≥90%.**

Stabilization wins already shipped: 0 false NaN terminations (after dropping
warning-counter heuristics), jump-filtered paths, bounded `L_*`, `teacher_source=none`.
