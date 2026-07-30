# Integrating S2R with `research/src` (Steps 1–4)

Package root: `research/s2r/` (renamed from `r2s_pipeline`).  
Python import: `s2r` · Defaults: **Unitree G1 29-DoF** @ 100 Hz.

## Mapping

| S2R concept | Research artifact |
|-------------|-------------------|
| `s2r.robot.G1_DOF` (=29) | `src/g1_constants.py`, `step2_esn_cuda_ridge.G1_DOF` |
| `esn` node / `train` | `src/step2_esn_cuda_ridge.py` |
| Dual-rate VLA→100 Hz | `src/step3_dual_thread_mujoco.py` |
| Baselines ZOH/linear | `src/step3_control_baselines.py` |
| Wipe MuJoCo | `src/step4_mujoco_evaluation.py` |
| Ablation configs | `s2r/config/ablation_*.yaml` |

## Next wiring

1. Load CUDA ridge checkpoint into S2R `esn` node (`models/esn_cuda_ridge/`).
2. Replace mock VLA with UnifoLM + `vla_ee_bridge`.
3. Point `sim_bridge` at Step 4 wipe scene.
4. Keep large JSONL / videos under `/raid`.
