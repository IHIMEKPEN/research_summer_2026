# Integrating S2R (Step 5) with Steps 1–4

S2R is the **deploy layer** of the same `research/` tree — not a sibling project.

- Package: `research/src/s2r/` → `import s2r`
- Config: `research/config/`
- Shared G1: `src/g1_constants.py` (re-exported by `s2r.robot`)

## Mapping

| S2R concept | Research artifact |
|-------------|-------------------|
| `s2r.robot.G1_DOF` (=29) | `src/g1_constants.py` |
| `esn` node / `train` | `src/step2_esn_cuda_ridge.py` |
| Dual-rate VLA→100 Hz | `src/step3_dual_thread_mujoco.py` |
| Baselines ZOH/linear | `src/step3_control_baselines.py` |
| Wipe MuJoCo | `src/step4_mujoco_evaluation.py` |
| Ablation configs | `config/ablation_*.yaml` |

## Next wiring

1. Load CUDA ridge checkpoint into S2R `esn` node (`models/esn_cuda_ridge/`).
2. Replace mock VLA with UnifoLM + `vla_ee_bridge`.
3. Point `sim_bridge` at Step 4 wipe scene.
4. Keep large JSONL / videos under `/raid`.

```bash
cd research
pip install -e ".[dev,train]"
python3 -m s2r.cli deploy -c config/default.yaml
```
