# Independent ESN + frozen UnifoLM final report

**Branch:** `main` (repo: `https://github.com/IHIMEKPEN/research_summer_2026.git`)  
**Created:** 2026-08-27T04:47:19Z (report refreshed 2026-08-27)  
**Host:** `dxs4-DGX-Station`  
**Total runtime:** 0.74 h (2647.5 s)

## Verdict

The frozen-UnifoLM teacher-guided independent ESN wipe experiment **completed end-to-end** with real teacher caches (`mock=false`). On the preregistered success gates, held-out performance is a **negative result**: **0 / 120** trials succeeded (40 episodes × 3 seeds).

## Workstation

| Item | Value |
|------|--------|
| GPUs | 4× Tesla V100-DGXS-32GB (32 GiB each) |
| Python | 3.10.12 |
| PyTorch / CUDA | 2.5.1+cu121 / 12.1 |
| transformers | 4.49.0 |
| flash_attn | missing (QWen SDPA fallback) |
| mujoco | 3.11.0 |
| unifolm_vla | present (source on `PYTHONPATH`) |

Full audit: `research/results/main_independent_esn/workstation_audit.json`

## UnifoLM teacher

| Item | Value |
|------|--------|
| Model | `unitreerobotics/UnifoLM-VLA-Base` |
| Checkpoint SHA-256 | `3a82a5ce5494ce85a3c5dca1f381195520f0f60824aafbbbe56ccf8d39f21f33` |
| Checkpoint path | `/raid/data/aihimekpen/hf_cache/hub/models--unitreerobotics--UnifoLM-VLA-Base/snapshots/06fee5014922ba6791cfe48d1e4aefac995dd8a2/checkpoints/pytorch_model.pt` |
| `unnorm_key` | `g1_wipe_table` |
| Mock fallback | **forbidden** (`allow_mock_fallback=False`; `mock_any=false`) |
| Chunk step | **0** of 25×23 (immediate next EE action for 570 ms sparse teacher) |
| DAgger rounds | 3 after BC init (history length 4 including round 0) |
| Real anchors | 22 per cache × 4 caches = **88** |
| Best DAgger checkpoint | `workstation_final/esn_dagger_round02_seed0.npz` |

## Optimizer selection

- Methods (budget 8, seeds 0/1/2): random, SPSA, CEM, CMA-ES-lite  
- Criterion: **mean validation L_task on episodes [4, 5, 6]** (inside training range 0–159; never held-out)  
- Scores: SPSA **646.0** < CMA-ES 1040.5 < random 1200.3 < CEM 1288.2  
- **Selected: SPSA**

## Splits

| Split | Episodes | Count |
|-------|----------|-------|
| BC initializer | 0–3 | 4 |
| Training range | 0–159 | 160 |
| Validation (selection) | 4, 5, 6 | 3 |
| Held-out (frozen) | 160–199 | 40 |

## Held-out (40 episodes × 3 seeds = 120 trials)

| Metric | Value |
|--------|--------|
| Successes | 0 |
| Success rate | **0.000** (95% Wilson CI **[0.000, 0.031]**) |
| Contact ratio | 0.041 ± 0.072 |
| Coverage (m²) | 0.082 ± 0.051 |
| Wipe path (m) | 58.01 ± 37.31 (inflated by unstable rollouts) |
| L_task | 982.9 ± 979.0 |
| L_teacher | 6407 ± 33294 |
| L_total | 982.9 ± 979.0 |
| Mean rollout wall time | ~4.88 s |

## Ablations (matched seeds; 0% success everywhere)

Mean L_task (lower is better):

1. `task_loss_only` — 520  
2. `best_vs_spsa:spsa` — 549  
3. `task_plus_real_sparse_teacher` — 936  
4. `bc_initializer_only` — 1171  
5. `real_sparse_teacher_only` — 3917  

**Conclusion:** real sparse UnifoLM teacher alone does not meet wipe gates; SPSA/task optimization reduces loss but does not produce task success under the preregistered criteria.

## Threshold sensitivity

At all 80% / 90% / 95% contact×coverage gate combinations on held-out trials: **0 / 120** successes (CI upper ≈ 0.031). Softening gates does not change the negative outcome.

## Reproduction

```bash
export HF_HOME=/raid/data/aihimekpen/hf_cache
export MUJOCO_GL=egl
export PYTHONPATH=research:unifolm-vla/src:research/notebooks
python research/notebooks/run_workstation_final_experiment.py
```

Mac-compatible stages (BC, collection, IK, task-only MuJoCo, artifact load) vs CUDA-only UnifoLM labeling are documented in `research/notebooks/main.ipynb`.

## Artifact index

All under `research/results/main_independent_esn/workstation_final/`:

- `run_config.json`
- `environment_audit.json`
- `teacher_cache_manifest.json`
- `dagger_history.json`
- `optimizer_comparison.csv` / `optimizer_comparison.json` / `optimizer_curves.png`
- `heldout_episode_results.csv` / `heldout_summary.json` / `heldout_metrics.png`
- `ablation_results.csv`
- `threshold_sensitivity.csv`
- `esn_final_selected.npz` (+ per-method/seed checkpoints)
- `large_artifact_hashes.txt`
- `FINAL_REPORT.md` (this file)

Teacher caches (committed): `research/results/main_independent_esn/teacher_*.npz`  
Visited RGB round bundles (local only; see hashes below).

## Large artifacts excluded from git

| Absolute path | SHA-256 | bytes |
|---------------|---------|-------|
| `/home/aihimekpen/research_summer_2026/research/results/main_independent_esn/visited_round01_ep0_seed0.npz` | `46c463cbd384f361438e3bd5d0139803b1d88736a361676f6eb9c7cdb391cb2f` | 880098 |
| `/home/aihimekpen/research_summer_2026/research/results/main_independent_esn/visited_round02_ep0_seed0.npz` | `778f3364f41512cdfac8249af93cd2f50b9a6b481fa0ab31b0ddde8084b2a207` | 879168 |
| `/home/aihimekpen/research_summer_2026/research/results/main_independent_esn/visited_round03_ep0_seed0.npz` | `21c7745b2a5abf52730a3969fba9804bf52dfea7bbc49a64b03b5d3cee9797f4` | 879197 |

Also retained on disk / `/raid`: UnifoLM weights (~19 GB), HF cache. Transfer via `scp`/`rsync` from the paths above; verify with SHA-256.

## Limitations

- Cloth contact is geometric (mocap), not calibrated force.
- Aggressive readout search triggered MuJoCo QACC NaNs; path/loss can be inflated while contact/coverage stay near zero.
- FlashAttention2 absent; SDPA used.
- Root disk nearly full; large RGB visited bundles stay on local disk / `/raid`.
