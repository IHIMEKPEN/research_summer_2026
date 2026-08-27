# Independent ESN + frozen UnifoLM final report

Created: 2026-08-27T04:47:19.091328+00:00
Host: dxs4-DGX-Station
Total runtime: 0.74 h (2647.5 s)

## Workstation
- GPUs: [{'index': '0', 'name': 'Tesla V100-DGXS-32GB', 'memory_total': '32768 MiB', 'memory_used': '117 MiB', 'memory_free': '32367 MiB'}, {'index': '1', 'name': 'Tesla V100-DGXS-32GB', 'memory_total': '32768 MiB', 'memory_used': '6 MiB', 'memory_free': '32486 MiB'}, {'index': '2', 'name': 'Tesla V100-DGXS-32GB', 'memory_total': '32768 MiB', 'memory_used': '6 MiB', 'memory_free': '32486 MiB'}, {'index': '3', 'name': 'Tesla V100-DGXS-32GB', 'memory_total': '32768 MiB', 'memory_used': '6 MiB', 'memory_free': '32486 MiB'}]
- Python/Torch/CUDA: 3.10.12 / 2.5.1+cu121 / 12.1
- Packages: {
  "transformers": "4.49.0",
  "flash_attn": "MISSING:ModuleNotFoundError",
  "mujoco": "3.11.0",
  "datasets": "5.0.1",
  "numpy": "2.2.6",
  "omegaconf": "2.3.1",
  "qwen_vl_utils": "present",
  "unifolm_vla": "present"
}

## UnifoLM teacher
- Model: `unitreerobotics/UnifoLM-VLA-Base`
- unnorm_key: `g1_wipe_table`
- Mock forbidden: true
- DAgger rounds recorded: 4
- Best checkpoint: `/home/aihimekpen/research_summer_2026/research/results/main_independent_esn/workstation_final/esn_dagger_round02_seed0.npz`

## Optimizer selection
- Selected: **spsa**
- Criterion: mean_validation_L_task_over_VAL_EPS_and_seeds
- Validation episodes (train range): [4, 5, 6]
- Method scores: {
  "random": 1200.3044450677733,
  "spsa": 646.0002744007123,
  "cem": 1288.2346109585696,
  "cmaes": 1040.4655564701025
}

## Splits
- BC init episodes: [0, 1, 2, 3]
- Training range: 0–159 (160 episodes)
- Validation subset: [4, 5, 6]
- Held-out: 160–199 (40 episodes)

## Held-out (all 40 episodes × seeds)
- Trials: 120
- Success rate: 0.0000 (95% CI [0.0, 0.031020271055929513])
- Contact: {'mean': 0.04140266801304791, 'std': 0.07154097236252803}
- Coverage: {'mean': 0.08209, 'std': 0.05125532069941618}
- Path: {'mean': 58.0129030152785, 'std': 37.31477835960555}
- Losses: L_task={'mean': 982.9449331020485, 'std': 978.9989233855177} L_teacher={'mean': 6407.0937659364845, 'std': 33294.11957858709} L_total={'mean': 982.9449331020485, 'std': 978.9989233855177}

## Ablations
See `ablation_results.csv`. Conditions: bc_initializer_only, task_loss_only,
real_sparse_teacher_only, task_plus_real_sparse_teacher, best_vs_spsa:*

## Threshold sensitivity
See `threshold_sensitivity.csv` for 80/90/95% contact×coverage gates on held-out trials.

## Large artifacts (not committed to git)
- `/home/aihimekpen/research_summer_2026/research/results/main_independent_esn/visited_round01_ep0_seed0.npz` sha256=46c463cbd384f361438e3bd5d0139803b1d88736a361676f6eb9c7cdb391cb2f bytes=880098
- `/home/aihimekpen/research_summer_2026/research/results/main_independent_esn/visited_round02_ep0_seed0.npz` sha256=778f3364f41512cdfac8249af93cd2f50b9a6b481fa0ab31b0ddde8084b2a207 bytes=879168
- `/home/aihimekpen/research_summer_2026/research/results/main_independent_esn/visited_round03_ep0_seed0.npz` sha256=21c7745b2a5abf52730a3969fba9804bf52dfea7bbc49a64b03b5d3cee9797f4 bytes=879197
- `/home/aihimekpen/research_summer_2026/research/results/main_independent_esn/visited_ep0_seed0.npz` sha256=549d59d73b03564c42b62f027f667dcb7b8a991182266806ec53767ca363ecdf bytes=170852

## Artifact index
- `/home/aihimekpen/research_summer_2026/research/results/main_independent_esn/workstation_final/run_config.json`
- `/home/aihimekpen/research_summer_2026/research/results/main_independent_esn/workstation_final/environment_audit.json`
- `/home/aihimekpen/research_summer_2026/research/results/main_independent_esn/workstation_final/teacher_cache_manifest.json`
- `/home/aihimekpen/research_summer_2026/research/results/main_independent_esn/workstation_final/dagger_history.json`
- `/home/aihimekpen/research_summer_2026/research/results/main_independent_esn/workstation_final/optimizer_comparison.csv`
- `/home/aihimekpen/research_summer_2026/research/results/main_independent_esn/workstation_final/optimizer_comparison.json`
- `/home/aihimekpen/research_summer_2026/research/results/main_independent_esn/workstation_final/heldout_episode_results.csv`
- `/home/aihimekpen/research_summer_2026/research/results/main_independent_esn/workstation_final/heldout_summary.json`
- `/home/aihimekpen/research_summer_2026/research/results/main_independent_esn/workstation_final/ablation_results.csv`
- `/home/aihimekpen/research_summer_2026/research/results/main_independent_esn/workstation_final/threshold_sensitivity.csv`
- `/home/aihimekpen/research_summer_2026/research/results/main_independent_esn/workstation_final/esn_final_selected.npz`
- `/home/aihimekpen/research_summer_2026/research/results/main_independent_esn/workstation_final/optimizer_curves.png`
- `/home/aihimekpen/research_summer_2026/research/results/main_independent_esn/workstation_final/heldout_metrics.png`
- `/home/aihimekpen/research_summer_2026/research/results/main_independent_esn/workstation_final/FINAL_REPORT.md`

## Limitations
- Cloth contact is geometric (mocap), not calibrated force.
- FlashAttention2 may be absent; SDPA fallback is used when needed.
- Root disk is nearly full; large caches/checkpoints live under `/raid`.
