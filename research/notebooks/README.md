# Notebooks

**Kernel (all notebooks):** **Research Summer 2026 (UnifoLM)**  
→ `/raid/data/aihimekpen/venvs/research_summer_2026` (symlinked as `research/.venv`)

```bash
bash research/scripts/ensure_jupyter_kernel.sh   # create/register kernel
python3 research/scripts/pin_notebook_kernels.py # re-pin every .ipynb
```

Run from `research/` (or use each notebook’s path-setup cell so `src` / `s2r` import).

| Notebook | Step | Purpose |
|----------|------|---------|
| `step1_unifolm_mock_profiling.ipynb` | 1 | Dry-run profiling (no weight download) |
| `step1_unifolm_profiling.ipynb` | 1 | UnifoLM latency on lab GPU |
| `step1_unifolm_nsight_systems_profiling.ipynb` | 1 | Nsight Systems traces |
| `step2_esn_cuda_ridge.ipynb` | 2 | CUDA ESN ridge on G1 wipe data |
| **`step3_control_baselines.ipynb`** | 3 | **Offline ZOH / linear / PID table** |
| **`step3_dual_thread_mujoco.ipynb`** | 3 | **Live UnifoLM dual-process** (`BRIDGE=esn\|zoh\|linear`, set `MOCK=False`) |
| **`step3_sim_comparison.ipynb`** | 3 | **This week’s suite:** offline + live bridges in one notebook |
| `step4_mujoco_evaluation.ipynb` | 4 | MuJoCo wipe oracle + metrics |
| `step5_s2r_*.ipynb` | 5 | S2R setup, ESN train, ablations, G1 / Jetson eval |

## This week (sim) — run in order

1. Open `step3_control_baselines.ipynb` → Run All (fills ZOH/linear).
2. Open `step3_dual_thread_mujoco.ipynb` → set `MOCK=False`, `BRIDGE="esn"` → Run All; repeat with `zoh` then `linear`.
3. Or just open `step3_sim_comparison.ipynb` → `MOCK=False` → Run All.

Next week: `step5_s2r_*.ipynb` on Jetson AGX Thor → G1.
