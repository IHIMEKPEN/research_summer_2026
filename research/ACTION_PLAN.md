# ICRA 2027 Action Plan (Deadline: September 15, 2026)

**Author:** Osemudiamen Andrew Ihimekpen · PVAMU CREDIT Center  
**Hard deadline:** September 15, 2026 (11:59 PM PST)  
**Working tree:** `research/` (single package — Steps 1–5)

Freeze feature creep. Prefer measured MuJoCo / UnifoLM numbers over mock priors.  
**Preferred entrypoint: Jupyter notebooks** (kernel: Research Summer 2026 / UnifoLM).

---

## This week (sim) vs next week (Jetson + G1)

| Window | Goal | Notebook |
|--------|------|----------|
| **This week** | Offline ZOH/linear + **live UnifoLM** bridges | `step3_sim_comparison.ipynb` |
| **Next week** | Same bridges on **Jetson AGX Thor** → G1 | `step5_s2r_*.ipynb` |

Do **not** leave `MOCK=True` for paper timing.

---

## Pipeline (one tree)

```text
Step 1  UnifoLM latency profile          → notebooks/step1_*.ipynb
Step 2  CUDA ESN ridge (u ∈ R^{58})      → notebooks/step2_esn_cuda_ridge.ipynb
Step 3  Dual-rate closed loop + baselines → notebooks/step3_*.ipynb
Step 4  MuJoCo wipe eval                 → notebooks/step4_mujoco_evaluation.ipynb
Step 5  S2R deploy (ZMQ / GUI / G1)      → notebooks/step5_s2r_*.ipynb
```

G1 defaults: **29-DoF** @ **100 Hz** (`src/g1_constants.py`).

---

## Immediate notebook runs (sim complete)

```text
notebooks/step3_control_baselines.ipynb     → offline ZOH / linear / PID
notebooks/step3_dual_thread_mujoco.ipynb    → MOCK=False, BRIDGE=esn|zoh|linear|pid (timing)
notebooks/step3_live_wipe_success.ipynb     → live UnifoLM wipe grasp/contact/task (Layer F)
notebooks/step3_sim_comparison.ipynb        → one-shot offline + live suite
```

Data split (`G1_Dex1_Wipe_Table`, 200 eps): **train 0–159 / held-out 160–199**  
Card: `results/step2_training/DATASET_CARD.md`

Reports land in:
- `results/step2_training/esn_heldout_eval.csv` (+ `DATASET_CARD.md`)
- `results/step3_baselines/baseline_comparison.csv` (default: held-out)
- `results/step3_dual_thread/dual_thread_report_{esn|zoh|linear|pid}_live.json`
- `results/step4_mujoco_evaluation/mujoco_eval_summary_heldout.csv`

CLI equivalents still exist (`python3 -m src.step3_*`) but notebooks are the primary workflow.

---

## Phase 2 — Ablations & consolidation (Aug 15–31)

1. Port ablations to **58-D CUDA ESN** (`N ∈ {500,1000,2000}`, `ρ ∈ {0.85,0.95,1.05}`).
2. Fill `results/step3_*`, `step4_paper_*`.
3. Wire ridge checkpoint into Step 5; run Thor ablations via `step5_s2r_esn_ablation.ipynb`.

---

## Phase 3–4 — Manuscript + submit (Sep 1–15)

IEEE: **6 pages + 1 optional refs page.** PaperPlaza ≥48 h before deadline.
