# ICRA 2027 Action Plan (Deadline: September 15, 2026)

**Author:** Osemudiamen Andrew Ihimekpen · PVAMU CREDIT Center  
**Hard deadline:** September 15, 2026 (11:59 PM PST) · ~6.5 weeks from 2026-07-30  
**Working tree:** `research/`

Freeze feature creep. Execute experimental triage below. Prefer measured MuJoCo / UnifoLM numbers over mock priors.

---

## Repo map (after reorganization)

| Path | Role |
|------|------|
| `src/` | Steps 1–4 ICRA core (UnifoLM profile, CUDA ESN, dual-process, MuJoCo wipe) |
| `s2r/` | Sim-to-real deploy layer (ZMQ nodes, GUI, G1 **29-DoF**, ablations) — renamed from `r2s_pipeline` |
| `notebooks/` | Step notebooks for lab GPU runs |
| `results/` | Measured logs (gitignored contents; scaffolds with README placeholders) |
| `models/` | ESN checkpoints (`esn_cuda_ridge/`) |
| `papers/icra2027/` | IEEE draft (`main.tex`) |

---

## Phase 1 — Close the loop & replace mock baselines (Aug 1–14)

**Goal:** Live closed-loop eval + real baseline tables. **Advisor Update 2: Aug 12.**

1. **Live UnifoLM Step 3** — `python3 -m src.step3_dual_thread_mujoco` without `--mock`; log success / Hz / p99 ms.
2. **Baselines** (same wipe env):
   - Pure VLA ZOH @ 100 Hz → `src/step3_control_baselines.py`
   - VLA + linear interpolation (and optional PID)
3. **Metrics**
   - `right_ee_rmse_m` populated in `wipe_task_metrics.py` / Step 4
   - Physical jerk \(d^3q/dt^3\) (next: extend `compute_jerk_metric`)
   - Cloth / table contact condition so contact ratio is meaningful

---

## Phase 2 — Ablations & consolidation (Aug 15–31)

1. Port ablations to **58-D CUDA ESN** (`N ∈ {500,1000,2000}`, `ρ ∈ {0.85,0.95,1.05}`).
2. Fill `results/step1_baselines`, `step3_evaluation`, `step3_ablation`, `step4_paper_*`.
3. `step4_paper_figures.py` **only** on measured CSV/JSON.

---

## Phase 3 — Manuscript (Sep 1–12)

Abstract → related work (diffusion chunking / CPGs) → method (`u(t)∈R^{58}`) → results (success, jerk, latency) → wipe videos.

IEEE: **6 pages + 1 optional refs page.**

---

## Phase 4 — Submit (Sep 13–15)

Font-embedded PDF, PaperPlaza upload ≥48 h before deadline.

---

## Immediate commands

```bash
cd research

# S2R deploy (29-DoF G1 defaults)
cd s2r && pip install -e ".[dev,train]" && pytest -q
python -m s2r.cli deploy -c config/default.yaml

# ICRA core
cd ..
python3 -m src.step3_dual_thread_mujoco --mock --duration_s 5
python3 -m src.step3_control_baselines --help
python3 -m src.step4_mujoco_evaluation --episode 0 --duration_s 12
```
