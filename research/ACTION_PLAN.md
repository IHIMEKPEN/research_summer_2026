# ICRA 2027 Action Plan (Deadline: September 15, 2026)

**Author:** Osemudiamen Andrew Ihimekpen · PVAMU CREDIT Center  
**Hard deadline:** September 15, 2026 (11:59 PM PST) · ~6.5 weeks from 2026-07-30  
**Working tree:** `research/` (single package — Steps 1–5)

Freeze feature creep. Prefer measured MuJoCo / UnifoLM numbers over mock priors.

---

## Pipeline (one tree)

```text
Step 1  UnifoLM latency profile          → src/step1_*
Step 2  CUDA ESN ridge (u ∈ R^{58})      → src/step2_esn_cuda_ridge
Step 3  Dual-rate closed loop + baselines → src/step3_*
Step 4  MuJoCo wipe eval                 → src/step4_*
Step 5  S2R deploy (ZMQ / GUI / G1)      → src/s2r/  (import s2r)
```

| Path | Role |
|------|------|
| `src/` | Steps 1–4 scripts + `src/s2r/` deploy package |
| `config/` | S2R YAML (`default.yaml`, ablations, `platforms/g1_edu.yaml`) |
| `notebooks/` | Lab notebooks (`step1`–`step5_s2r_*`) |
| `docs/` | S2R / G1 / ablation notes |
| `tests/` | S2R unit tests |
| `data/` | Runtime episodes / ESN pairs (gitignored contents) |
| `results/` · `models/` | Measured artifacts (gitignored) |
| `papers/icra2027/` | IEEE draft |

G1 defaults: **29-DoF** @ **100 Hz** (`src/g1_constants.py`, re-exported by `s2r.robot`).

---

## Phase 1 — Close the loop & replace mock baselines (Aug 1–14)

**Goal:** Live closed-loop eval + real baseline tables. **Advisor Update 2: Aug 12.**

1. **Live UnifoLM Step 3** — `python3 -m src.step3_dual_thread_mujoco` without `--mock`; log success / Hz / p99 ms.
2. **Baselines** (same wipe env): ZOH / linear / PID via `src.step3_control_baselines`.
3. **Metrics** — EE RMSE, physical jerk, contact condition for wipe.

---

## Phase 2 — Ablations & consolidation (Aug 15–31)

1. Port ablations to **58-D CUDA ESN** (`N ∈ {500,1000,2000}`, `ρ ∈ {0.85,0.95,1.05}`).
2. Fill `results/step1_baselines`, `step3_*`, `step4_paper_*`.
3. Wire ridge checkpoint into Step 5 `esn` node; run `config/ablation_*.yaml`.

---

## Phase 3 — Manuscript (Sep 1–12)

Abstract → related work → method (`u(t)∈R^{58}`) → measured results → wipe videos.  
IEEE: **6 pages + 1 optional refs page.**

---

## Phase 4 — Submit (Sep 13–15)

Font-embedded PDF, PaperPlaza ≥48 h before deadline.

---

## Immediate commands

```bash
cd research
pip install -e ".[dev,train]"

python3 -m src.step3_dual_thread_mujoco --mock --duration_s 5
python3 -m src.step3_control_baselines --method zoh --episode 0
python3 -m src.step4_mujoco_evaluation --episode 0 --duration_s 12

python3 -m s2r.cli deploy -c config/default.yaml
pytest -q
```
