# ICRA 2027 Action Plan (Deadline: September 15, 2026)

**Author:** Osemudiamen Andrew Ihimekpen · PVAMU CREDIT Center  
**Hard deadline:** September 15, 2026 (11:59 PM PST)  
**Working tree:** `research/` (single package — Steps 1–5)

Freeze feature creep. Prefer measured MuJoCo / UnifoLM numbers over mock priors.

---

## This week (sim) vs next week (Jetson + G1)

| Window | Goal |
|--------|------|
| **This week** | Offline ZOH/linear tables + **live UnifoLM** dual-process for `esn` / `zoh` / `linear` |
| **Next week** | Same bridges on **Jetson AGX Thor** integrated to G1 (`s2r`) |

Do **not** use `--mock` for paper timing. Mock is smoke-test only.

---

## Pipeline (one tree)

```text
Step 1  UnifoLM latency profile          → src/step1_*
Step 2  CUDA ESN ridge (u ∈ R^{58})      → src/step2_esn_cuda_ridge
Step 3  Dual-rate closed loop + baselines → src/step3_*
Step 4  MuJoCo wipe eval                 → src/step4_*
Step 5  S2R deploy (ZMQ / GUI / G1)      → src/s2r/  (import s2r)
```

G1 defaults: **29-DoF** @ **100 Hz** (`src/g1_constants.py`).

---

## Immediate commands (sim complete)

```bash
cd research
pip install -e ".[dev,train]"

# 1) Offline ZOH / linear / PID vs demo (fills results/step1_baselines/)
python3 -m src.step3_control_baselines --all --episode 0

# 2) Live UnifoLM timing + bridges (omit --mock)
python3 -m src.step3_dual_thread_mujoco --bridge esn --duration_s 10
python3 -m src.step3_dual_thread_mujoco --bridge zoh --duration_s 10
python3 -m src.step3_dual_thread_mujoco --bridge linear --duration_s 10

# Or one-shot orchestration:
python3 -m src.step3_sim_comparison --episode 0 --duration_s 10

# Smoke test only (not for paper):
python3 -m src.step3_dual_thread_mujoco --mock --bridge esn --duration_s 5

pytest -q tests/test_step3_baselines.py
```

Reports land in:
- `results/step1_baselines/baseline_comparison.csv`
- `results/step3_dual_thread/dual_thread_report_{esn|zoh|linear}_live.json`
- `results/step3_evaluation/sim_comparison_summary.json`

---

## Phase 2 — Ablations & consolidation (Aug 15–31)

1. Port ablations to **58-D CUDA ESN** (`N ∈ {500,1000,2000}`, `ρ ∈ {0.85,0.95,1.05}`).
2. Fill `results/step3_*`, `step4_paper_*`.
3. Wire ridge checkpoint into Step 5 `esn` node; run `config/ablation_*.yaml` on Thor.

---

## Phase 3–4 — Manuscript + submit (Sep 1–15)

IEEE: **6 pages + 1 optional refs page.** PaperPlaza ≥48 h before deadline.
