# CLAUDE.md — Rules for AI Tools in This Repository

## Project Identity

- **Author:** Osemudiamen Andrew Ihimekpen
- **Institution:** PVAMU CREDIT Center
- **Project:** VLA + ESN for Real-Time Humanoid Control (Unitree G1, 29-DoF)
- **Venue:** ICRA 2027 — hard deadline **15 Sep 2026 23:59 PST**
- **Repo root:** `/home/aihimekpen/research_summer_2026`
- **Code root:** `research/`
- **Paper:** `papers/icra2027/`
- **Plan:** `research/ACTION_PLAN.md`

---

## Execution Rules

1. Always use **`python3`**, never `python`.
2. Prefer **`--mock` first** before downloading UnifoLM / OpenVLA weights.
3. Never treat `SUCCESS_PRIORS` / mock Bernoulli tables as paper results. Paper numbers must come from measured JSON/CSV under `results/`.
4. Never delete or silently overwrite `results/` or `models/` without confirming with the user.
5. Do not commit large generated artifacts (`results/`, `models/`, `*.mp4`, chat JSON exports).
6. Large data/checkpoints/logs → **`/raid`** (root disk is often near full). See `.cursor/rules/local-hardware.mdc`.

---

## Code Layout (do not rename casually)

| Path | Role |
|------|------|
| `research/src/step*.py` | ICRA Steps 1–4 (profile, ESN, dual-process, MuJoCo wipe) |
| `research/src/g1_constants.py` | `G1_DOF=29`, rates |
| `research/src/step3_control_baselines.py` | ZOH / linear / PID baselines |
| `research/s2r/` | Sim-to-real ZMQ deploy (package `s2r`, 29-DoF defaults) |
| `research/notebooks/` | Lab GPU notebooks |
| `research/results/` | Measured outputs (gitignored) |
| `research/models/` | Checkpoints (gitignored) |
| `unifolm-vla/` | Upstream submodule |

Script naming: `step{N}_{description}.py`. Results: `results/step{N}_{description}/`.

---

## Architecture Decisions (user approval required to change)

| Decision | Rationale |
|----------|-----------|
| Reservoir fixed; only `W_out` trained | Echo-state property; ridge closed-form |
| Production ESN input `u(t)∈R^{58}` = `[q; q*_VLA]` | Matches CUDA path / wipe data — **not** 4096-D hidden states |
| Legacy `step2_esn_bridge.py` (4096-D) | Mock/legacy only; do not use for paper tables |
| Spectral radius `ρ < 1` | Echo-state stability |
| `G1_DOF = 29`, `CONTROL_HZ = 100`, `VLA_HZ ≈ 2` | Unitree G1 + measured UnifoLM rate |
| S2R package = `research/s2r` | Deploy / DIRT ablations; defaults match 29-DoF |
| Paper claims = measured only | No mock priors in `main.tex` tables |

Color scheme for figures: red=ZOH/pure VLA, orange=linear/PID, green=VLA+ESN (proposed).

---

## ICRA Critical Path (freeze features)

| Phase | Dates | Focus |
|-------|-------|--------|
| 1 | Aug 1–14 | Live UnifoLM Step 3; ZOH/linear baselines; metrics |
| 2 | Aug 15–31 | CUDA ESN ablations; fill `results/`; Advisor Update 2 was Aug 12 |
| 3 | Sep 1–12 | Manuscript in `papers/icra2027/` (6+1 pages) |
| 4 | Sep 13–15 | PaperPlaza upload buffer |

---

## Outputs Map

```
research/results/
  step1_profiling_unifolm_vla0/   # measured UnifoLM latency
  step1_baselines/                # ZOH / linear / PID JSON
  step2_training/                 # ESN sweep
  step3_dual_thread/              # dual-process latency
  step3_evaluation/               # closed-loop (fill)
  step3_ablation/                 # N × ρ (fill)
  step4_mujoco_evaluation/        # wipe oracle
  step4_paper_figures/            # PDF/PNG from measured data only
  step4_paper_tables/

research/models/esn_cuda_ridge/   # best ridge checkpoint
```

---

## Dependencies

- Core: `numpy`, `scipy`, `matplotlib`, `pandas`, `tqdm`, `torch`
- UnifoLM GPU: `research/requirements-unifolm-gpu.txt`
- S2R: `cd research/s2r && pip install -e ".[dev,train]"`
- No `df.to_latex()` (jinja2 may be missing) — write LaTeX tables manually
