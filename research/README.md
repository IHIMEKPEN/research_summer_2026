# VLA + ESN for Real-Time Humanoid Control (Unitree G1)

**Author:** Osemudiamen Andrew Ihimekpen · PVAMU CREDIT Center · **ICRA 2027**  
**Deadline:** 15 Sep 2026 · Plan: [`ACTION_PLAN.md`](ACTION_PLAN.md) · Paper: [`../papers/icra2027/`](../papers/icra2027/)

One research tree — profile → train ESN → closed-loop MuJoCo → **S2R deploy** (29-DoF).

```text
Camera → UnifoLM-VLA (~2 Hz) → ESN (u ∈ R^{58}) → G1 joints @ 100 Hz
                                      ↓
                         Step 5 S2R ZMQ nodes / GUI / G1 bridge
```

---

## Layout

```text
research/
├── ACTION_PLAN.md
├── README.md                 ← this file
├── config/                   ← S2R YAML (default + ablations + G1 platform)
├── src/
│   ├── step1_*.py … step4_*.py   # profiling, CUDA ESN, dual-process, MuJoCo wipe
│   ├── step3_control_baselines.py
│   ├── g1_constants.py           # G1_DOF = 29 (shared)
│   └── s2r/                      # Step 5 deploy package (import s2r)
├── notebooks/                # step1–step5 notebooks
├── scripts/                  # profiling helpers
├── docs/                     # S2R / G1 / ablation docs
├── tests/                    # S2R unit tests
├── data/                     # runtime episodes (gitignored contents)
├── results/ · models/        # measured artifacts (gitignored)
└── docker/
```

---

## Pipeline steps

| Step | Module | Deliverable |
|------|--------|-------------|
| 1 | `src.step1_profile_unifolm_vla0` | UnifoLM latency (~508 ms / ~2 Hz) |
| 2 | `src.step2_esn_cuda_ridge` | CUDA ESN + ridge `W_out` |
| 3 | `src.step3_dual_thread_mujoco` + `step3_control_baselines` | Live/mock loop; ZOH/linear/PID |
| 4 | `src.step4_mujoco_evaluation` | Wipe-table oracle + metrics |
| 5 | `s2r.cli` | ZMQ multi-node deploy, GUI, G1 bridge, ablations |

---

## Quick start

```bash
cd research
pip install -e ".[dev,train]"

# Steps 1–4
python3 -m src.step3_dual_thread_mujoco --mock --duration_s 5
python3 -m src.step3_control_baselines --method zoh --episode 0
python3 -m src.step4_mujoco_evaluation --episode 0 --duration_s 12

# Step 5 — S2R deploy (from research/)
python3 -m s2r.cli deploy -c config/default.yaml
# GUI → http://127.0.0.1:8080

pytest -q
```

GPU stacks: `requirements-unifolm-gpu.txt`, `requirements-gpu.txt`.  
S2R extras: `pip install -e ".[models,g1]"`.

More: [`docs/`](docs/) · [`notebooks/README.md`](notebooks/README.md)
