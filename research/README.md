# VLA + ESN for Real-Time Humanoid Robot Control

**Author:** Osemudiamen Andrew Ihimekpen  
**Institution:** PVAMU CREDIT Center  
**Target:** ICRA 2027 (submit by **September 15, 2026**)

**Paper:** [`../papers/icra2027/`](../papers/icra2027/) · **Deadline:** 15 Sep 2026

---

## Overview

Bridge low-rate Vision-Language-Action models (~2 Hz UnifoLM) to **100 Hz** Unitree **G1 29-DoF** control with a training-light Echo State Network, evaluated in MuJoCo wipe-table and an S2R deploy stack.

```text
Camera → UnifoLM-VLA (~2 Hz) → ESN (100 Hz) → G1 joints (29-DoF)
```

---

## Layout

```text
research/
├── ACTION_PLAN.md          ← ICRA 45-day plan (phases 1–4)
├── src/                    ← Steps 1–4 experiment scripts
│   ├── g1_constants.py     ← G1_DOF=29 shared
│   ├── step1_profile_unifolm_vla0.py
│   ├── step2_esn_cuda_ridge.py
│   ├── step3_dual_thread_mujoco.py
│   ├── step3_control_baselines.py   ← ZOH / linear / PID
│   ├── step4_mujoco_evaluation.py
│   └── wipe_task_metrics.py
├── s2r/                    ← Sim-to-real ZMQ deploy (was r2s_pipeline; 29-DoF)
├── notebooks/
├── results/                ← measured logs (scaffolded dirs)
├── models/esn_cuda_ridge/
└── assets/
```

---

## Quick start

```bash
cd research
pip install -e .
pip install -r requirements.txt

# Mock dual-process
python3 -m src.step3_dual_thread_mujoco --mock --duration_s 5

# Baselines (Phase 1)
python3 -m src.step3_control_baselines --method zoh --episode 0
python3 -m src.step3_control_baselines --method linear --episode 0

# S2R deploy (G1 29-DoF defaults)
cd s2r && pip install -e ".[dev,train]"
python -m s2r.cli deploy -c config/default.yaml
pytest -q
```

GPU UnifoLM stack: `requirements-unifolm-gpu.txt`. Details: [`s2r/README.md`](s2r/README.md), [`s2r/INTEGRATION.md`](s2r/INTEGRATION.md).

---

## Methods (paper table)

| Method | Control Hz | Code |
|--------|------------|------|
| Pure VLA ZOH | 2→100 hold | `step3_control_baselines --method zoh` |
| VLA + linear / PID | 100 | `step3_control_baselines --method linear\|pid` |
| **VLA + ESN (proposed)** | **100** | `step2_esn_cuda_ridge` + Step 3/4 / `s2r` |

---

## Citation

See root README / paper draft under `papers/main paper/`.
