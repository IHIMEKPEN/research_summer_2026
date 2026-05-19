# VLA + ESN for Real-Time Humanoid Robot Control

**Author:** Osemudiamen Andrew Ihimekpen  
**Institution:** PVAMU CREDIT Center  
**Summer 2026 Research Project**  
**Target:** ICRA 2027

---

## Overview

This project bridges Vision-Language-Action (VLA) models with Echo State Networks (ESN) to enable real-time control of the Unitree G1 humanoid robot.

**The Problem:** OpenVLA, a state-of-the-art VLA model, produces actions at ~3.2 Hz. The Unitree G1 low-level controller requires commands at 100 Hz. This 31× frequency gap causes unstable, jerky motion and task failure.

**The Solution:** An ESN bridge sits between OpenVLA and the G1 controller. It maintains a recurrent reservoir state that evolves at 100 Hz, interpolating smoothly between VLA decisions. The only trained parameter is the linear readout matrix W_out, fit in seconds via ridge regression — no backpropagation required.

```
Camera ──→ OpenVLA 7B ──→ ESN Reservoir ──→ W_out ──→ G1 Joints
              3.2 Hz           100 Hz                   100 Hz
              380 ms                                      <1 ms
```

---

## Repository Structure

```
research/
├── README.md
├── pyproject.toml
├── src/                         ← import as `src.step1_profile_openvla`, etc.
│   ├── step1_profile_openvla.py
│   ├── step1_baseline_comparison.py
│   ├── step2_esn_bridge.py
│   └── …
├── notebooks/
│   ├── step1_mock_profiling.ipynb
│   └── step1_openvla_profiling.ipynb
├── openvla/                     ← git clone (not in repo; see root README)
├── results/                     ← generated (gitignored)
└── models/                      ← generated (gitignored)
```

---

## Quick Start

### 1. Set up the environment (run once on HPC)

```bash
bash step1_setup_env.sh
conda activate openvla_g1
```

### 2. Run any script in mock mode (no GPU needed)

```bash
python3 step1_profile_openvla.py --n_trials 100 --mock
python3 step1_baseline_comparison.py --n_trials 50
python3 step2_esn_bridge.py                          # smoke test
python3 step2_train_esn.py --n_demos 200 --skip_sweep
python3 step3_full_evaluation.py --n_trials 100 --mock
python3 step3_ablation.py
python3 step4_paper_figures.py --mock
python3 step4_compile_results.py --mock
```

### 3. Run on real GPU (HPC cluster)

```bash
# Week 1–2: Profile real OpenVLA
python3 step1_profile_openvla.py --n_trials 100

# INT4 quantised (if VRAM < 13 GB)
python3 step1_profile_openvla.py --n_trials 100 --use_int4

# Week 7–8: Train ESN on demo data
python3 step2_train_esn.py --n_demos 200 --reservoir_size 1000

# Week 9–10: Full evaluation (loads trained ESN automatically)
python3 step3_full_evaluation.py --n_trials 100
```

---

## Research Timeline

| Week | Dates | Task | Deliverable |
|------|-------|------|-------------|
| 1–2 | May 19 – Jun 1 | Profile OpenVLA on G1 sim | `results/step1_profiling/` |
| 3–4 | Jun 2 – Jun 15 | 4-method baseline comparison | `results/step1_baselines/` |
| **Jun 3** | | **Advisor Update 1** | Profiling report + baseline table |
| 5–6 | Jun 16 – Jun 29 | ESN bridge architecture | `step2_esn_bridge.py` |
| 7–8 | Jun 30 – Jul 13 | Train ESN on demonstrations | `results/step2_training/` |
| 9–10 | Jul 14 – Jul 27 | Full evaluation (4 tasks) | `results/step3_evaluation/` |
| 11–12 | Jul 28 – Aug 10 | Ablation studies | `results/step3_ablation/` |
| **Aug 12** | | **Advisor Update 2** | Full eval table + ablations |
| 13–14 | Aug 11 – Aug 24 | Generate paper figures | `results/step4_paper_figures/` |
| 15–16 | Aug 25 – Sep 7 | LaTeX tables + submission | `results/step4_paper_tables/` |
| **~Sep 2026** | | **ICRA 2027 submission** | |

---

## Methods Compared

| Method | Control Hz | Mechanism |
|--------|-----------|-----------|
| Pure OpenVLA | ~3.2 Hz | Raw VLA output → joints (step-hold) |
| VLA + PID | ~3.2 Hz | PID error correction on top of VLA |
| VLA + LSTM | ~3.2 Hz | Learned temporal bridge (slow to train) |
| **VLA + ESN (Proposed)** | **~104 Hz** | ESN reservoir bridges VLA ticks |

---

## ESN Design

The Echo State Network bridge has three fixed matrices and one trained matrix:

| Matrix | Shape | Status |
|--------|-------|--------|
| W (reservoir) | N × N | Fixed random, scaled to ρ < 1 |
| W_in (input) | N × d | Fixed random |
| W_fb (feedback) | N × 29 | Fixed random (optional) |
| **W_out (readout)** | **29 × (N+d)** | **Trained via ridge regression** |

**Update equation at each 10 ms tick:**
```
x(t) = (1 - α) · x(t-1) + α · tanh(W·x(t-1) + W_in·u(t) + W_fb·y(t-1))
y(t) = W_out · [x(t); u(t)]
```

**Best configuration (from ablation):**

| Hyperparameter | Value |
|---------------|-------|
| Reservoir size N | 1000 |
| Spectral radius ρ | 0.95 |
| Sparsity | 0.90 |
| Leaking rate α | 1.0 |
| Washout steps | 50 |
| Ridge λ | 1e-4 |

---

## Key Results (Simulated — to be replaced with measured values)

**Pick-and-Place task (n=50 trials):**

| Method | Success | Control Hz | Latency |
|--------|---------|-----------|---------|
| Pure OpenVLA | 31.2% | 3.2 Hz | 382 ms |
| VLA + PID | 41.4% | 3.2 Hz | 385 ms |
| VLA + LSTM | 55.0% | 3.2 Hz | 422 ms |
| **VLA + ESN** | **84.2%** | **104 Hz** | **8.5 ms** |

ESN advantage is statistically significant (Wilcoxon, p < 0.001) across all 4 tasks.

---

## Paper Figures

| Figure | Description | Script |
|--------|-------------|--------|
| Fig 1 | System architecture | `step4_paper_figures.py` |
| Fig 2 | Latency gap motivation | `step4_paper_figures.py` |
| Fig 3 | 4-method baseline bars | `step4_paper_figures.py` |
| Fig 5 | Full evaluation (4 tasks) | `step4_paper_figures.py` |
| Fig 6 | Perturbation robustness | `step4_paper_figures.py` |
| Fig 7 | Ablation heat map (N × ρ) | `step4_paper_figures.py` |
| Fig 8 | Real-time control trace | `step4_paper_figures.py` |

All figures output to `results/step4_paper_figures/` as `.pdf` + `.png`.

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 7 GB (INT4) | 13 GB (FP16) |
| RAM | 32 GB | 64 GB |
| Storage | 20 GB | 50 GB |
| CUDA | 12.1+ | 12.1+ |

The ESN bridge itself runs on CPU in real time (W_out matmul ≈ 0.03 ms for N=1000).

---

## Citation

```bibtex
@inproceedings{ihimekpen2027vla,
  title     = {Closing the Frequency Gap: Real-Time Humanoid Robot Control
               via Vision-Language-Action Models and Echo State Networks},
  author    = {Ihimekpen, Osemudiamen Andrew},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2027},
  institution = {PVAMU CREDIT Center}
}
```
