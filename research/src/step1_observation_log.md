# Step 1 — Observation Log
**Research Plan:** VLA + ESN for Real-Time Humanoid Robot Control  
**Author:** Osemudiamen Andrew Ihimekpen | PVAMU CREDIT Center  
**Period:** Weeks 1–4 (May 19 – June 15, 2026)  
**Deliverable:** Inference Log · Profiling Report · 4-Method Baseline Table

---

## Week 1–2 (May 19 – Jun 1): Environment Setup & Latency Profiling

### Environment Setup Checklist

| Step | Status | Notes |
|------|--------|-------|
| Conda env `openvla_g1` created | ☐ | Python 3.10 |
| PyTorch 2.2 + CUDA installed | ☐ | Check with `torch.cuda.is_available()` |
| OpenVLA (`openvla/openvla-7b`) loaded | ☐ | HuggingFace; ~13 GB VRAM |
| MuJoCo 3.x installed | ☐ | `pip install mujoco==3.1.6` |
| Unitree G1 MJCF obtained | ☐ | `unitree_mujoco` or `mujoco_menagerie` |
| G1 sim renders at 100 Hz | ☐ | Verify with `mujoco.viewer` |
| `step1_profile_openvla.py` runs without error | ☐ | `--mock` flag if GPU unavailable |

---

### GPU Environment Report

```
GPU model       : ___________________________
VRAM total      : _____ GB
CUDA version    : ___________________________
Driver version  : ___________________________
PyTorch build   : ___________________________
OpenVLA loaded  : Yes / No  (INT4: Yes / No)
```

---

### Latency Profiling Results (fill after running `step1_profile_openvla.py`)

| Metric | Measured | Research Plan Expectation |
|--------|----------|--------------------------|
| Mean latency (ms) | | 300–500 ms |
| Std dev (ms) | | |
| Median (ms) | | |
| P95 (ms) | | |
| P99 (ms) | | |
| Effective control rate (Hz) | | 2–5 Hz |
| GPU memory used (GB) | | ~13 GB (FP16) / ~7 GB (INT4) |
| Frequency gap vs G1 target | | ~20–50× |

**Conclusion (circle):**  Confirmed 2–5 Hz / Faster than expected / Slower than expected  

---

### Failure Modes Observed

| Condition | Failure Type | Frequency | Example Action Values |
|-----------|--------------|-----------|-----------------------|
| Normal image | | | |
| Overexposed image | | | |
| Dark / underlit image | | | |
| Partial occlusion | | | |
| Out-of-distribution scene | | | |
| Rapid scene change | | | |

**Key failure pattern summary:**  
> _[Write 2–3 sentences here after profiling. E.g.: "Latency spikes to >800ms occur when the language instruction contains unusual object names. Dark images cause NaN action outputs in ~12% of trials. The 2–5 Hz control rate is confirmed, creating a 20–50× gap with the G1's 100 Hz requirement."]_

---

### Output Files Checklist

| File | Path | Status |
|------|------|--------|
| Inference log (JSON) | `results/step1_profiling/inference_log.json` | ☐ |
| Profiling report (JSON) | `results/step1_profiling/profiling_report.json` | ☐ |
| Profiling report (TXT) | `results/step1_profiling/profiling_summary.txt` | ☐ |
| Profiling figure (PDF) | `results/step1_profiling/openvla_profiling_report.pdf` | ☐ |

---

## Week 3–4 (Jun 2 – Jun 15): Baseline Experiments

### Tasks

| Task | Description | Success Criterion |
|------|-------------|-------------------|
| **Pick-and-Place** | Pick 5cm red cube → place in 15cm bin | EE error < 5 cm |
| **Corridor Navigation** | 5m corridor, 2 dynamic obstacles, reach goal | No collision + arrive within 10 cm |

### 4-Method Setup Checklist

| Method | Implementation | Status |
|--------|---------------|--------|
| **Pure OpenVLA** | Raw model output → G1 joint commands | ☐ |
| **VLA + PID** | PID error-correction layer on top of VLA output | ☐ |
| **VLA + LSTM** | Trained LSTM bridge at ~3 Hz (Week 7–8 full version) | ☐ |
| **VLA + ESN (Proposed)** | ESN bridge outputting 100+ Hz (preliminary W_out) | ☐ |

---

### Baseline Results Table (fill after running `step1_baseline_comparison.py`)

> Run: `python step1_baseline_comparison.py --n_trials 50`

**Pick-and-Place (n=50 trials each)**

| Method | Success (%) | Control Hz | EE Error (m) | Collision (%) | Recovery (s) | Wilcoxon p |
|--------|-------------|------------|--------------|---------------|--------------|------------|
| Pure OpenVLA | | | | | | — |
| VLA + PID | | | | | | |
| VLA + LSTM | | | | | | |
| **VLA + ESN ★** | | | | | | |

**Corridor Navigation (n=50 trials each)**

| Method | Success (%) | Control Hz | EE Error (m) | Collision (%) | Recovery (s) | Wilcoxon p |
|--------|-------------|------------|--------------|---------------|--------------|------------|
| Pure OpenVLA | | | | | | — |
| VLA + PID | | | | | | |
| VLA + LSTM | | | | | | |
| **VLA + ESN ★** | | | | | | |

★ = Proposed method

---

### Key Observations for Advisor Update 1 (June 3)

**1. OpenVLA baseline failure modes:**
> _[Fill in after Week 1–2 experiments]_

**2. Control frequency gap quantified:**
> _[E.g. "OpenVLA produces commands at 3.2 ± 0.4 Hz, confirming the 31× gap with the G1's 100 Hz requirement."]_

**3. PID improvement over pure VLA:**
> _[E.g. "VLA+PID improves success rate from 28% to 42% but does not address the frequency gap."]_

**4. ESN preliminary advantage:**
> _[E.g. "Even a preliminary ESN bridge (untrained W_out) raises control rate to 103 Hz."]_

**5. Motivation confirmed:**
> _[Summarize why the ESN bridge is the right solution based on observed data]_

---

### Output Files Checklist

| File | Path | Status |
|------|------|--------|
| Baseline table (CSV) | `results/step1_baselines/baseline_table.csv` | ☐ |
| Baseline table (LaTeX) | `results/step1_baselines/baseline_table.tex` | ☐ |
| Comparison figures (PDF) | `results/step1_baselines/baseline_comparison.pdf` | ☐ |
| Episode records (CSV) | `results/step1_baselines/episode_records.csv` | ☐ |

---

## Advisor Update 1 — June 3, 2026

**Send to supervisor:**
- [ ] Profiling report PDF (from Week 1–2)
- [ ] 4-method baseline table (CSV + LaTeX)
- [ ] 2–3 page summary with key findings
- [ ] Updated timeline (any adjustments needed)

**Key message:**  
> "OpenVLA confirms a ~[X]× frequency gap at [Y] Hz against G1's 100 Hz requirement. Failure modes under [conditions] are documented. The preliminary ESN bridge closes this gap in simulation, motivating Objective 02."

---

## Notes & Issues

| Date | Issue | Resolution | Status |
|------|-------|------------|--------|
| | | | |

---

*Log started: May 19, 2026 | Last updated: ___________*
