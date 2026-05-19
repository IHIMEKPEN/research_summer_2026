# CLAUDE.md — Rules for AI Tools Working in This Repository

## Project Identity

- **Author:** Osemudiamen Andrew Ihimekpen
- **Institution:** PVAMU CREDIT Center
- **Project:** VLA + ESN for Real-Time Humanoid Robot Control
- **Target:** ICRA 2027 submission (~September 2026)
- **Working directory:** `/Users/andrew/Desktop/competitions/research_summer_2026/research/`

---

## Execution Rules

### Always use `python3`, never `python`
This machine uses `python3`. All test and run commands must use `python3`.

### Always test with `--mock` first
Before running any script on real hardware or with the real model:
```bash
python3 <script>.py --mock
```
Never invoke a script that downloads the OpenVLA model (13 GB) without explicit user instruction.

### Never modify mock simulation priors without user confirmation
The success-rate priors in `G1TaskEnv.SUCCESS_PRIORS` and `G1FullEvalEnv` encode the research hypothesis. Changing them changes the paper's claimed results. Only modify these when the user explicitly provides measured data from the real simulator.

### Never delete or overwrite result files
All files under `results/` are research outputs. If a script would overwrite them, flag it to the user first.

### Never commit generated outputs
The `results/`, `models/`, and `__pycache__/` directories contain generated data, not source code.

---

## Code Conventions

### Naming
- Script files: `step{N}_{description}.py` — never rename existing step files
- Result directories: `results/step{N}_{description}/`
- Model checkpoints: `models/{name}/`

### Script structure
Every script follows this pattern:
1. Module docstring with purpose, deliverable week, author, usage
2. Constants block (ALL_CAPS)
3. Data structures (`@dataclass`)
4. Core logic classes/functions
5. `main()` with `argparse`
6. `if __name__ == "__main__": main()`

### Mock vs real separation
- Mock classes are prefixed with `Mock` or use a `--mock` flag
- Mock inference delays use `time.sleep()` to simulate realistic latency
- Real model calls are always inside `try/except` with graceful fallback to mock

### No silent failures
If a real model load fails, always log a `WARNING` stating that mock inference is being used. Never silently proceed without the user knowing.

### Figures
- All figures saved as both `.pdf` (for LaTeX) and `.png` (for previewing)
- Use `plt.close()` after saving — never leave figures open
- DPI: 150 for screen, 300 for `savefig`
- Color scheme: red=pure_openvla, orange=vla_pid, blue=vla_lstm, green=vla_esn (proposed)

### LaTeX tables
- Never use `df.to_latex()` — it requires jinja2 which is not installed
- Always write LaTeX tables manually with `\toprule`, `\midrule`, `\bottomrule`
- Proposed method rows use `\textbf{}` formatting

---

## Architecture Decisions (do not change without user approval)

| Decision | Rationale |
|----------|-----------|
| Reservoir is fixed (not trained) | Echo state property; only W_out is learned |
| W_out fit via ridge regression | Closed-form, no backprop, seconds not hours |
| Input dim = OPENVLA_HIDDEN_DIM (4096) | Full hidden state; user may add projection layer later |
| Spectral radius ρ < 1 | Required for echo state property (stability guarantee) |
| Zero-order hold for upsampling | Default; linear interpolation available via flag |
| `scipy.stats.wilcoxon` for significance | Non-parametric, appropriate for bounded success rates |
| G1_DOF = 29 | Unitree G1 has 29 controllable joints |
| TARGET_HZ = 100 | G1 low-level controller runs at 100 Hz |

---

## Step Timeline (do not change deliverable dates)

| Step | Weeks | Dates | Deliverable |
|------|-------|-------|-------------|
| 1a | 1–2 | May 19 – Jun 1 | Inference log + profiling report |
| 1b | 3–4 | Jun 2 – Jun 15 | 4-method baseline table |
| 2a | 5–6 | Jun 16 – Jun 29 | ESN architecture + unit tests |
| 2b | 7–8 | Jun 30 – Jul 13 | Trained model + training report |
| 3a | 9–10 | Jul 14 – Jul 27 | Full evaluation table |
| 3b | 11–12 | Jul 28 – Aug 10 | Ablation studies |
| 4a | 13–14 | Aug 11 – Aug 24 | All paper figures |
| 4b | 15–16 | Aug 25 – Sep 7 | LaTeX tables + submission package |

**Advisor Update 1:** June 3, 2026  
**Advisor Update 2:** August 12, 2026

---

## Dependencies

Installed in the `openvla_g1` conda environment (see `step1_setup_env.sh`).
Scripts also run in the system Python 3.11 environment with:
```
numpy, scipy, matplotlib, pandas, tqdm, torch (optional), transformers (optional)
```
`jinja2` is NOT installed — do not use any pandas `.to_latex()` or `.style` calls.

---

## Outputs Directory Map

```
results/
  step1_profiling/       ← inference_log.json, profiling_report.json, *.pdf
  step1_baselines/       ← baseline_table.csv, baseline_table.tex, *.pdf
  step2_training/        ← training_report.json, hyperparam_sweep.csv, *.pdf
  step3_evaluation/      ← full_evaluation_table.csv, all_episodes.csv, *.pdf
  step3_ablation/        ← ablation_*.csv, ablation_studies.pdf
  step4_paper_figures/   ← fig1_*.pdf … fig8_*.pdf (all paper figures)
  step4_paper_tables/    ← all_tables.tex, paper_results_summary.json

models/
  esn_bridge/            ← W_out.npy, config.json
```
