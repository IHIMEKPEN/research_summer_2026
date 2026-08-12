# Research Plan — VLA Frequency Gap + ESN Bridge (G1)

**Author:** Osemudiamen Andrew Ihimekpen · PVAMU CREDIT Center  
**Repo root:** `robotics/` (`main` @ origin; last pull includes Steps 2–4 held-out + Layer F live wipe)  
**Primary paper:** ICRA 2027 — deadline **15 Sep 2026 23:59 PST**  
**Hardware:** Unitree G1 Edu · UnifoLM-VLA · MuJoCo · DGX V100 · (next) Jetson Thor  

**Rule:** Prefer measured numbers over mock. Prefer held-out + task metrics over ep.0 RMSE.  
**Claim rule:** ESN is a rate/smoothness bridge — not Unitree’s WBC, and not a vision-learned press policy.

---

## P0 addendum — multi-task ESN library (DGX)

```bash
cd research
# Full suite (long): one checkpoint per UnifoLM G1 skill
python3 -m src.step2_esn_cuda_ridge --all_tasks --continue_on_error

# Or subset
python3 -m src.step2_esn_cuda_ridge --tasks wipe_table,clean_table,stack_block,fold_towel

# Deploy selection
python3 - <<'PY'
from src.step2_esn_cuda_ridge import load_esn_for_task
esn = load_esn_for_task("wipe_table")  # models/esn_cuda_ridge/
esn = load_esn_for_task("stack_block") # models/esn_cuda_ridge_stack_block/
PY
```

Artifacts: `models/esn_cuda_ridge_<task>/`, `models/esn_task_registry.json`, `results/step2_training/esn_multitask_summary.{csv,json}`.
Wipe remains the ICRA primary measured row until the suite CSV is fully filled.

---

## 0. What changed in the latest pull (reconcile prior brainstorm)

Your earlier “data ~75% contact vs ESN ~5%” confusion is now explained by **measured results in-repo**, not by “bad data” alone:

| Setting | Contact | What it actually tests |
|---|---|---|
| MuJoCo **dataset-oracle** ESN (demo tokens), held-out 40 eps | **74.5%** mean | Bridge + scene when the **plan is correct** |
| Live UnifoLM wipe **without** `press_table` | **~5%** (paper) | VLA lifts cloth; contact collapses |
| Live UnifoLM wipe **with** `press_table` + motion clamps | **100%** (ESN **and** ZOH) | Geometric wipe prior, **not** learned press-from-vision |

So the methodology still makes sense as a **2 Hz → 100 Hz bridge**. Live task success currently rides on **scene priors** (`proximity_synthetic` gripper + `press_table` + stance/EE clamps). Disclose that honestly.

---

## 1. What you can claim now (evidence-backed)

### A. Strong claims (tables filled; cite these)

| Claim | Evidence |
|---|---|
| UnifoLM-VLA-Base is ~57× too slow for 100 Hz G1 control | Step 1: **570.7 ms / 1.752 Hz** (PyTorch); Nsight **593.7 ms / 1.684 Hz** |
| Multi-ep CUDA ESN upsamples held intents with low held-out RMSE | Step 2: **2.78e-3 ± 3.44e-3 rad** (40 eps); train-ref ep.0 **1.10e-3** |
| ESN beats classical open-loop upsamplers on tracking | vs linear **1.55e-2** (~**5.6×**); ZOH **5.39e-2**; PID **7.14e-2** |
| Dataset-oracle MuJoCo wipe works when demo tokens drive ESN | Step 4 held-out: RMSE **3.00e-3**; grasp/task **100%**; contact **74.5%** |
| Live dual-process clears 100 Hz mean under UnifoLM | ESN **6.92 ms / 145 Hz** (tails >10 ms under GPU contention) |
| Live wipe can succeed under disclosed priors | Layer F: ESN 30 s → contact/task **100%** @ **124 Hz**; ZOH also **100%** with same priors |

### B. Soft / must-phrase-carefully claims

- ESN is a **rate / smoothness bridge** (anti-ZOH), conditioned on proprio + held VLA target.
- Architecture is compatible with G1 Edu **high-level** stack: VLA → ESN targets → Unitree mid-level → firmware.
- Live wipe proves the **stack can run** with interactive cloth; it does **not** prove ESN uniquely solves contact (ZOH also 100% with `press_table`).
- S2R ZMQ package exists for DIRT-guided transfer (deploy still remaining).

### C. Do **not** claim

| Do not claim | Why |
|---|---|
| Vision-learned table press / force wipe | `press_table` is a geometric prior |
| Real Unitree G1 Edu hardware success | No live Edu wipe numbers yet |
| ESN replaces balance / loco / WBC | Edu already owns this; freeze legs in live wipe |
| Multi-task / open-vocab humanoid skill | Single-task wipe corpus (~0.7 h) |
| ESN ≫ ZOH on live **task** success | Both pass under same priors; ESN wins on **tracking / jerk** |
| Live wipe without disclosing synthetic gripper | UnifoLM EE path lacks Dex1 gripper channel |

**One-sentence thesis:**  
*We quantify the VLA–humanoid frequency gap and show a cheap ESN bridge turns ~2 Hz intents into 100 Hz joint targets that beat ZOH/linear on held-out tracking, while live wipe success still requires an explicit contact prior and remains distinct from real-robot transfer.*

---

## 2. Pipeline status after pull

```text
Step 1  UnifoLM latency                 DONE (freeze numbers)
Step 2  CUDA ESN ridge (train 0–159)    DONE for wipe; **per-task suite** via `--all_tasks` (pending DGX fill)
Step 3a Offline ZOH/linear/PID          DONE (held-out 160–199)
Step 3b Dual-process live timing        DONE (esn/zoh/linear/pid)
Step 3c Live wipe success (Layer F)     DONE for ESN (+ ZOH report exists)
Step 4  MuJoCo oracle held-out          DONE (40 eps; contact 74.5%)
Step 4* Paper figures/tables dirs       PLACEHOLDER — still empty
Step 5  S2R → Jetson → G1 Edu           NEXT (primary remaining risk)
Ablations N/ρ                           PARTIAL / remaining
```

Key artifacts:

- `results/step2_training/esn_heldout_eval.csv` + `esn_heldout_summary.json`
- `results/step3_baselines/baseline_comparison_heldout.csv` + `*_summary.json`
- `results/step3_dual_thread/dual_thread_summary_all_live.csv`
- `results/step3_live_wipe/live_wipe_report_{esn,zoh}_live.json`
- `results/step4_mujoco_evaluation/mujoco_eval_summary_heldout.{csv,json}`
- Paper draft: `papers/icra2027/main.tex` (already updated to these numbers)

---

## 3. Forward tasks (priority order)

### Track P0 — ICRA submit (now → 15 Sep) — polish, don’t reinvent

Sim evidence is largely in. Remaining is honesty, packaging, and deploy credibility.

1. **Claim audit of `main.tex`:** keep frequency-gap + tracking superiority; keep live wipe; always pair with `press_table` / synthetic-gripper limitations (already partly written — harden abstract so skimmers don’t miss it).
2. **Regenerate `live_wipe_summary_live.csv`** to include **both** ESN and ZOH rows (ZOH JSON exists; summary CSV currently ESN-only).
3. **Fill `results/step4_paper_tables/` and `step4_paper_figures/`** from held-out + baseline + live summaries (replace placeholders).
4. **Optional contact ladder figure** (1 plot): oracle 74.5% vs live w/o press ~5% vs live w/ press 100% — resolves reviewer confusion you had yourself.
5. **N/ρ ablation table** if cheap (`N ∈ {500,1000,2000}`, `ρ ∈ {0.85,0.95,1.05}`) — nice-to-have, not blocker if time dies.
6. **ICRA video** (windows: Aug 5–Sep 9 and Sep 17–22): oracle wipe + live wipe clip + architecture slide.
7. PaperPlaza dry-run ≥48 h before **15 Sep**.

**Exit criteria:** 6+1 pages, all tables from held-out/live artifacts, limitations paragraph unmissable, video optional but preferred.

### Track P1 — Persona / real-robot credibility (parallel Aug–Dec)

Skills they hire for: robot learning, real-world eval, PyTorch, data/cloud, deploy NNs, manipulation.

1. **G1 Edu high-level deploy** (`config/platforms/g1_edu.yaml`): mock → stand → logged arm intent → metrics. Never bypass Unitree mid-level first.
2. **Same metric suite on robot logs:** grasp proximity, wipe path, contact proxy, EE error, Hz, failure tags.
3. **One BC baseline** on wipe demos (MLP/Transformer @ demo rate) vs VLA+ESN — shows robot-learning breadth.
4. **Cloud/data hygiene:** HF dataset card already exists; document train pipeline + checkpoint path for resume.
5. Keep software quality: schemas, tests, reproducible notebooks.

### Track P2 — Paper 2 (CoRL / RSS / NeurIPS) — after ICRA freeze (~Sep 1+)

**Working title:** *Hierarchical Multi-Rate Humanoid Manipulation: Sparse VLA Intents, Learned Upsamplers, and Mid-Level Control*

Must-add:

1. Real G1 Edu results + failure analysis.
2. Remove or learn the press prior (residual policy / impedance / force).
3. Upsampler bake-off: ESN vs RNN/MLP vs diffusion chunking.
4. Optional residual RL on frozen VLA intents (Persona “BC + RL for manipulation”).
5. Ablations + multi-seed live trials (current live wipe is n_trials=1).

---

## 4. Venue strategy

| Venue | Deadline | Fit | Action |
|---|---|---|---|
| **ICRA 2027** | **15 Sep 2026** | Best fit | **Submit Paper 1** |
| CoRL / NeurIPS 2026 | already passed | — | Skip |
| **RSS 2027** | ~Jan 2027 | Strong with real G1 | Paper 2 |
| **ICML 2027** | ~Jan 2027 | Only if ML-core upsampler theory/ablations | Stretch |
| **CoRL / NeurIPS 2027** | ~May 2027 | Robot learning | Paper 2 primary ML venues |

**Persona internship:** Fall 2026 starts now — target **Spring 2027** unless Fall still open. Packet: ICRA submission/preprint + G1 Edu plan + deploy/eval infra + contact-metric literacy.

---

## 5. Weekly schedule (post-pull)

| Week | Focus | Deliverable |
|---|---|---|
| **Aug 11–17** | Claim audit + paper tables/figures from existing CSVs | Non-placeholder `step4_paper_*`; dual-bridge live summary |
| **Aug 18–24** | G1 Edu dry-run (high-level, mock→limited) + ablation if time | Logged run; optional N/ρ table |
| **Aug 25–31** | Manuscript freeze + ICRA video | 6+1 pages locked |
| **Sep 1–7** | Polish + coauthor pass | Camera-ready draft |
| **Sep 8–14** | PaperPlaza | **Submit Sep 15** |
| **Sep 16–30** | Persona application | CV bullets §6 |
| **Oct–Dec** | Real wipe/manip + Paper 2 | CoRL/RSS track |
| **Jan 2027+** | RSS/ICML if ready; Spring Persona | Paper 2 |

---

## 6. Persona resume bullets (only measured work)

1. Profiled UnifoLM-VLA on humanoid clean-table; measured ~1.75 Hz vs 100 Hz control gap (~57×).
2. Trained CUDA ESN rate bridge on `G1_Dex1_Wipe_Table` (160/40 split); **~5.6×** lower held-out RMSE than linear upsampling.
3. Built MuJoCo dual-process + live wipe eval; reported grasp/contact/task metrics under live UnifoLM.
4. Separated oracle vs live vs prior-dependent success — shows deploy/eval maturity.
5. Shipping S2R ZMQ stack toward Unitree G1 Edu high-level control.

**30-second interview line:**  
*VLAs are semantically useful but temporally too slow for humanoids. I use a cheap ESN to turn ~2 Hz intents into 100 Hz targets that beat hold/linear on tracking, keep Unitree’s mid-level for balance, and evaluate with contact metrics — including when a press prior is required.*

---

## 7. Immediate runs (packaging only)

```text
# Prefer regenerating summaries/figures from existing reports — avoid re-training unless checkpoint drifts
notebooks/step3_live_wipe_success.ipynb   # ensure summary CSV has esn+zoh
notebooks/step4_mujoco_evaluation.ipynb   # already have heldout summary — export paper figs
src/step4_paper_figures.py / step4_compile_results.py
notebooks/step5_s2r_*.ipynb               # after manuscript freeze
```

---

## 8. Anti-goals

- Do not replace Unitree WBC with ESN.
- Do not hide `press_table` / synthetic gripper when quoting 100% live wipe.
- Do not claim ESN uniquely wins live task success vs ZOH under identical priors.
- Do not expand multi-task VLA training before ICRA submit.
- Do not leave placeholder `step4_paper_*` directories in the camera-ready bundle.
- Do not treat n_trials=1 live wipe as a large-N statistical claim.

---

## 9. Methodology verdict (for you, not for the abstract)

**Does VLA → ESN → joint targets make sense?** Yes — as a **frequency / smoothness bridge**, especially vs ZOH/linear on held-out tracking and oracle wipe when intents are good.

**Will it ship real work on G1 Edu alone?** Not by itself. On Edu you still need Unitree’s mid-level (balance) and, for contact-rich wipe, either better live intents or an explicit contact/impedance layer (today: `press_table`). That hierarchical story is exactly what Paper 2 / Persona interviews should emphasize.
