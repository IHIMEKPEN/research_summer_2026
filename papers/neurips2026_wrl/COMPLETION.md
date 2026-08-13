# NeurIPS 2026 WRL — Completion Tracker

**Venue:** 8th Robot Learning Workshop (WRL) @ NeurIPS 2026  
**Theme:** *Is Physical AI Going Zero-Shot?*  
**Site:** [http://www.robot-learning.ml/2026/](http://www.robot-learning.ml/2026/)  
**Submit:** OpenReview (≤6 pages, NeurIPS format, non-archival)  
**Deadline:** 26 Aug 2026, 23:59 AoE  
**Decisions:** 29 Sep 2026  
**Contact:** [wrl2026organizers@robot-learning.ml](mailto:wrl2026organizers@robot-learning.ml)  

Sister archival draft: `[../icra2027/](../icra2027/)`

---

## Fit (do not overclaim)


| Workshop topic                                     | Our paper                                         | Action                     |
| -------------------------------------------------- | ------------------------------------------------- | -------------------------- |
| VLA / generalist policies                          | Strong (UnifoLM + deployability)                  | Lead here                  |
| Real-world deployment / negative results           | Strong (press prior, contact collapse, GPU tails) | Lead here                  |
| Safety / failure analysis                          | Moderate                                          | Keep honest contact ladder |
| Zero-/few-shot adaptation                          | Weak                                              | Do **not** claim           |
| Cross-embodiment / world models / large-scale data | Weak                                              | Do **not** claim           |


**Pitch in one line:** *Physical AI deployment bottleneck — closing the VLA↔humanoid frequency gap with a cheap ESN bridge, plus honest live failure analysis.*

---



## Master checklist (speed path)

Use this if you need to cut scope before **26 Aug**. Check boxes as you go.

### A. Must-do before upload (blocking)

- [x] Send fit email to organizers (`EMAIL_TO_ORGANIZERS.md`) — **do this first**
- [ ] Create / verify OpenReview accounts for **both** co-authors (verification can take up to **2 weeks**)
- [ ] Replace draft `neurips.sty` with official NeurIPS author-kit style (current file is a close stand-in: Times, 5.5in text, page numbers only)
- [ ] Build PDF cleanly: `latexmk -pdf main.tex` (or `pdflatex` ×2 + `bibtex`)
- [ ] Confirm **≤6 pages** body (refs can follow NeurIPS workshop rules; verify on OpenReview page)
- [ ] Title + abstract match workshop framing (deployment / VLA systems / failure analysis — not zero-shot)
- [ ] Anonymous vs named: follow workshop OpenReview instructions
- [ ] Upload PDF + any required metadata on OpenReview before AoE deadline



### B. Content already in good shape (keep)

- [x] Frequency-gap profiling (PyTorch + Nsight)
- [x] ESN ridge training + held-out open-loop RMSE vs ZOH/linear/PID
- [x] Dual-process live timing (ESN clears 100 Hz mean)
- [x] MuJoCo dataset-oracle held-out wipe metrics
- [x] Live wipe with disclosed `press_table` prior + contact ladder
- [x] Declared dataset split and claim scope



### C. Nice-to-have before submit (do if time)

- [ ] One-paragraph workshop-specific intro sentence on “Physical AI going zero-shot?” → our answer is *not yet for contact-rich humanoid control without rate bridges / priors*
- [ ] Compact Table I only (already condensed vs ICRA) + 4–5 strongest figures (drop fig3/fig5 if over page limit)
- [ ] Add fig3 dataset split only if space remains
- [ ] Spell-check + citation pass against `references.bib`
- [ ] Explicit “non-archival; concurrent ICRA prep” note only if required by workshop policy



### D. Explicitly defer (speed cuts — do **not** block WRL)

- [ ] Full S2R / DIRT Jetson→G1 hardware deploy *(defer to ICRA)*
- [ ] Full $N$/$\rho$ ESN ablations *(defer)*
- [ ] Multi-task ESN suite RMSE table filled for all 11 tasks *(optional; wipe-only is enough for WRL)*
- [ ] Removing geometric press prior / learning press-from-vision *(out of scope for this deadline)*
- [ ] Cross-embodiment experiments *(out of scope)*

---



## What is left to complete



### 1. Process / venue (highest urgency)

1. **Organizer email** — ask if systems / deployment / VLA-rate-gap paper is in scope (draft ready).
2. **OpenReview** — both authors register + verify early.
3. **Official NeurIPS style** — swap in author-kit `.sty` (current file is a draft layout stand-in).
4. **Page budget** — trim figures if build exceeds 6 pages (priority keep: architecture, latency, baselines/rates, contact ladder).



### 2. Writing polish (1–2 evenings)

1. Tighten Discussion to workshop language: deployment bottleneck, negative results, failure analysis.
2. Ensure every claim is scoped to wipe corpus + disclosed priors.
3. Sync numbers with ICRA `main.tex` if any result files update (single source of truth: measured JSON under `research/results/`).



### 3. Optional experiments (only if email says “yes, and stronger with X”)


| Experiment | Effort | WRL value | Cut? |
|---|---|---|---|
| Quantify jerk / EE RMSE in one extra table row | Low | Strengthens ESN vs ZOH when task success ties | **Done** (2026-08-13) |
| Fill multi-task ESN CSV for 2–3 extra UnifoLM tasks | Medium | Shows bridge is not wipe-only | **Done** (zero-GPU; wipe/clean/stack) |
| Longer live wipe ($>$30 s) or $n{>}1$ seeds | Low–Med | Stronger live claim | **Defer** — do not spend GPU next |
| S2R smoke test on Jetson (even 10 s) | High | Real deploy | Defer unless already wired |

**Done: jerk + live joint-RMSE boost**
- Provenance: `research/results/step3_evaluation/jerk_joint_rmse_boost_summary.json`
- Paper Table 2 (offline): RMSE + jerk ratio — ZOH $50\times$, ESN $1.44\times$ vs linear
- Live wipe (tied 100% contact/task under `press_table`): ESN joint RMSE **0.514** vs ZOH **7.17** rad ($\sim$14$\times$)
- Do **not** claim EE RMSE — `right_ee_rmse_m` is identical (0.096) across methods in current logs

**Done: multi-task open-loop table (zero-GPU)**
- Provenance: `research/results/step3_evaluation/multitask_esn_table_summary.json`
- Paper Table 3: `wipe_table` / `clean_table` / `stack_block` (all n=40 held-out)
- Used `stack_block` instead of `fold_towel` on this laptop (local `fold_towel` is a stale 4+2-ep run); swap to DGX `fold_towel` 1.05e-3 after sync if preferred
- Claim scope: open-loop joint RMSE only — not live multi-task success

**DGX sync note:** NeurIPS draft lives on this Mac at `robotics/papers/neurips2026_wrl/`. Push/rsync to `/home/aihimekpen/research_summer_2026` before DGX edits.




### 4. Build / repo hygiene

```bash
cd robotics/papers/neurips2026_wrl
# after installing official neurips.sty:
latexmk -pdf main.tex
```

- [x] `main.pdf` builds without font/overfull disasters (6 pages after jerk/RMSE boost)
- [x] Figures path `figures/*.pdf` present
- [x] Bibliography resolves
- [x] Update `[../README.md](../README.md)` row stays accurate

---



## Speed plan (if email comes back “yes, submit”)

**Day 0 (today):** send email; OpenReview signup both authors.  
**Day 1:** official style + build PDF; cut to ≤6 pages.  
**Day 2:** writing polish + optional jerk/EE row.  
**Day 3–4 buffer:** only multi-task RMSE or longer live wipe if GPU free.  
**Day 5 (≤25 Aug):** final PDF + OpenReview upload (leave 1 day AoE margin).

**If email says “borderline / prefer zero-shot”:** do **not** force-fit; keep pushing ICRA 2027 archival track instead.

---



## File map


| Path                     | Role                                         |
| ------------------------ | -------------------------------------------- |
| `main.tex`               | NeurIPS workshop draft (reframed)            |
| `neurips.sty`            | **Draft** layout — replace with official kit |
| `references.bib`         | Shared cites (copied from ICRA)              |
| `figures/`               | Measured figure PDFs                         |
| `COMPLETION.md`          | This tracker                                 |
| `EMAIL_TO_ORGANIZERS.md` | Pre-submission fit email                     |


---



## Result sources (same campaign as ICRA)

- Profiling: `research/results/step1_profiling_unifolm_vla0/`
- ESN / wipe demos: `research/results/step2_training/`
- Dual-process: `research/results/step3_dual_thread/`
- Live wipe: `research/results/step3_live_wipe/`
- MuJoCo oracle: `research/results/step4_mujoco_evaluation/`

