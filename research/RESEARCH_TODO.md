# Research TODO — ICRA 2027 (VLA + ESN / G1)

Last updated: **12 Aug 2026**. Source of truth with `ACTION_PLAN.md`.

## Verdict

**Wipe Paper-1 experiments ≈ 85% complete.** Not fully done: multi-task suite still running, wipe ckpt needs restore after smoke, S2R/Edu not started, video/PaperPlaza pending.

---

## P0 — must finish before submit (15 Sep)

- [x] Step 1 UnifoLM latency (PyTorch n=100 + Nsight) frozen
- [x] Step 2 wipe multi-episode ESN (historical seed RMSE in `esn_task_registry_seed.json`)
- [x] Step 3a offline ZOH/linear/PID held-out baselines
- [x] Step 3b live dual-process timing (esn/zoh/linear/pid)
- [x] Step 3c live wipe Layer F (ESN + ZOH JSONs; press_table disclosed)
- [x] Step 4 MuJoCo dataset-oracle held-out (40 eps)
- [x] **Paper figures (measured)** — `python3 -m src.step4_paper_figures` → `papers/icra2027/figures/fig1…8_*`
- [x] Wire figures into `papers/icra2027/main.tex` / rebuild `main.pdf` (6+1)
- [ ] **Finish DGX `--all_tasks --continue_on_error`** (PID/log under `/raid/.../esn_all_tasks_*.log`)
- [ ] Update Table II from final `esn_multitask_summary.csv` (measured rows only)
- [ ] **Restore wipe full train** (smoke overwrote 4-ep ckpt):  
      `python3 -m src.step2_esn_cuda_ridge --task wipe_table --train_episodes train --heldout_episodes heldout`
- [ ] Dual-row `live_wipe_summary_live.csv` (ESN + ZOH)
- [ ] Claim audit: abstract/skimmer cannot miss press_table + synthetic gripper
- [ ] ICRA video (oracle + live + architecture)
- [ ] PaperPlaza dry-run ≥48 h before deadline

## P1 — credibility / Persona (parallel)

- [ ] G1 Edu high-level dry-run (`config/platforms/g1_edu.yaml`): mock → stand → logged intent
- [ ] Same metric suite on robot logs
- [ ] Optional BC baseline vs VLA+ESN on wipe demos
- [ ] Optional N/ρ ablation table

## P2 — after ICRA freeze

- [ ] S2R ZMQ → Jetson → G1 Edu (DIRT)
- [ ] Learn/remove press prior (Paper 2)
- [ ] Upsampler bake-off (ESN vs RNN/MLP/chunking)
- [ ] Multi-seed live trials (current live wipe n=1)

## Anti-goals (do not)

- Do not replace Unitree WBC with ESN
- Do not hide `press_table` / synthetic gripper when quoting 100% live wipe
- Do not claim ESN uniquely wins live **task** success vs ZOH under identical priors
- Do not invent figure numbers / mock `SUCCESS_PRIORS`
- Do not leave smoke (4-ep) wipe checkpoint as the paper artifact

## Commands cheat-sheet

```bash
cd /home/aihimekpen/research_summer_2026/research
source /raid/data/aihimekpen/venvs/research_summer_2026/bin/activate
export HF_HOME=/raid/data/aihimekpen/hf_cache HUGGINGFACE_HUB_CACHE=/raid/data/aihimekpen/hf_cache/hub

# Suite status
tail -f /raid/data/aihimekpen/research_logs/esn_all_tasks_*.log
cat results/step2_training/esn_multitask_summary.csv

# Figures + paper
python3 -m src.step4_paper_figures
cd ../papers/icra2027 && ../../tools/tectonic main.tex
```
