# Tracked showcase videos

Small MP4s kept in git for the NeurIPS WRL / ICRA drafts (allowlisted in `.gitignore`).

| Clip | Path | What it shows |
|------|------|----------------|
| Oracle | `step4_mujoco_evaluation/table_wipe_ep160_oracle_esn.mp4` | Held-out ep 160: dataset-oracle tokens → ESN → MuJoCo wipe |
| Live | `step3_live_wipe/live_wipe_esn_live.mp4` | Live UnifoLM → ESN, `press_table` + motion clamps |

Do **not** commit other `*.mp4` under `results/` (dual-thread timing dumps, long loops, etc.).
