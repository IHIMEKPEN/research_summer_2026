# Step 1 — UnifoLM-VLA profiling results

Each notebook / CLI run creates a timestamped folder:

```text
pytorch_profiler_YYYYMMDD_HHMMSS/
  run_config.json
  profiling_report.json
  profiling_summary.txt
  inference_log.json
  vla_action_profiler_ops.{txt,json}
  unifolm_vla0_profiling_report.{png,pdf}
  async_runtime.json          # optional async demo cell
```

`latest` → most recent run.

Also mirrored to `/raid/data/aihimekpen/research_summer_2026/results/`.
