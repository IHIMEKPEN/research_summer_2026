# research_summer_2026

**Author:** Osemudiamen Andrew Ihimekpen · **PVAMU CREDIT Center** · ICRA 2027  
**Deadline:** 15 Sep 2026 — [`research/ACTION_PLAN.md`](research/ACTION_PLAN.md)  
**Paper:** [`papers/icra2027/`](papers/icra2027/)

UnifoLM-VLA ≈ 2 Hz; Unitree **G1 (29-DoF)** needs 100 Hz. One research tree: CUDA ESN bridge, MuJoCo wipe, S2R deploy.

## Layout

```text
research_summer_2026/
├── research/                 # All experiment code (Steps 1–5)
│   ├── ACTION_PLAN.md
│   ├── src/                  # step1–4 + src/s2r/ (Step 5)
│   ├── config/ · notebooks/ · docs/ · tests/
│   └── results/ · models/    # gitignored measured artifacts
├── papers/icra2027/          # IEEE draft
├── unifolm-vla/              # Upstream submodule
└── .cursor/rules/            # Hardware + ICRA rules
```

## Quick start

```bash
cd research
pip install -e ".[dev,train]"

python3 -m src.step3_dual_thread_mujoco --mock --duration_s 5
python3 -m src.step3_control_baselines --method zoh --episode 0
python3 -m s2r.cli deploy -c config/default.yaml
pytest -q
```

GPU UnifoLM: `research/requirements-unifolm-gpu.txt`. Details: [`research/README.md`](research/README.md).
