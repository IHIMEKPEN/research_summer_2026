# VLA + ESN for Real-Time Humanoid Robot Control

**Author:** Osemudiamen Andrew Ihimekpen · **PVAMU CREDIT Center** · ICRA 2027  
**Deadline:** 15 Sep 2026 — [`research/ACTION_PLAN.md`](research/ACTION_PLAN.md)  
**Paper:** [`papers/icra2027/`](papers/icra2027/)

UnifoLM-VLA ≈ 2 Hz; Unitree **G1 (29-DoF)** needs 100 Hz. CUDA ESN bridge + MuJoCo wipe + S2R deploy.

## Layout

```text
research_summer_2026/
├── research/                 # All experiment code
│   ├── ACTION_PLAN.md
│   ├── src/                  # Steps 1–4 (python3 -m src.*)
│   ├── s2r/                  # Sim-to-real ZMQ package (29-DoF)
│   ├── notebooks/
│   ├── results/ · models/    # gitignored measured artifacts
│   └── CLAUDE.md             # AI / contributor rules
├── papers/icra2027/          # IEEE draft
├── unifolm-vla/              # Upstream submodule
└── .cursor/rules/            # Hardware + ICRA rules
```

## Quick start

```bash
cd research
pip install -e . && pip install -r requirements.txt

python3 -m src.step3_dual_thread_mujoco --mock --duration_s 5
python3 -m src.step3_control_baselines --method zoh --episode 0

cd s2r && pip install -e ".[dev,train]" && python3 -m pytest -q
python3 -m s2r.cli deploy
```

GPU UnifoLM: `requirements-unifolm-gpu.txt`. Details: [`research/README.md`](research/README.md).
