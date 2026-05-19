# VLA + ESN for Real-Time Humanoid Robot Control

**Author:** Osemudiamen Andrew Ihimekpen · **PVAMU CREDIT Center** · Summer 2026 · ICRA 2027 target

OpenVLA runs at ~3 Hz; the Unitree G1 expects 100 Hz. This repo profiles that gap and implements an Echo State Network bridge for high-rate control.

## Quick start

All code lives under [`research/`](research/README.md).

```bash
cd research
pip install -e .
pip install numpy scipy matplotlib pandas tqdm torch transformers accelerate timm sentencepiece einops pillow

# OpenVLA upstream (not vendored in git)
git clone https://github.com/openvla/openvla.git openvla

# Mock profiling (Mac / no GPU)
python -m src.step1_profile_openvla --mock --n_trials 20

# Real profiling (NVIDIA GPU, ~13 GB VRAM)
python -m src.step1_profile_openvla --n_trials 100 --no-mock-fallback
```

Notebooks: [`research/notebooks/`](research/notebooks/README.md)

## Layout

```
research_summer_2026/
├── research/          # Python package (src/), notebooks, pyproject.toml
├── papers/            # LaTeX paper draft
└── others/            # figures / misc
```

## License

Research code: see repository license. OpenVLA is [Apache-2.0](https://github.com/openvla/openvla) (clone separately).
