# S2R Pipeline — Sim↔Real Robotics (ZMQ + ESN + G1 29-DoF)

Ultra low-latency Python pipeline for **deploying, monitoring, training, and data collection** on Unitree **G1 29-DoF** with VLA→ESN rate bridging.

> Path: `research/s2r/` (renamed from `r2s_pipeline`).  
> Wiring to MuJoCo Steps 1–4: **[INTEGRATION.md](INTEGRATION.md)** · ICRA plan: **[../ACTION_PLAN.md](../ACTION_PLAN.md)**

## What you get

| Piece | Role |
|---|---|
| **ZMQ bus** | msgpack + PUB/SUB (conflate), process-per-node scale-out |
| **YOLO detector** | Open-source image recognition (pen/table/person) |
| **Qwen2.5-VL** | Open-source scene VLM captions / grounding |
| **Qwen2.5 reasoner** | Open-source mission decisions (intent/risk/gate) |
| **Mission node** | `bring_pen` phase timeline for GUI + logs |
| **VLA node** | Sparse **~2 Hz** action tokens |
| **ESN engine** | Upsample to **50–100 Hz+** joint commands |
| **G1 bridge** | Unitree G1 Edu high-level loco hook (`unitree_sdk2py`) |
| **Data collector** | Full decision-process episode JSONL |
| **Monitor GUI** | Map, perception, mission, decisions, metrics |
| **Profiler** | Tesla V100 / Jetson Thor model timing |

```text
camera --> YOLO + Qwen2.5-VL --> perception
                |
                v
         Qwen reasoner + mission FSM --> decisions
                |
                v
         VLA (2Hz) --> ESN (100Hz+) --> sim_bridge / g1_bridge
                |                         |
                +---- data_collector <----+
                           |
                      Monitor GUI
```

## Quick start (mock models, no GPU required)

```bash
cd s2r
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

python -m s2r.cli deploy
# open http://127.0.0.1:8080
```

You should see mission phases evolve (`explore → locate → approach → grasp → …`) as the mock detector cycles a lab scene.

## Enable open-source models

```bash
pip install -e ".[models]"
```

Then set in `config/default.yaml`:

```yaml
models:
  detector: { mock: false, model_id: yolov8n.pt }
  vlm: { mock: false, model_id: Qwen/Qwen2.5-VL-3B-Instruct }
  reasoner: { mock: false, model_id: Qwen/Qwen2.5-3B-Instruct }
```

Or serve with vLLM and point `api_base` (recommended on Jetson Thor). Details: **[docs/MODELS.md](docs/MODELS.md)**

## Unitree G1 Edu

```bash
python -m s2r.cli deploy -c config/platforms/g1_edu.yaml
```

Full network/SDK/safety runbook: **[docs/UNITREE_G1_EDU.md](docs/UNITREE_G1_EDU.md)**

## Profiling (V100 / Thor)

```bash
# mock smoke
python -m s2r.cli profile --platform generic --backend mock

# on GPU hosts
python scripts/profile_models.py --platform v100 --backend real
python scripts/profile_models.py --platform thor --reasoner-api http://127.0.0.1:8000/v1 \
  --vlm-api http://127.0.0.1:9000/v1
```

Guide: **[docs/PROFILING_V100_THOR.md](docs/PROFILING_V100_THOR.md)**

## Useful commands

```bash
python -m s2r.cli run-node vision
python -m s2r.cli synth --seconds 20
python -m s2r.cli train
python -m s2r.cli deploy --only gui,camera,vision,mission,reasoning,vla,esn,state_publisher
```

## Notebooks & 12-task benchmark data

```bash
pip install -e ".[notebooks]"
jupyter lab notebooks/
```

| Notebook | Use |
|---|---|
| `notebooks/00_setup_and_data_tour.ipynb` | Data tour |
| `notebooks/01_train_esn.ipynb` | Train ESN from episodes |
| `notebooks/02_benchmark_12_tasks.ipynb` | Score my model vs Unitree VLA |
| `notebooks/03_unitree_vla_eval.ipynb` | UnifoLM-VLA eval harness |
| `notebooks/04_perception_reasoner_sandbox.ipynb` | YOLO/Qwen sandbox |
| `notebooks/05_inspect_data_distributions.ipynb` | Inspect distributions / robotics fitness |
| `notebooks/06_esn_ablation_compare.ipynb` | Compare with vs without ESN |

```bash
# Inspect whether your logs look usable for robotics learning
python -m s2r.cli inspect-data -s data/raw

# Research ablation: ESN vs no-ESN
python -m s2r.cli deploy -c config/ablation_with_esn.yaml
python -m s2r.cli deploy -c config/ablation_no_esn_zoh.yaml
python -m s2r.cli compare-ablation
```

Data folders: see [`data/README.md`](data/README.md) and [`data/benchmark/tasks.yaml`](data/benchmark/tasks.yaml) (12 Unitree G1 tasks).

## Docs index

- [Open-source models](docs/MODELS.md)
- [Unitree G1 Edu deployment](docs/UNITREE_G1_EDU.md)
- [V100 / Jetson Thor profiling](docs/PROFILING_V100_THOR.md)
- [Notebooks](notebooks/README.md)
- [Data layout](data/README.md)
- [ESN ablation study](docs/ESN_ABLATION.md)

## Tests

```bash
pytest -q
```

## Layout

```text
s2r/
  config/
  notebooks/                 # experiment notebooks (import s2r)
  data/
    benchmark/tasks/         # 12 Unitree G1 tasks
    unitree_vla/
    esn/
    raw/
  docs/
  scripts/profile_models.py
  src/s2r/
    experiments/   # benchmark + ESN data helpers for notebooks
    models/
    nodes/
    gui/
    training/
    deploy/
```
