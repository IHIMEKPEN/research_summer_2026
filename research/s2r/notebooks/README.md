# Experiment notebooks

All notebooks import the **existing** `s2r` package (no duplicated logic).

```bash
cd s2r
source .venv/bin/activate
pip install -e ".[notebooks]"
jupyter lab notebooks/
# or: jupyter notebook notebooks/
```

| Notebook | Purpose |
|---|---|
| `00_setup_and_data_tour.ipynb` | Path bootstrap + 12-task catalog tour |
| `01_train_esn.ipynb` | Build splits from `data/raw`, train/save ESN |
| `02_benchmark_12_tasks.ipynb` | Score my model vs Unitree VLA on 12 tasks |
| `03_unitree_vla_eval.ipynb` | UnifoLM-VLA adapter harness + prediction dumps |
| `04_perception_reasoner_sandbox.ipynb` | YOLO/Qwen quick experiments on task instructions |
| `05_inspect_data_distributions.ipynb` | Distributions + robotics fitness checklist |
| `06_esn_ablation_compare.ipynb` | Compare pipeline with vs without ESN |

## Import pattern (used by every notebook)

```python
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd() / "_lib"))
from bootstrap import setup
ROOT = setup()

from s2r.experiments.benchmark import load_tasks
from s2r.nodes.esn_engine import EchoStateNetwork
```

Helpers live in:

- `notebooks/_lib/bootstrap.py`
- `src/s2r/experiments/` (`paths`, `benchmark`, `esn_data`)
