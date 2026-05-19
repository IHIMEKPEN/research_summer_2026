"""Project root and artifact paths (cwd-independent)."""

from pathlib import Path

# research/ — parent of src/
RESEARCH_ROOT = Path(__file__).resolve().parent.parent


def results_path(*subpath: str) -> Path:
    """e.g. results_path('step1_profiling') -> research/results/step1_profiling"""
    p = RESEARCH_ROOT / "results" / Path(*subpath)
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_path(*subpath: str) -> Path:
    """e.g. models_path('esn_bridge') -> research/models/esn_bridge"""
    p = RESEARCH_ROOT / "models" / Path(*subpath)
    p.mkdir(parents=True, exist_ok=True)
    return p


def result_file(*subpath: str) -> Path:
    """Path under research/results/ without creating directories."""
    return RESEARCH_ROOT / "results" / Path(*subpath)
