"""Project root and artifact paths (cwd-independent).

Writes under ``research/results`` and ``research/models`` (in-repo, visible to
``@results``). When ``/raid`` is writable, also mirrors into
``/raid/data/aihimekpen/research_summer_2026/{results,models}`` for durability
on the DGX root disk.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

# research/ — parent of src/
RESEARCH_ROOT = Path(__file__).resolve().parent.parent

_RAID_PROJECT = Path("/raid/data/aihimekpen/research_summer_2026")


def _mirror_to_raid(local: Path) -> None:
    """Best-effort copy of a file/dir into the raid mirror (never raises)."""
    if os.environ.get("RESEARCH_DISABLE_RAID_MIRROR", "").strip() in {"1", "true", "yes"}:
        return
    if not Path("/raid").is_dir():
        return
    try:
        rel = local.resolve().relative_to(RESEARCH_ROOT.resolve())
    except ValueError:
        return
    if rel.parts[:1] not in {("results",), ("models",)} and rel.parts[0] not in {"results", "models"}:
        return
    dest = _RAID_PROJECT / rel
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if local.is_dir():
            shutil.copytree(local, dest, dirs_exist_ok=True)
        elif local.is_file():
            shutil.copy2(local, dest)
    except OSError:
        pass


def results_root() -> Path:
    override = os.environ.get("RESEARCH_RESULTS_ROOT")
    if override:
        p = Path(override)
        p.mkdir(parents=True, exist_ok=True)
        return p
    p = RESEARCH_ROOT / "results"
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_root() -> Path:
    override = os.environ.get("RESEARCH_MODELS_ROOT")
    if override:
        p = Path(override)
        p.mkdir(parents=True, exist_ok=True)
        return p
    p = RESEARCH_ROOT / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def results_path(*subpath: str) -> Path:
    """e.g. results_path('step1_profiling_unifolm_vla0') -> research/results/…"""
    p = results_root() / Path(*subpath)
    p.mkdir(parents=True, exist_ok=True)
    _mirror_to_raid(p)
    return p


def experiment_results_path(vla_id: str, *subpath: str) -> Path:
    """
    Namespaced results for a chosen VLA backend (never mixes stacks).

    Example: experiment_results_path('unifolm', 'step3_live_wipe')
      → research/results/experiments/unifolm/step3_live_wipe/
    Legacy UnifoLM paths (``step1_profiling_unifolm_vla0``, …) stay unchanged.
    """
    slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(vla_id).strip().lower())
    if not slug:
        raise ValueError("vla_id must be non-empty")
    return results_path("experiments", slug, *subpath)


def models_path(*subpath: str) -> Path:
    """e.g. models_path('esn_cuda_ridge') -> research/models/esn_cuda_ridge"""
    p = models_root() / Path(*subpath)
    p.mkdir(parents=True, exist_ok=True)
    _mirror_to_raid(p)
    return p


def result_file(*subpath: str) -> Path:
    """Path under research/results/ without creating leaf directories."""
    return results_root() / Path(*subpath)


def mirror_result(path: Path) -> Path:
    """Copy a saved artifact to the raid mirror; return the original path."""
    _mirror_to_raid(Path(path))
    return Path(path)
