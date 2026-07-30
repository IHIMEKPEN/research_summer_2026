"""Import this first in every notebook to put research/ and src/ on sys.path."""

from __future__ import annotations

import sys
from pathlib import Path


def setup() -> Path:
    here = Path.cwd().resolve()
    # Allow launching from repo root, research/, or notebooks/
    candidates = [here, here.parent, *here.parents]
    root = None
    for c in candidates:
        if (c / "src" / "s2r").exists() and (c / "config" / "default.yaml").exists():
            root = c
            break
    if root is None:
        raise RuntimeError(
            "Could not locate research/ (expected src/s2r + config/default.yaml)"
        )
    for p in (str(root), str(root / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)
    return root
