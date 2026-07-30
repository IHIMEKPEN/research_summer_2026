"""Import this first in every notebook to put `s2r` on sys.path."""

from __future__ import annotations

import sys
from pathlib import Path


def setup() -> Path:
    here = Path.cwd().resolve()
    # Allow launching from repo root or notebooks/
    candidates = [here, here.parent, *here.parents]
    root = None
    for c in candidates:
        if (c / "src" / "s2r").exists() and (c / "config" / "default.yaml").exists():
            root = c
            break
    if root is None:
        raise RuntimeError("Could not locate s2r root (expected src/s2r + config/default.yaml)")
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return root
