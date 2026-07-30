#!/usr/bin/env python3
"""Pin every research/notebooks/*.ipynb to the project UnifoLM kernel."""

from __future__ import annotations

import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NB_DIR = REPO / "research" / "notebooks"
VENV_PY = os.environ.get(
    "RESEARCH_VENV_PYTHON",
    "/raid/data/aihimekpen/venvs/research_summer_2026/bin/python",
)
# Fall back to in-repo symlink if raid path missing
if not Path(VENV_PY).exists():
    alt = REPO / "research" / ".venv" / "bin" / "python"
    if alt.exists():
        VENV_PY = str(alt.resolve())

KERNELSPEC = {
    "display_name": "Research Summer 2026 (UnifoLM)",
    "language": "python",
    "name": "python3",
}


def pin(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    md = nb.setdefault("metadata", {})
    md["kernelspec"] = dict(KERNELSPEC)
    md["language_info"] = {"name": "python", "version": "3.10.12"}
    md["interpreter"] = {"path": VENV_PY}
    md["vscode"] = {"interpreter": {"path": VENV_PY}}
    path.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    paths = sorted(NB_DIR.glob("*.ipynb"))
    if not paths:
        raise SystemExit(f"No notebooks under {NB_DIR}")
    for p in paths:
        pin(p)
        print(f"pinned {p.name}")
    print(f"OK: {len(paths)} notebooks → {KERNELSPEC['display_name']}")
    print(f"    python = {VENV_PY}")


if __name__ == "__main__":
    main()
