#!/usr/bin/env bash
# Create/repair research/.venv (on /raid) and register the Jupyter kernel.
# Run from anywhere:
#   bash research/scripts/ensure_jupyter_kernel.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROJ="${REPO_ROOT}/research"
VENV_REAL="${VENV_REAL:-/raid/data/aihimekpen/venvs/research_summer_2026}"
KERNEL_NAME="${KERNEL_NAME:-research_summer_2026}"
DISPLAY_NAME="${DISPLAY_NAME:-Research Summer 2026 (UnifoLM)}"

mkdir -p "$(dirname "$VENV_REAL")"

if [[ ! -x "${VENV_REAL}/bin/python" ]]; then
  if command -v virtualenv >/dev/null 2>&1; then
    virtualenv -p python3 "$VENV_REAL"
  else
    python3 -m pip install --user -U virtualenv
    "$HOME/.local/bin/virtualenv" -p python3 "$VENV_REAL"
  fi
fi

ln -sfn "$VENV_REAL" "${PROJ}/.venv"

export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/raid/data/aihimekpen/pip_cache}"
mkdir -p "$PIP_CACHE_DIR"

"${PROJ}/.venv/bin/pip" install -U pip wheel
"${PROJ}/.venv/bin/pip" install -r "${PROJ}/requirements.txt"
"${PROJ}/.venv/bin/pip" install ipykernel

"${PROJ}/.venv/bin/python" -m ipykernel install \
  --user \
  --name="$KERNEL_NAME" \
  --display-name="$DISPLAY_NAME"

# Also bind the default `python3` kernelspec to this venv so every notebook
# that asks for name=python3 lands on the same interpreter.
"${PROJ}/.venv/bin/python" -m ipykernel install \
  --user \
  --name=python3 \
  --display-name="$DISPLAY_NAME"

# Force absolute argv in both kernelspecs (avoid bare `python` on PATH).
python3 - <<PY
import json
from pathlib import Path
venv_py = "${VENV_REAL}/bin/python"
display = "${DISPLAY_NAME}"
for name in ("python3", "${KERNEL_NAME}"):
    p = Path.home() / ".local/share/jupyter/kernels" / name / "kernel.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "argv": [venv_py, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": display,
        "language": "python",
        "metadata": {"debugger": True, "vscode": {"interpreter": {"path": venv_py}}},
    }, indent=1) + "\n")
    print("kernelspec", p)
PY

# Pin all notebooks to this kernel
"${PROJ}/.venv/bin/python" "${PROJ}/scripts/pin_notebook_kernels.py"

# Help Cursor / VS Code pick this env from the workspace root
REPO_ROOT="$(cd "${PROJ}/.." && pwd)"
ln -sfn "${PROJ}/.venv" "${REPO_ROOT}/.venv"
VSCODE_DIR="${REPO_ROOT}/.vscode"
mkdir -p "$VSCODE_DIR"
SETTINGS="${VSCODE_DIR}/settings.json"
python3 - <<PY
import json
from pathlib import Path
p = Path("${SETTINGS}")
data = {}
if p.is_file():
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError:
        data = {}
data["python.defaultInterpreterPath"] = "/raid/data/aihimekpen/venvs/research_summer_2026/bin/python"
data["python.terminal.activateEnvironment"] = True
data["notebook.defaultKernel"] = "python3"
p.write_text(json.dumps(data, indent=2) + "\n")
print("updated", p)
PY

echo ""
echo "OK  : ${PROJ}/.venv -> ${VENV_REAL}"
echo "OK  : Jupyter kernel '${DISPLAY_NAME}' (id: ${KERNEL_NAME})"
echo "OK  : Cursor interpreter -> research/.venv"
echo "OK  : all research/notebooks/*.ipynb pinned to '${DISPLAY_NAME}'"
echo "In any notebook: kernel picker → ${DISPLAY_NAME}"
echo "Re-pin later: python3 research/scripts/pin_notebook_kernels.py"
