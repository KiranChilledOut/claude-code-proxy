#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required to start macoscontrol-mcp" >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  python3 -m venv "$VENV_DIR"
fi

if ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
  "$PYTHON_BIN" -m ensurepip --upgrade >/dev/null 2>&1 || true
fi

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import importlib
for name in ("mcp", "pyautogui", "mss", "PIL", "server"):
    importlib.import_module(name)
PY
then
  "$PYTHON_BIN" -m pip install --upgrade pip
  "$PYTHON_BIN" -m pip install -e "$SCRIPT_DIR"
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/server.py"
