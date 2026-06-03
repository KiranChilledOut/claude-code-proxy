#!/usr/bin/env bash
# install.sh — bootstrap the TUI installer.
#
# Zero-knowledge: if .venv exists, use it; otherwise create one.
# Asks whether to use uv (if available) or standard venv.
# Installs requirements.txt + textual, then launches the TUI.
#
# Full log written to: install.log (in the same directory as this script)

set -uo pipefail

C_RESET='\033[0m'
C_RED='\033[38;5;197m'
C_GREEN='\033[38;5;46m'
C_CYAN='\033[38;5;45m'
C_DIM='\033[38;5;244m'
C_YELLOW='\033[33m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/install.log"

_log() {
    printf '%b\n' "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG_FILE"
}

fatal() { _log "FATAL: $*"; printf '%b%s%b\n' "$C_RED" "$*" "$C_RESET" >&2; }
info()  { _log "INFO:  $*"; printf '%b%s%b\n' "$C_CYAN" "$*" "$C_RESET"; }
ok()    { _log "OK:    $*"; printf '%b%s%b\n' "$C_GREEN" "$*" "$C_RESET"; }
dim()   { _log "DIM:   $*"; printf '%b%s%b\n' "$C_DIM" "$*" "$C_RESET"; }
ask()   { printf '%b%s%b'  "$C_YELLOW" "$*" "$C_RESET"; }

# Log stderr from pip/uv commands into the log as well as the terminal.
_log_to_both() {
    local prefix="$1"
    while IFS= read -r line; do
        _log "$prefix $line"
        printf '%s\n' "$line"
    done
}

trap '_log "--- install.sh exited with code $? ---"' EXIT

# ─── 1. Python ───────────────────────────────────────────────────
_log "===== Step 1: Python ====="
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        ver=$("$cmd" -c 'import sys; print(".".join(map(str, sys.version_info[:2])))' 2>/dev/null)
        arg1=${1:-}
        if [ "$arg1" != "--skip-version-check" ] && [ -n "$ver" ]; then
            major=$(echo "$ver" | cut -d. -f1)
            minor=$(echo "$ver" | cut -d. -f2)
            if [ "$major" -gt 3 ] || { [ "$major" -eq 3 ] && [ "$minor" -ge 9 ]; }; then
                PYTHON="$cmd"
                break
            fi
        else
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    fatal "Python 3.9+ is required but not found."
    info  "Install from https://python.org/downloads/"
    exit 1
fi
ok "Python → $PYTHON"

# ─── 2. Virtual environment ─────────────────────────────────────
_log "===== Step 2: Virtual environment ====="
VENV_DIR="$SCRIPT_DIR/.venv"
UV_AVAILABLE=""
command -v uv &>/dev/null && UV_AVAILABLE="1"

_log "uv on PATH: ${UV_AVAILABLE:-no}"

if [ -d "$VENV_DIR" ]; then
    info "Re-using existing virtual environment: $VENV_DIR"
    _log ".venv exists. Checking for pip binary …"

    if [ -n "$UV_AVAILABLE" ] && [ ! -f "$VENV_DIR/bin/pip" ] && [ ! -f "$VENV_DIR/bin/pip3" ]; then
        USE_UV="1"
        _log "No pip/pip3 in $VENV_DIR — treating as uv-managed."
    else
        if [ -f "$VENV_DIR/bin/pip" ] || [ -f "$VENV_DIR/bin/pip3" ]; then
            _log "Found pip in $VENV_DIR."
        else
            _log "No uv, no pip. Will need to bootstrap."
        fi
        USE_UV=""
    fi
else
    # Pick package manager
    if [ -n "$UV_AVAILABLE" ]; then
        ask "uv found. Use uv for faster installs?"
        printf '%b [Y/n]%b ' "$C_GREEN" "$C_RESET"
        if [ -t 0 ]; then read -r answer; else answer=""; fi
        [ -z "$answer" ] || echo "$answer" | grep -qi '^y'
        if [ $? -eq 0 ]; then
            USE_UV="1"
            _log "User chose uv."
        else
            USE_UV=""
            _log "User chose standard venv."
        fi
    else
        USE_UV=""
        _log "uv not found."
    fi

    # Create venv
    if [ "$USE_UV" = "1" ]; then
        info "Creating virtual environment with uv …"
        if uv venv "$VENV_DIR" --python "$PYTHON" >>"$LOG_FILE" 2>&1; then
            ok "Virtual environment created with uv."
        else
            fatal "uv venv failed."
            _log "uv venv command output:
$(cat "$LOG_FILE" | tail -5)"
            info "Falling back to python -m venv …"
            USE_UV=""
        fi
    fi

    if [ ! -d "$VENV_DIR" ]; then
        info "Creating virtual environment with python -m venv …"
        if "$PYTHON" -m venv "$VENV_DIR" >>"$LOG_FILE" 2>&1; then
            ok "Virtual environment ready."
        else
            _log "python -m venv failed. See log."
            fatal "Could not create virtual environment."
            info "Make sure the python3-venv (or python3.x-venv) package is installed."
            exit 1
        fi
    fi
fi

PYTHON="$VENV_DIR/bin/python"

# ─── Helper: pip bootstrap ─────────────────────────────────────────
_ensure_pip() {
    if [ -f "$VENV_DIR/bin/pip" ] || [ -f "$VENV_DIR/bin/pip3" ]; then
        _log "pip already present in .venv"
        return 0
    fi
    info "pip not found in .venv — bootstrapping …"
    _log "Running: $PYTHON -m ensurepip --upgrade"
    if "$PYTHON" -m ensurepip --upgrade >>"$LOG_FILE" 2>&1; then
        _log "ensurepip succeeded"
        return 0
    fi
    _log "ensurepip failed; trying _bootstrap() …"
    if "$PYTHON" -c "import ensurepip; ensurepip._bootstrap()" >>"$LOG_FILE" 2>&1; then
        _log "ensurepip._bootstrap() succeeded"
        return 0
    fi
    fatal "Could not install pip into the virtual environment."
    info  "Try: curl https://bootstrap.pypa.io/get-pip.py | $PYTHON"
    return 1
}

# ─── 3. Dependencies ─────────────────────────────────────────────
_log "===== Step 3: Dependencies ====="
REQ_FILE="$SCRIPT_DIR/requirements.txt"

if [ -f "$REQ_FILE" ]; then
    info "Installing dependencies …"

    if [ "$USE_UV" = "1" ]; then
        _log "Running: uv pip install --python $PYTHON -r $REQ_FILE"
        if uv pip install --python "$PYTHON" -r "$REQ_FILE" >>"$LOG_FILE" 2>&1; then
            ok "Dependencies ready."
        else
            _log "uv pip install stderr:\n$(tail -20 "$LOG_FILE")"
            fatal "uv pip install failed."
            exit 1
        fi
    else
        if ! _ensure_pip; then exit 1; fi
        _log "Running: $PYTHON -m pip install -r $REQ_FILE"
        if "$PYTHON" -m pip install --quiet -r "$REQ_FILE" >>"$LOG_FILE" 2>&1; then
            ok "Dependencies ready."
        else
            _log "pip install stderr:\n$(tail -20 "$LOG_FILE")"
            fatal "pip install failed."
            exit 1
        fi
    fi
else
    dim "No requirements.txt found. Skipping."
fi

# ─── 4. Textual ──────────────────────────────────────────────────
_log "===== Step 4: Textual ====="
if ! "$PYTHON" -c "import textual" 2>/dev/null; then
    info "Installing Textual …"

    if [ "$USE_UV" = "1" ]; then
        _log "Running: uv pip install --python $PYTHON textual"
        if uv pip install --python "$PYTHON" textual >>"$LOG_FILE" 2>&1; then
            ok "Textual installed."
        else
            _log "uv install textual stderr:\n$(tail -20 "$LOG_FILE")"
            fatal "uv install of textual failed."
            exit 1
        fi
    else
        if ! _ensure_pip; then exit 1; fi
        _log "Running: $PYTHON -m pip install textual"
        if "$PYTHON" -m pip install --quiet textual >>"$LOG_FILE" 2>&1; then
            ok "Textual installed."
        else
            _log "pip install textual stderr:\n$(tail -20 "$LOG_FILE")"
            fatal "pip install of textual failed."
            exit 1
        fi
    fi
else
    _log "Textual already importable."
fi

# ─── 5. Launch TUI ───────────────────────────────────────────────
_log "===== Step 5: Launch TUI ====="
dim "Launching installer …"
"$PYTHON" "$SCRIPT_DIR/install_tui.py" "$@"
