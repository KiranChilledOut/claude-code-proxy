#!/usr/bin/env bash
# install.sh — thin bootstrap; the real wizard lives in install_tui.py.
#
# 1. Checks python3 >= 3.9.
# 2. Installs textual (into the system Python or a temporary venv).
# 3. Launches python3 install_tui.py

set -uo pipefail

C_RESET='\033[0m'
C_RED='\033[38;5;197m'
C_GREEN='\033[38;5;46m'
C_CYAN='\033[38;5;45m'

fatal() { printf '%b%s%b\n' "$C_RED" "$*" "$C_RESET" >&2; }
info()  { printf '%b%s%b\n' "$C_CYAN" "$*" "$C_RESET"; }

# ─── 1. Python ───────────────────────────────────────────────────
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

# ─── 2. Textual ──────────────────────────────────────────────────
# Try to import textual.  If missing, install it temporarily.
if ! "$PYTHON" -c "import textual" 2>/dev/null; then
    info "Installing Textual (TUI library) …"
    if ! "$PYTHON" -m pip install --quiet textual 2>/dev/null; then
        # System pip may be blocked (PEP 668) — create a throw-away venv
        TEMPDIR=$(mktemp -d /tmp/claude-proxy-tui.XXXXXX)
        "$PYTHON" -m venv "$TEMPDIR/.venv" >/dev/null 2>&1 || {
            fatal "Could not install Textual."
            info  "Try manually:  $PYTHON -m pip install textual"
            exit 1
        }
        "$TEMPDIR/.venv/bin/pip" install --quiet textual >/dev/null 2>&1 || {
            fatal "Could not install Textual."
            exit 1
        }
        PYTHON="$TEMPDIR/.venv/bin/python"
    fi
fi

# ─── 3. Launch TUI ───────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$PYTHON" "$SCRIPT_DIR/install_tui.py" "$@"
