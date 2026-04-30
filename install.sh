#!/usr/bin/env bash
# install.sh — bootstrap claude-code-proxy against a live Nebius account.
#
# Implements lessons learned from real installs:
#   - pip <22 in a fresh venv can't do editable installs → upgrade pip first
#   - the bundled .env.example pins models that Nebius has retired → validate
#     configured model IDs against /v1/models before declaring success
#   - "server bound to :8083" is not the same as "request succeeds" → smoke
#     test /test-connection and exit non-zero if it fails
#   - prompting for the API key with `read -rs` keeps it out of shell history

set -euo pipefail

red()    { printf '\033[31m%s\033[0m\n' "$*" >&2; }
green()  { printf '\033[32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[33m%s\033[0m\n' "$*"; }
info()   { printf '\033[36m==>\033[0m %s\n' "$*"; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

info "Checking prerequisites"
command -v python3 >/dev/null || { red "python3 not found"; exit 1; }
command -v curl    >/dev/null || { red "curl not found";    exit 1; }

PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' \
  || { red "Python >= 3.9 required, found $PY_VER"; exit 1; }
green "  python3 $PY_VER"

if [[ ! -d .venv ]]; then
  info "Creating .venv"
  python3 -m venv .venv
fi

info "Upgrading pip in .venv (fresh venvs ship pip <22 which fails on pyproject editable installs)"
.venv/bin/python -m pip install --quiet --upgrade pip

info "Installing dependencies from requirements.txt"
.venv/bin/pip install --quiet -r requirements.txt

if [[ -f .env ]]; then
  yellow ".env already exists — leaving it alone. Edit it manually to change keys or models."
else
  info "Creating .env from .env.example"
  cp .env.example .env
  printf "Paste your Nebius API key (input hidden, press Enter when done): "
  read -rs NEBIUS_KEY
  echo
  [[ -n "$NEBIUS_KEY" ]] || { red "No key provided"; exit 1; }

  NEBIUS_KEY="$NEBIUS_KEY" .venv/bin/python <<'PY'
import os, pathlib, re
key = os.environ["NEBIUS_KEY"]
p = pathlib.Path(".env")
text = p.read_text()
text = re.sub(
    r'^OPENAI_API_KEY=.*$',
    f'OPENAI_API_KEY={key}',
    text,
    count=1,
    flags=re.MULTILINE,
)
p.write_text(text)
PY
  chmod 600 .env
  green "  wrote .env (mode 600)"
fi

info "Validating configured models against Nebius /v1/models"
.venv/bin/python <<'PY'
import json, pathlib, sys, urllib.request

env = {}
for line in pathlib.Path(".env").read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    env[k.strip()] = v.strip().strip('"').strip("'").split("#", 1)[0].strip()

key  = env.get("OPENAI_API_KEY", "")
base = env.get("OPENAI_BASE_URL", "https://api.tokenfactory.nebius.com/v1").rstrip("/")
if not key or "YOUR_NEBIUS_API_KEY_HERE" in key:
    print("  no usable API key in .env — fill it in and re-run", file=sys.stderr)
    sys.exit(1)

req = urllib.request.Request(f"{base}/models", headers={"Authorization": f"Bearer {key}"})
try:
    with urllib.request.urlopen(req, timeout=15) as r:
        available = {m["id"] for m in json.load(r).get("data", [])}
except Exception as e:
    print(f"  could not list models from {base}: {e}", file=sys.stderr)
    sys.exit(1)

configured = {k: env[k] for k in ("BIG_MODEL", "MIDDLE_MODEL", "SMALL_MODEL", "VISION_MODEL") if env.get(k)}
missing = {k: v for k, v in configured.items() if v not in available}
if missing:
    print("  some configured models are not available on Nebius:", file=sys.stderr)
    for k, v in missing.items():
        print(f"      {k}={v}", file=sys.stderr)
    print("  examples of currently-available IDs:", file=sys.stderr)
    for m in sorted(available)[:15]:
        print(f"      {m}", file=sys.stderr)
    print("  edit .env and re-run install.sh.", file=sys.stderr)
    sys.exit(1)

print(f"  all {len(configured)} configured models are live")
PY
green "  models validated"

info "Smoke-testing the proxy (boot, /test-connection, shut down)"
LOG=$(mktemp -t claude-proxy-smoke.XXXXXX.log)
.venv/bin/python start_proxy.py >"$LOG" 2>&1 &
PROXY_PID=$!
cleanup() { kill "$PROXY_PID" 2>/dev/null || true; }
trap cleanup EXIT

PORT="$(grep -E '^PORT=' .env 2>/dev/null | tail -n1 | cut -d= -f2 | tr -d '"' || echo 8083)"
PORT="${PORT:-8083}"

for _ in $(seq 1 30); do
  if curl -sf -m 2 "http://localhost:${PORT}/health" >/dev/null 2>&1; then break; fi
  sleep 0.5
done

if ! curl -sf -m 2 "http://localhost:${PORT}/health" >/dev/null 2>&1; then
  red "  proxy did not bind to :${PORT}; first 40 lines of log:"
  head -40 "$LOG" >&2
  exit 1
fi

RESULT="$(curl -s -m 30 "http://localhost:${PORT}/test-connection")"
STATUS="$(printf '%s' "$RESULT" | .venv/bin/python -c 'import json,sys; print(json.load(sys.stdin).get("status",""))' 2>/dev/null || true)"

cleanup
trap - EXIT

if [[ "$STATUS" != "success" ]]; then
  red "  /test-connection did not return success:"
  printf '%s\n' "$RESULT" >&2
  exit 1
fi
green "  /test-connection: success"

green ""
green "Install complete."
cat <<MSG

To use the proxy, two things need to be running:

  1) The proxy itself, in another terminal:
       cd $REPO_ROOT && .venv/bin/python start_proxy.py

  2) Claude Code wired up to talk to the proxy. Add to your shell rc
     (~/.zshrc or ~/.bashrc):

       export ANTHROPIC_BASE_URL=http://localhost:${PORT}
       export ANTHROPIC_API_KEY=claude-local

     Then open a new shell. Or run as a one-off:
       ANTHROPIC_BASE_URL=http://localhost:${PORT} ANTHROPIC_API_KEY=claude-local claude
MSG
