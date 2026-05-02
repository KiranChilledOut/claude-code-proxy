#!/usr/bin/env bash
# install.sh — bootstrap claude-code-proxy against a live Nebius account.

set -euo pipefail

# Claude-style colors
C_RESET='\033[0m'
C_BOLD='\033[1m'
C_ICYAN='\033[38;5;45m'    # Claude cyan
C_PURPLE='\033[38;5;129m'  # Claude purple
C_GREEN='\033[38;5;46m'    # Success green
C_YELLOW='\033[38;5;220m'  # Warning yellow
C_RED='\033[38;5;197m'     # Error red
C_GRAY='\033[38;5;244m'    # Dim gray
C_BLUE='\033[38;5;75m'     # Info blue

red()    { printf "${C_RED}✘${C_RESET} %s\n" "$*" >&2; }
green()  { printf "${C_GREEN}✔${C_RESET} %s\n" "$*"; }
yellow() { printf "${C_YELLOW}⚠${C_RESET} %s\n" "$*"; }
info()   { printf "${C_BLUE}▸${C_RESET} %s\n" "$*"; }
step()   { printf "${C_ICYAN}▐▛▜▌${C_RESET} ${C_BOLD}%s${C_RESET}\n" "$*"; }

# Banner
banner() {
    cat <<'EOF'
╭──────────────────────────────────────────────────────────────────╮
│                                                                  │
│   ${C_ICYAN}▐▛▜▌${C_RESET}  ${C_BOLD}Claude Code Proxy for Nebius${C_RESET}                              │
│                                                                  │
│   Install with style. Connect with ease.                         │
│                                                                  │
╰──────────────────────────────────────────────────────────────────╯
EOF
    # Apply colors after heredoc
    printf "${C_ICYAN}▐▛▜▌${C_RESET}  ${C_BOLD}Claude Code Proxy for Nebius${C_RESET}\n\n"
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Print banner
printf "${C_ICYAN}▐▛▜▌${C_RESET}  ${C_BOLD}Claude Code Proxy for Nebius${C_RESET}\n"
printf "${C_GRAY}Install with style. Connect with ease.${C_RESET}\n\n"

step "Checking prerequisites"
command -v python3 >/dev/null || { red "python3 not found"; exit 1; }
command -v curl    >/dev/null || { red "curl not found";    exit 1; }

PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' \
  || { red "Python >= 3.9 required, found $PY_VER"; exit 1; }
green "python3 $PY_VER"

if [[ ! -d .venv ]]; then
  step "Creating virtual environment"
  python3 -m venv .venv
  green ".venv created"
fi

step "Upgrading pip"
.venv/bin/python -m pip install --quiet --upgrade pip
green "pip upgraded"

step "Installing dependencies"
.venv/bin/pip install --quiet -r requirements.txt
green "dependencies installed"

if [[ -f .env ]]; then
  yellow ".env already exists —/edit manually to change keys or models"
else
  step "Creating .env"
  cp .env.example .env
  printf "${C_GRAY}Paste your Nebius API key${C_RESET} ${C_YELLOW}(input hidden, Enter when done)${C_RESET}: "
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
  green ".env created (mode 600)"
fi

step "Validating configured models"
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

print(f"  {len(configured)} models validated")
PY
green "all models available"

step "Testing proxy connection"
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
  red "proxy did not bind to :${PORT}"
  head -20 "$LOG" >&2
  exit 1
fi

RESULT="$(curl -s -m 30 "http://localhost:${PORT}/test-connection")"
STATUS="$(printf '%s' "$RESULT" | .venv/bin/python -c 'import json,sys; print(json.load(sys.stdin).get("status",""))' 2>/dev/null || true)"

cleanup
trap - EXIT

if [[ "$STATUS" != "success" ]]; then
  red "connection test failed"
  printf '%s\n' "$RESULT" >&2
  exit 1
fi

# Success banner
printf "\n"
printf "${C_PURPLE}╭───────────────────────────────────────────────────────────${C_RESET}\n"
printf "${C_PURPLE}│${C_RESET}  ${C_GREEN}✔${C_RESET} ${C_BOLD}All systems operational${C_RESET}                            ${C_PURPLE}│${C_RESET}\n"
printf "${C_PURPLE}│${C_RESET}                                                            ${C_PURPLE}│${C_RESET}\n"
printf "${C_PURPLE}│${C_RESET}  ${C_ICYAN}▐▛▜▌${C_RESET}  Proxy ready at ${C_YELLOW}http://localhost:${PORT}${C_RESET}                    ${C_PURPLE}│${C_RESET}\n"
printf "${C_PURPLE}╰───────────────────────────────────────────────────────────${C_RESET}\n"
printf "\n"

# Shell function setup
printf "${C_PURPLE}╭───────────────────────────────────────────────────────────${C_RESET}\n"
printf "${C_PURPLE}│${C_RESET}  ${C_BOLD}Claude Shell Function${C_RESET}                                  ${C_PURPLE}│${C_RESET}\n"
printf "${C_PURPLE}│${C_RESET}                                                            ${C_PURPLE}│${C_RESET}\n"
printf "${C_PURPLE}│${C_RESET}  ${C_GRAY}Quick switch between direct and proxy:${C_RESET}                    ${C_PURPLE}│${C_RESET}\n"
printf "${C_PURPLE}│${C_RESET}                                                            ${C_PURPLE}│${C_RESET}\n"
printf "${C_PURPLE}│${C_RESET}    ${C_GREEN}claude${C_RESET}         → Direct (subscription login)           ${C_PURPLE}│${C_RESET}\n"
printf "${C_PURPLE}│${C_RESET}    ${C_PURPLE}claude --proxy${C_RESET}  → Via local proxy (Nebius)            ${C_PURPLE}│${C_RESET}\n"
printf "${C_PURPLE}│${C_RESET}    ${C_PURPLE}claudius${C_RESET}       → Alias for --proxy                    ${C_PURPLE}│${C_RESET}\n"
printf "${C_PURPLE}╰───────────────────────────────────────────────────────────${C_RESET}\n"
printf "\n"

# Detect user's default shell profile
SHELL_RC=""
user_shell="${SHELL:-/bin/bash}"
case "$user_shell" in
    */zsh)
        SHELL_RC="$HOME/.zshrc"
        ;;
    */bash)
        SHELL_RC="$HOME/.bashrc"
        ;;
    *)
        if [[ -f "$HOME/.zshrc" ]]; then
            SHELL_RC="$HOME/.zshrc"
        elif [[ -f "$HOME/.bashrc" ]]; then
            SHELL_RC="$HOME/.bashrc"
        fi
        ;;
esac

if [[ -f "$SHELL_RC" ]] && grep -q "claude() {" "$SHELL_RC" 2>/dev/null; then
    yellow "Already configured in $SHELL_RC"
else
    printf "${C_GRAY}Add shell function to $SHELL_RC?${C_RESET} [${C_GREEN}Y${C_RESET}/n]: "
    read -r response
    case "${response:-y}" in
        [Yy]|"")
            if [[ -n "$SHELL_RC" ]]; then
                cat >> "$SHELL_RC" <<'SHELL_FUNC'

# Claude Shell Function — enables claude, claude --proxy, and claudius
claude() {
    local proxy_url="http://localhost:${PORT:-8083}"

    if [[ "$1" == "--proxy" ]] || [[ "$1" == "claudius" ]]; then
        printf "\033[38;5;129m▐▛▜▌ Claude via Proxy\033[0m  \033[38;5;244m→ API key auth via local proxy\033[0m\n"
        ANTHROPIC_AUTH_TOKEN="tokenfactory" \
        ANTHROPIC_API_KEY="dummy" \
        ANTHROPIC_BASE_URL="$proxy_url" \
        command claude "${@:2}"
    else
        printf "\033[38;5;46m▐▛▜▌ Claude Direct\033[0m  \033[38;5;244m→ subscription login auth\033[0m\n"
        env -u ANTHROPIC_BASE_URL -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN \
        command claude "$@"
    fi
}

# Alias for users who prefer claudius style
alias claudius='claude --proxy'
SHELL_FUNC
                green "Added to $SHELL_RC"
                yellow "Run: source $SHELL_RC"
            else
                red "Could not detect shell profile"
            fi
            ;;
        *)
            info "Skipped — see docs/SHELL_FUNCTION.md"
            ;;
    esac
fi

printf "\n"
printf "${C_ICYAN}▐▛▜▌${C_RESET}  ${C_BOLD}Next steps:${C_RESET}\n"
printf "\n"
printf "${C_GRAY}1.${C_RESET} Start the proxy:\n"
printf "      ${C_YELLOW}cd $REPO_ROOT && .venv/bin/python start_proxy.py${C_RESET}\n"
printf "\n"
printf "${C_GRAY}2.${C_RESET} Use Claude:\n"
printf "      ${C_PURPLE}claude --proxy${C_RESET}   # via Nebius proxy\n"
printf "      ${C_GREEN}claude${C_RESET}            # direct (subscription)\n"
printf "\n"