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
  yellow ".env already exists — edit manually to change keys or models"
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
cleanup() {
  kill "$PROXY_PID" 2>/dev/null || true
  wait "$PROXY_PID" 2>/dev/null || true
}
trap cleanup EXIT

PORT="$(grep -E '^PORT=' .env 2>/dev/null | tail -n1 | cut -d= -f2 | tr -d '"' || echo 8083)"
PORT="${PORT:-8083}"
BIG_LIMIT="$(grep -E '^BIG_MODEL_CONTEXT_LIMIT=' .env 2>/dev/null | tail -n1 | cut -d= -f2 | tr -d '"' || echo 0)"
BIG_LIMIT="${BIG_LIMIT:-0}"

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

# Detect user's shell profile. PowerShell usually keeps SHELL=/bin/zsh on macOS,
# so prefer the parent process when install.sh was launched from pwsh.
SHELL_KIND=""
SHELL_RC=""

pwsh_profile_path() {
    command -v pwsh >/dev/null || return 1
    pwsh -NoLogo -NoProfile -NonInteractive -Command '$PROFILE' 2>/dev/null \
        | tr -d '\r' \
        | tail -n 1
}

detect_shell_profile() {
    local parent_comm parent_name user_shell pwsh_profile
    parent_comm="$(ps -p "${PPID:-0}" -o comm= 2>/dev/null || true)"
    parent_name="${parent_comm##*/}"

    case "$parent_name" in
        bash)
            SHELL_KIND="bash"
            SHELL_RC="$HOME/.bashrc"
            return
            ;;
        zsh)
            SHELL_KIND="zsh"
            SHELL_RC="$HOME/.zshrc"
            return
            ;;
        pwsh|pwsh-*|powershell|powershell.exe)
            pwsh_profile="$(pwsh_profile_path || true)"
            if [[ -n "$pwsh_profile" ]]; then
                SHELL_KIND="pwsh"
                SHELL_RC="$pwsh_profile"
                return
            fi
            ;;
    esac

    user_shell="${SHELL:-/bin/bash}"
    case "$user_shell" in
        */pwsh|*/pwsh-*|*/powershell|*/powershell.exe)
            pwsh_profile="$(pwsh_profile_path || true)"
            if [[ -n "$pwsh_profile" ]]; then
                SHELL_KIND="pwsh"
                SHELL_RC="$pwsh_profile"
            fi
            ;;
        */zsh)
            SHELL_KIND="zsh"
            SHELL_RC="$HOME/.zshrc"
            ;;
        */bash)
            SHELL_KIND="bash"
            SHELL_RC="$HOME/.bashrc"
            ;;
        *)
            if [[ -f "$HOME/.zshrc" ]]; then
                SHELL_KIND="zsh"
                SHELL_RC="$HOME/.zshrc"
            elif [[ -f "$HOME/.bashrc" ]]; then
                SHELL_KIND="bash"
                SHELL_RC="$HOME/.bashrc"
            else
                pwsh_profile="$(pwsh_profile_path || true)"
                if [[ -n "$pwsh_profile" ]]; then
                    SHELL_KIND="pwsh"
                    SHELL_RC="$pwsh_profile"
                fi
            fi
            ;;
    esac
}

shell_label() {
    case "$SHELL_KIND" in
        pwsh) printf "PowerShell" ;;
        zsh) printf "zsh" ;;
        bash) printf "bash" ;;
        *) printf "shell" ;;
    esac
}

shell_already_configured() {
    case "$SHELL_KIND" in
        pwsh)
            [[ -f "$SHELL_RC" ]] && grep -Eqi 'function[[:space:]]+(global:)?claude([^[:alnum:]_]|$)' "$SHELL_RC" 2>/dev/null
            ;;
        *)
            [[ -f "$SHELL_RC" ]] && grep -q "claude() {" "$SHELL_RC" 2>/dev/null
            ;;
    esac
}

prepare_shell_profile() {
    mkdir -p "$(dirname "$SHELL_RC")"
    [[ -f "$SHELL_RC" ]] || touch "$SHELL_RC"
    cp "$SHELL_RC" "$SHELL_RC.bak.$(date +%s)"
    green "Backed up $SHELL_RC"
}

append_posix_shell_function() {
    # REPO_ROOT is already absolute at this point (set line 31)
    local _repo_root="$REPO_ROOT"
    local _ctx_limit="${BIG_LIMIT:-0}"
    cat >> "$SHELL_RC" <<SHELL_FUNC

# Claude Shell Function — enables claude, claude --proxy, and claudius
# Per-session forwarder: each --proxy run gets a unique port + session name
claude() {
    local main_proxy="http://localhost:${PORT}"
    local repo_root="${_repo_root}"

    if [[ "\$1" == "--proxy" ]]; then
        printf "\033[38;5;129m▐▛▜▌ Claude via Proxy\033[0m  \033[38;5;244m→ bearer auth via local proxy\033[0m\n"

        # Prompt for a session name (pre-fill with timestamp)
        local default_name
        default_name="session-\$(date +%Y%m%d-%H%M%S)"
        printf "\033[38;5;244mSession name\033[0m [\033[38;5;75m%s\033[0m]: " "\$default_name"
        read -r session_name
        session_name="\${session_name:-\$default_name}"

        # Pick a random free port
        local local_port
        local_port=\$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')

        # Start the forwarder in the background
        python3 "\$repo_root/scripts/session_forwarder.py" "\$local_port" "localhost:${PORT}" "\$session_name" &
        local forwarder_pid=\$!

        # Wait a moment for the forwarder to bind
        sleep 0.5

        local forwarder_url="http://localhost:\$local_port"
        (
            unset ANTHROPIC_API_KEY
            export ANTHROPIC_AUTH_TOKEN="claude-local"
            export ANTHROPIC_BASE_URL="\$forwarder_url"
            [ -n "${_ctx_limit}" ] && [ "${_ctx_limit}" -gt 0 ] && [ "${_ctx_limit}" -lt 1000000 ] && export ANTHROPIC_MODEL="claude-opus-4-7"
            command claude "\${@:2}"
        )
        local claude_exit=\$?

        # Clean up forwarder
        kill "\$forwarder_pid" 2>/dev/null || true
        wait "\$forwarder_pid" 2>/dev/null || true

        return \$claude_exit
    else
        printf "\033[38;5;46m▐▛▜▌ Claude Direct\033[0m  \033[38;5;244m→ subscription login auth\033[0m\n"
        (
            unset ANTHROPIC_BASE_URL ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN
            command claude "\$@"
        )
    fi
}

# Alias for users who prefer claudius style
alias claudius='claude --proxy'
SHELL_FUNC
}

append_pwsh_shell_function() {
    local _repo_root="$REPO_ROOT"
    local _ctx_limit="${BIG_LIMIT:-0}"
    cat >> "$SHELL_RC" <<PWSH_FUNC

# Claude Shell Function - enables claude, claude --proxy, and claudius
# Per-session forwarder: each --proxy run gets a unique port + session name
function claude {
    param(
        [Parameter(ValueFromRemainingArguments = \$true)]
        [string[]] \$ClaudeArgs
    )

    \$mainProxy = "http://localhost:${PORT}"
    \$repoRoot = "${_repo_root}"
    \$claudeCommand = (Get-Command claude -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
    \$oldAuthToken = \$env:ANTHROPIC_AUTH_TOKEN
    \$oldApiKey = \$env:ANTHROPIC_API_KEY
    \$oldBaseUrl = \$env:ANTHROPIC_BASE_URL

    if (\$ClaudeArgs.Count -gt 0 -and \$ClaudeArgs[0] -eq "--proxy") {
        Write-Host "\`e[38;5;129m▐▛▜▌ Claude via Proxy\`e[0m  \`e[38;5;244m-> bearer auth via local proxy\`e[0m"

        # Prompt for session name
        \$defaultName = "session-" + (Get-Date -Format "yyyyMMdd-HHmmss")
        Write-Host "Session name [\`e[38;5;75m\$defaultName\`e[0m]: " -NoNewline
        [string] \$sessionName = Read-Host
        if ([string]::IsNullOrWhiteSpace(\$sessionName)) { \$sessionName = \$defaultName }

        # Pick random free port
        [int] \$localPort = python3 -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1]); s.close()"

        # Start forwarder
        \$forwarderJob = Start-Job -ScriptBlock {
            param(\$port, \$target, \$name, \$repo)
            python3 "\$repo/scripts/session_forwarder.py" \$port \$target \$name
        } -ArgumentList \$localPort, "localhost:${PORT}", \$sessionName, \$repoRoot

        # Wait for forwarder to bind
        Start-Sleep -Milliseconds 800

        [string[]] \$remainingArgs = @()
        if (\$ClaudeArgs.Count -gt 1) {
            \$remainingArgs = [string[]] \$ClaudeArgs[1..(\$ClaudeArgs.Count - 1)]
        }

        \$forwarderUrl = "http://localhost:\$localPort"
        try {
            \$env:ANTHROPIC_AUTH_TOKEN = "claude-local"
            Remove-Item Env:ANTHROPIC_API_KEY -ErrorAction SilentlyContinue
            \$env:ANTHROPIC_BASE_URL = \$forwarderUrl
            if (${_ctx_limit} -gt 0 -and ${_ctx_limit} -lt 1000000) { \$env:ANTHROPIC_MODEL = "claude-opus-4-7" }
            & \$claudeCommand @remainingArgs
        } finally {
            # Clean up forwarder
            if (\$forwarderJob) {
                Stop-Job \$forwarderJob -ErrorAction SilentlyContinue
                Remove-Job \$forwarderJob -ErrorAction SilentlyContinue
            }
            if (\$null -eq \$oldAuthToken) { Remove-Item Env:ANTHROPIC_AUTH_TOKEN -ErrorAction SilentlyContinue } else { \$env:ANTHROPIC_AUTH_TOKEN = \$oldAuthToken }
            if (\$null -eq \$oldApiKey) { Remove-Item Env:ANTHROPIC_API_KEY -ErrorAction SilentlyContinue } else { \$env:ANTHROPIC_API_KEY = \$oldApiKey }
            if (\$null -eq \$oldBaseUrl) { Remove-Item Env:ANTHROPIC_BASE_URL -ErrorAction SilentlyContinue } else { \$env:ANTHROPIC_BASE_URL = \$oldBaseUrl }
        }
    } else {
        Write-Host "\`e[38;5;46m▐▛▜▌ Claude Direct\`e[0m  \`e[38;5;244m-> subscription login auth\`e[0m"

        try {
            Remove-Item Env:ANTHROPIC_AUTH_TOKEN -ErrorAction SilentlyContinue
            Remove-Item Env:ANTHROPIC_API_KEY -ErrorAction SilentlyContinue
            Remove-Item Env:ANTHROPIC_BASE_URL -ErrorAction SilentlyContinue
            & \$claudeCommand @ClaudeArgs
        } finally {
            if (\$null -ne \$oldAuthToken) { \$env:ANTHROPIC_AUTH_TOKEN = \$oldAuthToken }
            if (\$null -ne \$oldApiKey) { \$env:ANTHROPIC_API_KEY = \$oldApiKey }
            if (\$null -ne \$oldBaseUrl) { \$env:ANTHROPIC_BASE_URL = \$oldBaseUrl }
        }
    }
}

function claudius {
    param(
        [Parameter(ValueFromRemainingArguments = \$true)]
        [string[]] \$ClaudeArgs
    )

    claude --proxy @ClaudeArgs
}
PWSH_FUNC
}

detect_shell_profile

if [[ -z "$SHELL_RC" ]]; then
    red "Could not detect shell profile"
elif shell_already_configured; then
    yellow "Already configured in $SHELL_RC"
else
    printf "${C_GRAY}Add $(shell_label) function to $SHELL_RC?${C_RESET} [${C_GREEN}Y${C_RESET}/n]: "
    read -r response
    case "${response:-y}" in
        [Yy]|"")
            prepare_shell_profile
            if [[ "$SHELL_KIND" == "pwsh" ]]; then
                append_pwsh_shell_function
                green "Added to $SHELL_RC"
                yellow "Restart PowerShell or run: . \$PROFILE"
            else
                append_posix_shell_function
                green "Added to $SHELL_RC"
                yellow "Run: source $SHELL_RC"
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
