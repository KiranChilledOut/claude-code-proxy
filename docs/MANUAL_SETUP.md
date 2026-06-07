# Manual Setup Guide

Bypass the TUI and set everything up by hand.

---

## 1. Dependencies

```bash
git clone <repo-url>
cd claude-code-proxy
```

---

## 2. Create Environment & Install

Choose one of the three methods below.

### 2a. Standard venv

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 2b. uv

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```

> `uv` is a fast Python package manager. If it is not installed, see [astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/).

### 2c. Docker

```bash
# Create .env first (see §3), then:
docker compose up --build -d
```

> The proxy is then reachable on the host at `http://localhost:8083`.

---

## 3. Environment Variables

Create `.env` from the example:

```bash
cp .env.example .env
```

Edit the file. The values you **must** change are marked below.

| Variable | Required? | Description |
|----------|-----------|-------------|
| `OPENAI_API_KEY` | **Yes** | Your Nebius API key from [nebius.com](https://nebius.com) |
| `ANTHROPIC_API_KEY` | No | Keep as `claude-local` (proxy uses bearer auth) |
| `OPENAI_BASE_URL` | No | Defaults to `https://api.tokenfactory.nebius.com/v1` |
| `BIG_MODEL` | No | Defaults to `moonshotai/Kimi-K2.6` |
| `MIDDLE_MODEL` | No | Defaults to `moonshotai/Kimi-K2.6` |
| `SMALL_MODEL` | No | Defaults to `moonshotai/Kimi-K2.6` |
| `VISION_MODEL` | No | Defaults to `moonshotai/Kimi-K2.6` |
| `HOST` | No | Defaults to `0.0.0.0` |
| `PORT` | No | Defaults to `8083` |
| `LOG_LEVEL` | No | `INFO` or `DEBUG` |
| `MAX_TOKENS_LIMIT` | No | Defaults to `16384` |
| `STATUSLINE_PERCENT_ADJUST` | No | Offset the free-% shown in the statusline (e.g. `-10` means 80% shows as 90% free). Range `-100` to `+100` |

> Verify model IDs against the live endpoint before committing: `GET https://api.tokenfactory.nebius.com/v1/models`

---

## 4. Start the Proxy

| Method | Command |
|--------|---------|
| Standard venv | `.venv/bin/python start_proxy.py` |
| uv | `.venv/bin/python start_proxy.py` (or `uv run python start_proxy.py` if you prefer) |
| Docker | `docker compose up -d` (already running from §2c) |

The dashboard opens at `http://localhost:8083/dashboard`.

---

## 5. Use Claude Code Through the Proxy

### One-shot (no shell changes)

```bash
# 1. Start the proxy in a terminal (see §4)

# 2. In another terminal:
export ANTHROPIC_AUTH_TOKEN=claude-local
export ANTHROPIC_BASE_URL=http://localhost:8083
claude
```

### Persistent via shell config

Add these exports to your shell so `claude` always talks to the proxy:

**Bash / Zsh** — append to `~/.bashrc` or `~/.zshrc`:

```bash
export ANTHROPIC_AUTH_TOKEN=claude-local
export ANTHROPIC_BASE_URL=http://localhost:8083
```

**Fish** — append to `~/.config/fish/config.fish`:

```fish
set -gx ANTHROPIC_AUTH_TOKEN claude-local
set -gx ANTHROPIC_BASE_URL http://localhost:8083
```

**PowerShell** — append to your profile (`$PROFILE`):

```powershell
$env:ANTHROPIC_AUTH_TOKEN = "claude-local"
$env:ANTHROPIC_BASE_URL = "http://localhost:8083"
```

> Reload the config with `source ~/.bashrc` (or `~/.zshrc`, or restart your terminal).

### Persistent via a sourced script

Keep everything in one file that you source per-session:

**`~/claude-proxy.sh`** (Bash / Zsh)

```bash
export ANTHROPIC_AUTH_TOKEN=claude-local
export ANTHROPIC_BASE_URL=http://localhost:8083
alias claudius='claude'
echo "claude-proxy env loaded"
```

Usage:

```bash
source ~/claude-proxy.sh
claude
```

---

## 6. Claude Code Statusline

The statusline shows your current model and remaining context % inside Claude Code.

Create or edit `~/.claude/settings.json`:

```json
{
  "statusLine": {
    "type": "command",
    "command": "[ -z \"$ANTHROPIC_BASE_URL\" ] && exit 0; base=\"${ANTHROPIC_BASE_URL%/}\"; cfg=$(curl -fsS --max-time 1 \"$base/api/observability/config\" 2>/dev/null || true); model=$(printf '%s' \"$cfg\" | python3 -c 'import json,sys; d=json.load(sys.stdin); print((d.get(\"configured_models\") or {}).get(\"big\") or \"\")' 2>/dev/null || true); ctx=$(curl -fsS --max-time 1 \"$base/api/observability/context-usage\" 2>/dev/null || true); free=$(printf '%s' \"$ctx\" | python3 -c 'import json,sys; d=json.load(sys.stdin); r=d.get(\"remaining_tokens\",1048576) or 1048576; t=d.get(\"context_limit\",1048576) or 1048576; print(f\"{round((r/t)*100)}\")' 2>/dev/null || true); if [ -n \"$model\" ]; then if [ -n \"$free\" ] && [[ \"$free\" =~ ^[0-9]+$ ]]; then if [ \"$free\" -le 20 ]; then c=\"\\033[31m\"; elif [ \"$free\" -le 40 ]; then c=\"\\033[38;5;208m\"; elif [ \"$free\" -le 50 ]; then c=\"\\033[33m\"; else c=\"\\033[32m\"; fi; e=\"\\033[0m\"; echo \"[nebius://$model $c${free}% free$e] $base/dashboard\"; else echo \"[nebius://$model] $base/dashboard\"; fi; else echo \"[proxy://$base]\"; fi"
  }
}
```

> Restart Claude Code to see the statusline update.

---

### Codex Statusline

Codex CLI uses `~/.codex/config.toml`, not `~/.claude/settings.json`. Codex's `tui.status_line` accepts predefined identifiers only (e.g., `model`, `mode`, `project`) and does not support a shell-command statusline like Claude Code.

To see the **real backend model** and **context %**, choose from:

1. **Use the Codex dashboard** — open `http://localhost:8083/dashboard` in a browser for live session info.
2. **Add context usage to your shell prompt** — see [docs/codex/CODEX_STATUSLINE.md](../codex/CODEX_STATUSLINE.md#3-context-usage-via-shell-prompt-workaround).

---

## 7. Convenience Shell Shortcuts (optional)

These are what the TUI optionally installs. Add them manually if you want `claude --proxy` and `claudius` aliases.

**Bash / Zsh**

Add to `~/.bashrc` or `~/.zshrc`:

```bash
# Claude Shell Function — enables claude, claude --proxy, and claudius
claude() {
    local main_proxy="http://localhost:8083"
    local repo_root="$(pwd)"
    if [[ "$1" == "--proxy" ]]; then
        printf "\033[38;5;129m▐▛▜▌ Claude via Proxy\033[0m  \033[38;5;244m→ bearer auth via local proxy\033[0m\n"
        local default_name="session-$(date +%Y%m%d-%H%M%S)"
        printf "\033[38;5;244mSession name\033[0m [\033[38;5;75m%s\033[0m]: " "$default_name"
        read -r session_name
        session_name="${session_name:-$default_name}"
        local local_port
        local_port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
        mkdir -p "$repo_root/logs"
        python3 "$repo_root/scripts/session_forwarder.py" "$local_port" "localhost:8083" "$session_name" >> "$repo_root/logs/session-forwarder.log" 2>&1 &
        local forwarder_pid=$!
        sleep 0.5
        local forwarder_url="http://localhost:$local_port"
        (
            unset ANTHROPIC_API_KEY
            export ANTHROPIC_AUTH_TOKEN="claude-local"
            export ANTHROPIC_BASE_URL="$forwarder_url"
            command claude "${@:2}"
        )
        local claude_exit=$?
        kill "$forwarder_pid" 2>/dev/null || true
        wait "$forwarder_pid" 2>/dev/null || true
        return $claude_exit
    else
        printf "\033[38;5;46m▐▛▜▌ Claude Direct\033[0m  \033[38;5;244m→ subscription login auth\033[0m\n"
        (
            unset ANTHROPIC_BASE_URL ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN
            command claude "$@"
        )
    fi
}
alias claudius='claude --proxy'
```

> Change `repo_root` to the absolute path of your clone if you source this from outside the repo directory.

---

## Summary of Commands

| Task | Command |
|------|---------|
| Start proxy (venv/uv) | `.venv/bin/python start_proxy.py` |
| Start proxy (Docker) | `docker compose up -d` |
| Open dashboard | `http://localhost:8083/dashboard` |
| Run Claude via proxy | `claude --proxy` (if you added the function) |
| Quick env vars | `export ANTHROPIC_AUTH_TOKEN=claude-local; export ANTHROPIC_BASE_URL=http://localhost:8083` |

