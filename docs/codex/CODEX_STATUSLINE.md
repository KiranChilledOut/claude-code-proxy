# Codex Statusline

Codex CLI routes to OpenAI by default. This guide shows how to point it at the proxy and display the real backend model and context usage.

---

## 1. Proxy Routing

Add to `~/.codex/config.toml` (user-level) or `.codex/config.toml` (project-level):

```toml
model_provider = "openai"
openai_base_url = "http://localhost:8083/v1"

# Pick a model name Codex will send; the proxy maps it to your real backend
model = "gpt-5.5"
model_context_window = 128000
```

The proxy maps Codex model names to your actual backend:

| Codex requests | Proxy serves                   |
|----------------|--------------------------------|
| `gpt-*`, `o1-*`, `o3-*` | `BIG_MODEL` (e.g., `moonshotai/Kimi-K2.6`) |
| Any "mini" variant        | `SMALL_MODEL`                  |

> Note: The Codex CLI shows the model name it sent (`gpt-5.5`), not the backend model the proxy is serving. Use the proxy dashboard `http://localhost:8083/dashboard` to see the real model.

---

## 2. Desktop App: Setting Environment Variables

The Codex Desktop App (macOS) runs outside your shell and does **not** inherit `.zshrc`/`.bashrc` exports. You must set the API key via `launchctl` so the app can reach the proxy:

```bash
launchctl setenv OPENAI_API_KEY nb-...
```

Then launch Codex from the same terminal:

```bash
open -a "Codex"
```

Or add the `setenv` to your shell rc file and re-source it before launching.

---

## 3. Status Line

Codex's statusbar is controlled by `tui.status_line`, an ordered list of built-in identifiers:

```toml
[tui]
status_line = ["model", "mode", "project"]
```

Unlike Claude Code's `statusLine.command` (which can run arbitrary shell commands), Codex's `tui.status_line` accepts **predefined identifiers only**. The available identifiers depend on your Codex CLI version; common ones include:

- `model` — the active model name (e.g., `gpt-5.5`)
- `mode` — current interaction mode
- `project` — project directory name
- `spinner` — loading indicator

To disable the status line entirely:

```toml
[tui]
status_line = null
```

---

## 4. Context Usage via Shell Prompt (Workaround)

Since Codex does not yet support command-based statusline items, you can surface the proxy's live context usage through your shell prompt instead.

Add this to `~/.zshrc` or `~/.bashrc`:

```bash
# Proxy context usage in shell prompt
__proxy_status() {
    local base="${OPENAI_BASE_URL:-http://localhost:8083/v1}"
    base="${base%/v1}"
    base="${base%/}"

    local cfg ctx model free
    cfg=$(curl -fsS --max-time 1 "$base/api/observability/config" 2>/dev/null || true)
    model=$(printf '%s' "$cfg" | python3 -c 'import json,sys; d=json.load(sys.stdin); print((d.get("configured_models") or {}).get("big") or "")' 2>/dev/null || true)

    ctx=$(curl -fsS --max-time 1 "$base/api/observability/context-usage" 2>/dev/null || true)
    free=$(printf '%s' "$ctx" | python3 -c 'import json,sys; d=json.load(sys.stdin); r=d.get("remaining_tokens",1048576) or 1048576; t=d.get("context_limit",1048576) or 1048576; print(f"{round((r/t)*100)}")' 2>/dev/null || true)

    if [[ -n "$model" && -n "$free" && "$free" =~ ^[0-9]+$ ]]; then
        if (( free <= 20 )); then
            echo "%F{red}[$model ${free}% free]%f"
        elif (( free <= 40 )); then
            echo "%F{208}[$model ${free}% free]%f"
        elif (( free <= 50 )); then
            echo "%F{yellow}[$model ${free}% free]%f"
        else
            echo "%F{green}[$model ${free}% free]%f"
        fi
    elif [[ -n "$model" ]]; then
        echo "[$model]"
    fi
}

# Include in prompt (zsh)
PROMPT='$(__proxy_status) '$PROMPT
```

> Bash users: replace `%F{color}` with ANSI escape sequences, e.g., `\[\e[32m\]` and `%f` with `\[\e[0m\]`.

---

## 5. Full Codex Config Example

```toml
# ~/.codex/config.toml
model = "gpt-5.5"
model_provider = "openai"
openai_base_url = "http://localhost:8083/v1"
model_context_window = 128000

[tui]
status_line = ["model", "mode"]
```

---

## 6. Observability Endpoints Reference

The proxy exposes these endpoints for both Claude Code and Codex:

| Endpoint | Data |
|----------|------|
| `GET /api/observability/config` | `configured_models.big`, pricing |
| `GET /api/observability/context-usage` | `remaining_tokens`, `context_limit`, `percentage_used` |
| `GET /api/observability/summary` | Request count, tokens, latency |

All endpoints are available at the proxy base URL (without the `/v1` path).
