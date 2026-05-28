# Documentation Index

Full reference for `claude-code-proxy`. For the quick-start guide, see the root [README.md](../README.md).

---

## Table of Contents

- [Quick Start](#quick-start)
- [Configuration](#configuration)
  - [Required values](#required-values)
  - [Local request optimizations](#local-request-optimizations)
  - [Reasoning models](#reasoning-models)
- [Shell integration](#shell-integration)
- [Statusline](#statusline)
- [MCP Support](#mcp-support)
- [Testing](#testing)
- [Observability](#observability)
- [Development](#development)
- [Reference docs](#reference-docs)
- [License](#license)

---

## Quick Start

The recommended way is the TUI installer:

```bash
./install.sh
```

It walks you through creating a virtual environment, installing dependencies, testing your Nebius API key, picking models from a live dropdown, running a smoke test, and optionally adding shell shortcuts and Claude Code statusline.

After installation:

```bash
.venv/bin/python start_proxy.py
claude --proxy        # or just 'claudius' if shell shortcuts were added
```

Manual install (if you skip the TUI):

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
# Edit .env: set OPENAI_API_KEY, BIG_MODEL, etc.
.venv/bin/python start_proxy.py
```

## Configuration

### Required values

```bash
OPENAI_API_KEY="your-nebius-api-key"
OPENAI_BASE_URL="https://api.tokenfactory.nebius.com/v1"
```

Common model settings:

```bash
BIG_MODEL="moonshotai/Kimi-K2.6"
MIDDLE_MODEL="moonshotai/Kimi-K2.6"
SMALL_MODEL="moonshotai/Kimi-K2.6"
VISION_MODEL="Qwen/Qwen2.5-VL-72B-Instruct"
STRIP_IMAGE_CONTEXT="true"
```

Verify model availability:

```bash
curl -s https://api.tokenfactory.nebius.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY" | jq '.data[].id'
```

### Local request optimizations

Enabled by default to skip unnecessary Nebius calls for Claude Code housekeeping:

```bash
ENABLE_REQUEST_OPTIMIZATIONS="true"
FAST_PREFIX_DETECTION="true"
ENABLE_NETWORK_PROBE_MOCK="true"
ENABLE_TITLE_GENERATION_SKIP="true"
ENABLE_SUGGESTION_MODE_SKIP="true"
ENABLE_FILEPATH_EXTRACTION_MOCK="true"
```

### Reasoning models

Models that emit hidden reasoning tokens before visible output. Keep `MAX_TOKENS_LIMIT` generous (>=4096; 16k+ recommended for agentic loops):

- `moonshotai/Kimi-K2.6`
- `deepseek-ai/DeepSeek-V3.2`
- `zai-org/GLM-5`
- `Qwen/Qwen3-Next-80B-A3B-Thinking`
- `Qwen/Qwen3-235B-A22B-Thinking-2507-fast`

If a reasoning model returns empty text with a non-zero `output_tokens` count, the budget was exhausted by hidden reasoning — raise the limit and retry.

## Using with Claude Code

Wire the proxy to Claude Code via environment variables. Prefer `ANTHROPIC_AUTH_TOKEN` over `ANTHROPIC_API_KEY` to avoid auth-conflict warnings in Claude Code.

Add to your shell rc (`~/.zshrc`, `~/.bashrc`, or PowerShell profile), then open a new terminal:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8083
export ANTHROPIC_AUTH_TOKEN=claude-local
```

Or as a one-off:

```bash
ANTHROPIC_BASE_URL=http://localhost:8083 ANTHROPIC_AUTH_TOKEN=claude-local claude
```

If `IGNORE_CLIENT_API_KEY=false`, the client token must also match `ANTHROPIC_API_KEY` in the proxy `.env`.

## Shell integration

The project includes a `claude()` shell function that automatically switches between subscription login and proxy mode.

- `claude` — direct (subscription) mode
- `claude --proxy` or `claudius` — proxy (Nebius) mode

See [SHELL_FUNCTION.md](SHELL_FUNCTION.md) for the full function code. The TUI installer can add this to your shell profile automatically.

## Statusline

Claude Code displays the model *it requested*, not the backend model the proxy actually served. A custom statusline fixes this by querying the proxy's live context-usage endpoint.

Add to `~/.claude/settings.json`:

```json
{
  "statusLine": {
    "type": "command",
    "command": "[ -z \"$ANTHROPIC_BASE_URL\" ] && exit 0; base=\"${ANTHROPIC_BASE_URL%/}\"; cfg=$(curl -fsS --max-time 1 \"$base/api/observability/config\" 2>/dev/null || true); model=$(printf '%s' \"$cfg\" | python3 -c 'import json,sys; d=json.load(sys.stdin); print((d.get(\"configured_models\") or {}).get(\"big\") or \"\")' 2>/dev/null || true); ctx=$(curl -fsS --max-time 1 \"$base/api/observability/context-usage\" 2>/dev/null || true); free=$(printf '%s' \"$ctx\" | python3 -c 'import json,sys; d=json.load(sys.stdin); r=d.get(\"remaining_tokens\",1048576) or 1048576; t=d.get(\"context_limit\",1048576) or 1048576; print(f\"{round((r/t)*100)}\")' 2>/dev/null || true); if [ -n \"$model\" ]; then if [ -n \"$free\" ] && [[ \"$free\" =~ ^[0-9]+$ ]]; then if [ \"$free\" -le 20 ]; then c=\"\\033[31m\"; elif [ \"$free\" -le 40 ]; then c=\"\\033[38;5;208m\"; elif [ \"$free\" -le 50 ]; then c=\"\\033[33m\"; else c=\"\\033[32m\"; fi; e=\"\\033[0m\"; echo \"[nebius://$model $c${free}% free$e] $base/dashboard\"; else echo \"[nebius://$model] $base/dashboard\"; fi; else echo \"[proxy://$base]\"; fi"
  }
}
```

**Behaviour:**

- Bare `claude` → statusline is blank, no clutter.
- Proxy-routed + data available → `[nebius://Kimi-K2.6 96% free] http://localhost:8083/dashboard`.
- Proxy-routed + no data yet → `[nebius://Kimi-K2.6] http://localhost:8083/dashboard`.
- Proxy unreachable → `[proxy://<base>]`.

The TUI installer can add this automatically and safely merges into existing `settings.json` files.

## MCP Support

Bundled MCP servers live under `MCP/`:

- `MCP/macoscontrol-mcp` — local macOS screen control

The project-level `.mcp.json` uses repo-relative paths so it works from a fresh clone.

Full guide: [MCP_SERVER_GUIDE.md](MCP_SERVER_GUIDE.md)

## Testing

```bash
pytest -q                          # full suite
pytest -q tests/test_request_converter.py tests/test_response_converter.py
pytest -q tests/test_image_routing.py
RUN_PROXY_INTEGRATION_TESTS=1 pytest -q tests/test_main.py
```

## Observability

Embedded dashboard at `http://localhost:8083/dashboard`. Tracks:

- Model routing, token usage, estimated cost
- Latency, failures, tool calls
- SQLite-backed persistence (Docker: `./data/observability.sqlite3`)

Full notes: [OBSERVABILITY.md](OBSERVABILITY.md)

## Development

```bash
uv run black src tests
uv run isort src tests
uv run mypy src
```

## Reference docs

| Doc | Topic |
|-----|-------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System flow, request conversion, file map |
| [TOOL_CALL_FORMAT.md](TOOL_CALL_FORMAT.md) | Claude SSE tool-call streaming |
| [MCP_SERVER_GUIDE.md](MCP_SERVER_GUIDE.md) | MCP compatibility and usage |
| [OBSERVABILITY.md](OBSERVABILITY.md) | Dashboard config, persistence |
| [SHELL_FUNCTION.md](SHELL_FUNCTION.md) | Shell shortcut reference |
| [BINARY_PACKAGING.md](BINARY_PACKAGING.md) | Standalone binary notes |

## License

MIT
