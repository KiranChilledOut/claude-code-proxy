# Claude Code Proxy for Nebius

This repository contains a Nebius-focused Claude API proxy plus bundled MCP servers for local tool integration.

The proxy accepts Claude-compatible requests from Claude Code, translates them into OpenAI-compatible requests for Nebius-backed models, and converts responses back into Claude format. The repository also includes bundled MCP servers under `MCP/`.

## Table of Contents

- [Repository Layout](#repository-layout)
- [Features](#features)
- [Quick Start](#quick-start)
- [MCP Support](#mcp-support)
- [Testing](#testing)
- [Observability](#observability)
- [Development](#development)
- [Documentation](#documentation)
- [Scope](#scope)
- [License](#license)

## Repository Layout

```text
claude-code-proxy/
├── src/                      # Proxy implementation
├── tests/                    # Automated tests
├── docs/                     # Architecture and integration docs
├── MCP/                      # Bundled MCP servers
├── scripts/                  # Developer utilities
├── start_proxy.py            # Local convenience launcher
├── .mcp.json                 # Project-level Claude Code MCP config
├── pyproject.toml            # Python package metadata
└── README.md
```

## Features

- Claude `/v1/messages` proxying to Nebius OpenAI-compatible endpoints
- Claude-to-OpenAI request conversion and OpenAI-to-Claude response conversion
- Streaming SSE support
- Schema-less Claude Code tool conversion
- Image-aware routing to a vision model
- Bundled MCP support with repo-relative launchers
- Deterministic prefix-cache discipline for vLLM/SGLang KV reuse on Nebius
- Anthropic-compatible `/v1/messages/count_tokens` (counts tools too)
- Pair-aware context auto-truncation (never orphans tool_results)
- Tool-call JSON repair (trailing commas, unescaped newlines) and duplicate
  tool-call dedup for open models — always on, no configuration needed

## Quick Start

### Prerequisites

- Python 3.9+
- Claude Code
- Nebius API credentials
- `uv` optional but recommended

### Install

Using `uv`:

```bash
uv sync
```

Using `pip`:

```bash
python -m pip install -e ".[dev]"
```

### Configure

```bash
cp .env.example .env
```

Required values:

```bash
OPENAI_API_KEY="your-nebius-api-key"
OPENAI_BASE_URL="https://api.tokenfactory.nebius.com/v1"
```

Common model settings:

```bash
BIG_MODEL="zai-org/GLM-4.7-FP8"
MIDDLE_MODEL="zai-org/GLM-4.7-FP8"
SMALL_MODEL="zai-org/GLM-4.7-FP8"
VISION_MODEL="Qwen/Qwen2.5-VL-72B-Instruct"
STRIP_IMAGE_CONTEXT="true"
```

#### Reasoning models

Several Nebius-hosted models emit *hidden* reasoning tokens before producing
visible output. These tokens count against `max_tokens`, so very small budgets
can return empty content. Known reasoning-style models on Nebius:

- `moonshotai/Kimi-K2.5`
- `deepseek-ai/DeepSeek-V3.2`
- `zai-org/GLM-5`
- `Qwen/Qwen3-Next-80B-A3B-Thinking`
- `Qwen/Qwen3-235B-A22B-Thinking-2507-fast`

Implication: keep `MAX_TOKENS_LIMIT` and per-request `max_tokens` generous
(>=4096 is recommended; 16k+ is safer for agentic tool-use loops). If a
reasoning model returns empty text with a non-zero `output_tokens` count, the
budget was exhausted by reasoning before any visible output was produced —
raise the limit and retry.

Verify model availability and pick alternatives at:

```bash
curl -s https://api.tokenfactory.nebius.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY" | jq '.data[].id'
```

### Run

```bash
python start_proxy.py
```

Or:

```bash
uv run claude-code-proxy-nebius
```

### Use with Claude Code

Claude Code talks to the proxy via two environment variables:
`ANTHROPIC_BASE_URL` (where to send requests) and `ANTHROPIC_API_KEY`
(by default, the proxy ignores the client key and accepts any non-empty
string).

To wire this up permanently, add the following to your shell rc
(`~/.zshrc` or `~/.bashrc`), then open a new terminal:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8083
export ANTHROPIC_API_KEY=claude-local
```

Or run as a one-off, prefixing the env vars on the command line:

```bash
ANTHROPIC_BASE_URL=http://localhost:8083 ANTHROPIC_API_KEY=claude-local claude
```

If `IGNORE_CLIENT_API_KEY=false`, the client key must match `ANTHROPIC_API_KEY`.

#### Statusline indicator (optional)

Claude Code displays the model *it requested* (e.g. `claude-sonnet-4-5`),
not the backend model the proxy actually served (e.g. `moonshotai/Kimi-K2.5`),
so by default there is no in-UI indicator that you're routed through this
proxy. A custom statusline fixes that. Add to `~/.claude/settings.json`:

```json
{
  "statusLine": {
    "type": "command",
    "command": "[ -z \"$ANTHROPIC_BASE_URL\" ] && exit 0; env_file=\"/Users/kiran/Desktop/git/claude-code-proxy/.env\"; model=$(grep -m1 '^BIG_MODEL=' \"$env_file\" 2>/dev/null | cut -d= -f2-); obs=$(grep -m1 '^OBSERVABILITY_ENABLED=' \"$env_file\" 2>/dev/null | cut -d= -f2-); port=$(grep -m1 '^PORT=' \"$env_file\" 2>/dev/null | cut -d= -f2-); port=${port:-8083}; if [ \"$obs\" = \"true\" ] && [ -n \"$model\" ]; then echo \"[nebius://$model] http://localhost:$port/dashboard\"; elif [ -n \"$model\" ]; then echo \"[nebius://$model]\"; else echo \"[proxy://$ANTHROPIC_BASE_URL]\"; fi"
  }
}
```

Replace `/path/to/claude-code-proxy/.env` with the absolute path to your
`.env`. Behavior:

- Bare `claude` (no proxy) → statusline is blank, no clutter.
- Proxy-routed + observability enabled → statusline shows e.g. `[nebius://MiniMax-M2.5] http://localhost:8083/dashboard`.
- Proxy-routed + observability disabled → statusline shows e.g. `[nebius://MiniMax-M2.5]`.
- If the `.env` path is unreadable → falls back to `[proxy://<ANTHROPIC_BASE_URL>]`
  so you still know an interceptor is active.

The command is read at session start, so re-open Claude Code after editing
`settings.json`.

## MCP Support

Bundled MCP servers live under `MCP/`.

Current MCPs:

- `MCP/macoscontrol-mcp`: local macOS screen-control MCP

The project-level `.mcp.json` is checked in with repo-relative paths so the bundled MCP can be launched from a fresh clone without machine-specific absolute paths.

## Testing

Run the full suite:

```bash
pytest -q
```

Useful targeted runs:

```bash
pytest -q tests/test_request_converter.py tests/test_response_converter.py
pytest -q tests/test_image_routing.py
RUN_PROXY_INTEGRATION_TESTS=1 pytest -q tests/test_main.py
```

## Observability

The proxy serves a local dashboard at:

```bash
http://localhost:8083/dashboard
```

It tracks configured provider/model routing, token usage, estimated cost from
`MODEL_PRICES_JSON`, latency, failures, and tool calls. Docker Compose persists
the dashboard database under `./data/observability.sqlite3`.

## Development

Common commands:

```bash
uv run black src tests
uv run isort src tests
uv run mypy src
```

## Documentation

Tracked project documentation lives in `docs/`:

- [docs/README.md](./docs/README.md)
- [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md)
- [docs/TOOL_CALL_FORMAT.md](./docs/TOOL_CALL_FORMAT.md)
- [docs/MCP_SERVER_GUIDE.md](./docs/MCP_SERVER_GUIDE.md)
- [docs/OBSERVABILITY.md](./docs/OBSERVABILITY.md)
- [docs/GLM_QUIRKS.md](./docs/GLM_QUIRKS.md)
- [docs/BUGS_FIXED.md](./docs/BUGS_FIXED.md)
- [docs/BINARY_PACKAGING.md](./docs/BINARY_PACKAGING.md)

## Scope

This project is designed and tested specifically for Nebius token factory infrastructure. The current proxy behavior, defaults, and troubleshooting guidance are Nebius-centric rather than provider-agnostic.

## License

MIT
