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

## Quick Start

### Prerequisites

- Python 3.9+
- Claude Code
- Nebius API credentials
- `uv` optional but recommended

### Install

The fastest path is the bundled installer, which creates a virtualenv,
installs runtime dependencies, prompts for your Nebius API key without
echoing it to your shell, validates the configured model IDs against
`GET /v1/models`, and runs an end-to-end smoke test before declaring
success:

```bash
./install.sh
```

#### Manual install

If you'd rather wire things up yourself:

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip   # fresh venvs ship pip <22 which fails on pyproject editable installs
.venv/bin/pip install -r requirements.txt
cp .env.example .env                            # then edit .env to set OPENAI_API_KEY
```

Or with `uv`:

```bash
uv sync
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

There are two natural ways to wire this up. Pick one and append it to
your shell rc (`~/.zshrc` or `~/.bashrc`), then open a new terminal.

#### Option A — `claudius` wrapper function

A shell function that prefixes the env vars only when you call it. Bare
`claude` keeps using Anthropic's API.

```bash
claudius() {
  ANTHROPIC_BASE_URL=http://localhost:8083 ANTHROPIC_API_KEY=claude-local claude "$@"
}
```

Then run: `claudius`

#### Option B — global exports

Every `claude` invocation routes through the proxy.

```bash
export ANTHROPIC_BASE_URL=http://localhost:8083
export ANTHROPIC_API_KEY=claude-local
```

Then run: `claude`

Option A keeps the provider visible in the command name. Option B is
more convenient if you mostly use Claude Code through Nebius.

The bundled `install.sh` can write either snippet for you at the end of
install.

If `IGNORE_CLIENT_API_KEY=false`, the client key must match `ANTHROPIC_API_KEY`.

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
