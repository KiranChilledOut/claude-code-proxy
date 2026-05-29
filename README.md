# Claude Code Proxy for Nebius

A Claude Code → Nebius bridge. Accepts Claude-Code requests, converts to OpenAI-compatible requests, and converts responses back.

## Quick Start (the easy way)

One step. Run the installer:

```bash
./install.sh
```

This launches a step-by-step TUI that:
1. Checks prerequisites
2. Creates a virtual environment
3. Installs dependencies
4. Tests your Nebius API key
5. Lets you pick models from live dropdowns
6. Writes your `.env` file
7. Runs a smoke test
8. Optionally adds `claude` / `claudius` shell shortcuts
9. Optionally configures Claude Code statusline

After it finishes:

```bash
claude --proxy        # or just 'claudius' if you added the shell shortcuts
# OR
claudius
# OR
.venv/bin/python start_proxy.py
```

Then open http://localhost:8083/dashboard for the observability dashboard.

## Prerequisites

- Python 3.9+
- Nebius API key (from https://nebius.com)
- Claude Code (optional; install from Anthropic)

## Quick Start (manual)

If you prefer not to use the TUI:

```bash
# 1. Clone and enter directory
cd claude-code-proxy

# 2. Create venv & install deps
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env: set OPENAI_API_KEY, BIG_MODEL, etc.

# 4. Run
.venv/bin/python start_proxy.py

# 5. Use
ANTHROPIC_BASE_URL=http://localhost:8083 ANTHROPIC_AUTH_TOKEN=claude-local claude
```

## Observability

Open http://localhost:8083/dashboard for usage, latency, cost, and model routing info.

## Documentation

For full configuration options, model details, architecture, troubleshooting, and the complete feature list, see **[docs/README.md](docs/README.md)**.

| Doc | What's inside |
|-----|-------------|
| [docs/README.md](docs/README.md) | Full reference index — everything |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System flow, key files, design |
| [docs/TOOL_CALL_FORMAT.md](docs/TOOL_CALL_FORMAT.md) | Claude SSE tool-call streaming |
| [docs/OBSERVABILITY.md](docs/OBSERVABILITY.md) | Dashboard configuration |
| [docs/SHELL_FUNCTION.md](docs/SHELL_FUNCTION.md) | Shell shortcut reference |
| [docs/BINARY_PACKAGING.md](docs/BINARY_PACKAGING.md) | Standalone binary notes |

## Features (high-level)

- Claude `/v1/messages` proxying to Nebius OpenAI-compatible endpoints
- Streaming SSE
- Automatic model routing (big / middle / small / vision)
- Built-in request optimizations for Claude Code housekeeping
- Tool-call JSON repair and deduplication
- Context auto-truncation (never orphans tool results)
- Observability dashboard with cost tracking

## Scope

Designed for Nebius token factory infrastructure. Defaults and troubleshooting guidance are Nebius-centric rather than provider-agnostic.

## License

MIT
