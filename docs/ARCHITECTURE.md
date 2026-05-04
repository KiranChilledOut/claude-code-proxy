# Architecture

## Overview

This project exposes a Claude-compatible API surface for Claude Code and forwards requests to Nebius-hosted OpenAI-compatible models.

It also bundles MCP servers under `MCP/` so Claude Code can use local tools alongside the proxy.

## High-Level Flow

```text
Claude Code
  ├─ Claude API request -> Proxy (`POST /v1/messages`)
  └─ MCP stdio -> Bundled MCP servers in `MCP/`

Proxy
  ├─ request conversion: Claude -> OpenAI-compatible payload
  ├─ model routing: text vs vision / small vs medium vs large
  └─ response conversion: OpenAI SSE -> Claude SSE

Nebius
  └─ OpenAI-compatible inference endpoint
```

## Key Files

| Path | Purpose |
| --- | --- |
| `src/main.py` | FastAPI entry point |
| `src/api/endpoints.py` | HTTP route handling |
| `src/core/config.py` | environment-driven config |
| `src/core/model_manager.py` | model selection and routing |
| `src/conversion/request_converter.py` | Claude request -> OpenAI request |
| `src/conversion/response_converter.py` | OpenAI response -> Claude SSE |
| `src/conversion/computer_use.py` | schema-less tool conversion |
| `MCP/macoscontrol-mcp/server.py` | bundled macOS control MCP |
| `start_proxy.py` | local convenience launcher |

## Model Configuration

Core environment variables:

```bash
OPENAI_API_KEY=<nebius-key>
OPENAI_BASE_URL=https://api.tokenfactory.nebius.com/v1
BIG_MODEL=moonshotai/Kimi-K2.5
MIDDLE_MODEL=zai-org/GLM-5
SMALL_MODEL=google/gemma-3-27b-it
VISION_MODEL=Qwen/Qwen2.5-VL-72B-Instruct
```

### `/model` aliases

Inside Claude Code, users can type `/model <alias>` to switch upstream models without restarting the proxy or editing `.env`. `ModelManager` resolves the alias on each request, so the choice is stateless from the proxy's perspective.

| Alias | Default upstream | Override env var |
| --- | --- | --- |
| `glm` | `zai-org/GLM-5` | `GLM_MODEL` |
| `kimi` | `moonshotai/Kimi-K2.5` | `KIMI_MODEL` |
| `gemma` | `google/gemma-3-27b-it` | `GEMMA_MODEL` |
| `qwen` | `Qwen/Qwen3.5-397B-A17B` | `QWEN_MODEL` |
| `nemotron` | `nvidia/Llama-3_1-Nemotron-Ultra-253B-v1` | `NEMOTRON_MODEL` |
| `super` | `nvidia/nemotron-3-super-120b-a12b` | `NEMOTRON_SUPER_MODEL` |
| `nano` | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` | `NEMOTRON_NANO_MODEL` |
| `minimax` | `MiniMaxAI/MiniMax-M2.5` | `MINIMAX_MODEL` |
| `hermes` | `NousResearch/Hermes-4-405B` | `HERMES_MODEL` |
| `gpt` | `openai/gpt-oss-120b` | `GPT_MODEL` |
| `llama` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | `LLAMA_MODEL` |
| `prime` | `PrimeIntellect/INTELLECT-3` | `PRIME_MODEL` |
| `deepseek` | `deepseek-ai/DeepSeek-V3.2` | `DEEPSEEK_MODEL` |

Resolution rules in order:

1. **Native passthrough.** Ids beginning with `gpt-`, `o1-`, `ep-`, `doubao-`, or `deepseek-` pass through verbatim.
2. **Slash-passthrough.** Ids containing a `/` (Token Factory / HF-style, e.g. `meta-llama/Llama-3.3-70B-Instruct`) pass through verbatim. This lets users pick any catalog entry directly.
3. **Exact alias match.** Lookup in the table above.
4. **Keyword match** for `glm` / `kimi` / `gemma` substrings (e.g. `glm-5`, `kimi-2.5`) — resolves via the alias.
5. **Fallback** → `BIG_MODEL`.

### Picker contents (`/v1/models`)

`/v1/models` surfaces three groups, in this order:

1. The short aliases above (`glm`, `kimi`, `gemma`).
2. The full upstream Token Factory catalog, fetched from `{OPENAI_BASE_URL}/v1/models` and cached at module level for `MODELS_CACHE_TTL_SECONDS` (default 600). On upstream error the listing degrades to whatever was cached, or to just the aliases if nothing has been cached yet.
3. Any extra ids from `BIG_MODEL` / `MIDDLE_MODEL` / `SMALL_MODEL` / `VISION_MODEL` that the upstream catalog didn't already include.

Claude Code's built-in `/model` picker is hardcoded and only enumerates its native model entries plus the *currently-selected* custom model (looked up by id from this listing). For an actual picker UX over the full catalog, install the bundled `/models` custom slash command from `scripts/claude-code/` (see `scripts/claude-code/README.md`). It shows a numbered list (curated shortcuts + any live catalog extras) and writes the choice to `~/.claude/settings.local.json`.

To use any catalog id directly, type it as `/model meta-llama/Llama-3.3-70B-Instruct` etc. — Claude Code accepts arbitrary `--model` strings and the proxy's slash-passthrough routes them verbatim.

## Request Lifecycle

1. Claude Code sends a Claude-compatible request to `/v1/messages`.
2. `request_converter.py` maps the request into OpenAI chat-completions format.
3. Schema-less Claude Code tools are converted into explicit JSON-schema tools.
4. The request is sent to the configured Nebius endpoint.
5. OpenAI-format streaming chunks are received.
6. `response_converter.py` converts them into Claude SSE events.
7. Claude Code receives a native Claude-style response stream.
