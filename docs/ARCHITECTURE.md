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
BIG_MODEL=zai-org/GLM-4.5
MIDDLE_MODEL=zai-org/GLM-4.5
SMALL_MODEL=zai-org/GLM-4.5
VISION_MODEL=Qwen/Qwen2.5-VL-72B-Instruct
```

## Request Lifecycle

1. Claude Code sends a Claude-compatible request to `/v1/messages`.
2. `request_converter.py` maps the request into OpenAI chat-completions format.
3. Schema-less Claude Code tools are converted into explicit JSON-schema tools.
4. The request is sent to the configured Nebius endpoint.
5. OpenAI-format streaming chunks are received.
6. `response_converter.py` converts them into Claude SSE events.
7. Claude Code receives a native Claude-style response stream.
