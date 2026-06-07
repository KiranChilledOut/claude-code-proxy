# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Convention:** every change-worthy PR adds an entry under `[Unreleased]` in the
> matching group (`Added` / `Changed` / `Fixed` / `Removed` / `Deprecated` / `Security`).
> On release, rename `[Unreleased]` to the new version + date and start a fresh
> `[Unreleased]` block.

## [Unreleased]

### Added
- Server-side web search (Tavily). When `TAVILY_API_KEY` is set, the proxy
  executes `web_search`/`WebSearch` tool calls itself in a bounded loop and
  feeds results back to the model, returning the final answer. Fixes "Did 0
  searches" — Claude Code's search can't run behind a non-Anthropic backend.
  Search tools are also forwarded with a real `{query}` schema. Only engaged
  when a search tool is present; all other requests are unchanged. Knobs:
  `SERVER_SEARCH_ENABLED`, `TAVILY_MAX_RESULTS`, `SERVER_SEARCH_MAX_ITERS`.
- Honor `thinking.display` for adaptive thinking (Opus 4.7/4.8). Thinking text
  is surfaced only when `display` is `"summarized"`; adaptive mode defaults to
  `"omitted"` (matching Anthropic), so the backend's reasoning is no longer
  shown unless asked for. Operator override via `THINKING_DISPLAY_OVERRIDE`.
- Strip `<think>…</think>` from visible text (streaming + non-streaming) whenever
  thinking is not being surfaced, so provider reasoning never leaks as assistant
  text. Fixes the "reasoning visible in output" community reports.
- Dynamic effort forwarding: the effort chosen in Claude Code (`/effort`,
  carried in `output_config.effort`) is automatically mapped to a backend
  `reasoning_effort` (xhigh/max -> high) — no configuration. If a backend
  rejects `reasoning_effort`, the proxy strips it, retries once, and remembers
  not to send it to that model again (self-healing, no repeated latency).
- Surface a model's separate reasoning channel (`reasoning_content` / `reasoning`,
  as emitted by DeepSeek-R1, Qwen, GLM-thinking, etc.) as Claude `thinking`
  content blocks — both streaming and non-streaming.
- Accept inbound `thinking` / `redacted_thinking` blocks in assistant history
  (interleaved thinking). They are parsed without error and dropped during
  conversion, since OpenAI-compatible backends cannot consume them.
- Opt-in reasoning passthrough so reasoning-capable backends actually think:
  `REASONING_EFFORT` (operator override) and `MAP_THINKING_BUDGET_TO_EFFORT`
  (bucket a client `thinking.budget_tokens` into an effort level). Both default
  to no-op, so non-reasoning backends are unaffected.
- `docs/GAP_ANALYSIS_SPEC.md` — gap analysis vs. the current Claude Code CLI and
  a roadmap for agentic/harness work.
- `tests/test_thinking_and_reasoning.py` — unit coverage for the thinking,
  reasoning, and stop-reason changes.

### Fixed
- Server-search path no longer drops tool calls. The non-streamed loop result
  was being streamed back via a text-only serializer, so on turns that merely
  *offered* `WebSearch` the model's real tool call (Read/Edit/Agent/...) was
  silently dropped -> "The model's tool call could not be parsed." Added a
  tool_use/thinking-aware SSE serializer (`claude_response_to_sse`) for that path.
- Self-heal on upstream context-length 400s: the proxy now detects when a
  backend rejects a request for exceeding its context window, sheds the oldest
  messages with the existing safe trimmer (system prompt, latest turn, and tool
  pairs preserved) and retries once, instead of surfacing a hard error. If it
  still doesn't fit, a clear message points at `<ROLE>_MODEL_CONTEXT_LIMIT`.
- Strip Kimi-K2's native tool-call control tokens (`<|tool_call_begin|>`,
  `<|tool_call_argument_begin|>`, `functions.NAME:N`, ...) that leak into tool
  arguments when a tool is forwarded without a real parameter schema (e.g. the
  Anthropic `web_search` server tool). The inner JSON is now extracted so the
  client receives clean arguments instead of a token blob.
- Extended-thinking config now understands Anthropic's real wire shape
  `{"type": "enabled"|"disabled", "budget_tokens": N}` (via `is_enabled()`),
  in addition to the legacy `{"enabled": bool}`. Previously `{"type":"disabled"}`
  was ignored and thinking stayed on, and `budget_tokens` was dropped.
- Requests whose assistant history contained `thinking` blocks no longer return
  HTTP 422.
- `thinking.type` accepts any mode string (e.g. `adaptive`), not just
  `enabled`/`disabled`. A strict enum was 422'ing real Claude Code requests that
  send newer thinking modes; only `disabled` turns thinking off.
- A provider `content_filter` finish reason now maps to the Claude `refusal`
  stop reason instead of masquerading as `end_turn`.

### Changed
- Added `refusal`, `pause_turn`, and `model_context_window_exceeded` stop-reason
  constants (only `refusal` is emitted today; the others are reserved for
  server-tool / upstream-error wiring).

### Removed
- Reverted an experimental 1M-context `betas` override. Context window size
  remains owned solely by the per-model `*_MODEL_CONTEXT_LIMIT` settings, which
  are also what the statusline's `/api/observability/context-usage` endpoint
  reads (capped at 200K to match Claude Code). To run a 1M-capable model, set
  its `*_MODEL_CONTEXT_LIMIT` accordingly.

## [1.0.0]

- Initial baseline: Claude `/v1/messages` proxy to Nebius OpenAI-compatible
  endpoints, with streaming SSE, model routing (big/middle/small/vision),
  tool-call JSON repair, local request optimizations, and the observability
  dashboard. (Pre-changelog history is in the git log.)
