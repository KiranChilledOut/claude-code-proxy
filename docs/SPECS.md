# Spec: Codex Proxy Integration

## Overview

Add OpenAI Responses API (`POST /v1/responses`) support to the existing Python-based Claude-to-OpenAI proxy. This enables Codex CLI and other Responses API clients to connect through the same proxy infrastructure.

## Motivation

The current proxy only exposes Anthropic's `/v1/messages` endpoint. Codex CLI uses the OpenAI Responses API (`/v1/responses`) which has a different request/response format, tool conventions, and streaming event model. Source reference: CCX (`/Users/kiran/Desktop/git/ccx`) implements this in Go; we port the protocol layer to Python and integrate with our existing OpenAI client, observability, and request-optimization stack.

## Scope

1. New `POST /v1/responses` endpoint (streaming + non-streaming)
2. Request parsing via Pydantic models
3. Request conversion: Responses API → OpenAI Chat Completions
4. Response conversion: OpenAI Chat Completions → Responses API
5. Streaming conversion: OpenAI SSE → Responses API SSE
6. Codex-specific tool compatibility (string tools, custom tools, namespace tools)
7. Session management for `previous_response_id`
8. Auth, observability, cancellation, retry reuse
9. WebSocket fallback handling (`426 Upgrade Required`)
10. Tests and documentation

## Out of Scope

- Multi-provider routing (Gemini, Claude native upstream); upstream remains OpenAI-compatible only
- Persistent session storage (in-memory with TTL is sufficient)
- Full passthrough to an upstream `/v1/responses` endpoint (we convert to Chat Completions)

## Architecture

```
+--------+   POST /v1/responses   +------------------+   OpenAI Chat   +----------+
| Client |----------------------->|  Proxy (FastAPI) |--------------->| Nebius   |
| (Codex)|  (Responses API format)|                  |  Completions  | (OpenAI  |
+--------+                        |  - Auth           |               | compat)  |
   ^                              |  - Session        |               +----------+
   |  SSE stream                  |  - Conversion     |                    |
   |  (Responses events)         |  - Observability  |<-------------------+
   +------------------------------+  - Retry/Cancel   |   SSE (OpenAI format)
                                  +------------------+
                                          |
                                          v
                                   +--------------+
                                   | Tavily search|
                                   | (optional)   |
                                   +--------------+
```

## Data Models

### `ResponsesRequest`

```python
class ResponsesRequest(BaseModel):
    model: str
    instructions: Optional[str] = None  # system prompt
    input: Union[str, List[ResponsesItem]]  # user input
    previous_response_id: Optional[str] = None
    store: Optional[bool] = None
    max_output_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: bool = False
    tools: Optional[List[Union[str, Dict[str, Any]]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = None
    reasoning: Optional[Dict[str, Any]] = None  # {"effort": "low" | "medium" | "high"}
    user: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
```

### `ResponsesItem`

```python
class ResponsesItem(BaseModel):
    type: str  # "message", "function_call", "function_call_output", "text", "image"
    id: Optional[str] = None
    role: Optional[str] = None      # "user" | "assistant"
    status: Optional[str] = None    # "in_progress" | "completed"
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    call_id: Optional[str] = None
    name: Optional[str] = None
    namespace: Optional[str] = None
    arguments: Optional[str] = None   # JSON string
    output: Optional[Any] = None
```

### `ResponsesResponse`

```python
class ResponsesResponse(BaseModel):
    id: str
    model: str
    output: List[ResponsesItem]
    status: str   # "completed" | "failed" | "in_progress"
    previous_id: Optional[str] = None
    usage: ResponsesUsage
```

### `ResponsesUsage`

```python
class ResponsesUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_tokens_details: Optional[InputTokensDetails] = None
    output_tokens_details: Optional[OutputTokensDetails] = None
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None
```

### `ResponsesStreamEvent`

```python
class ResponsesStreamEvent(BaseModel):
    type: str  # "response.created", "response.in_progress",
               # "response.output_text.delta", "response.output_item.added",
               # "response.function_call_arguments.delta", "response.completed"
    id: Optional[str] = None
    model: Optional[str] = None
    output_index: Optional[int] = None
    item: Optional[ResponsesItem] = None
    delta: Optional[str] = None   # for deltas
    usage: Optional[ResponsesUsage] = None  # on completed
```

## Request Conversion Pipeline

### Step 1: Parse and Validate
- Parse JSON via `ResponsesRequest` Pydantic model
- Extract `previous_response_id` → look up session history
- Resolve session: merge previous input/output items into request context

### Step 2: Tool Compatibility
- If `tools` present, run through `CodexToolContext`:
  - String tools: `"exec_command"` → proxy function with generic description
  - Custom tools: `{"type":"custom","name":"apply_patch"}` → proxy function prefixed/suffixed
  - Namespace tools: `{"type":"namespace","name":"mcp__","tools":[...]}` → flatten names to `mcp__tool_name`
  - Built-in: `web_search`, `local_shell`, `computer_use` → passthrough or strip
- `tool_choice` conversion:
  - `"required"` → `"required"`
  - `"auto"` → `"auto"`
  - `"none"` → `"none"`
  - Object with namespace → flatten to function name
  - Object with custom → map to proxy function name

### Step 3: Build OpenAI Chat Request

```python
{
    "model": map_codex_model(request.model),  # "gpt-4" -> big_model, "mini" -> small_model
    "messages": [
        # system from instructions
        {"role": "system", "content": request.instructions},
        # input items converted
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "...", "tool_calls": [...]},
        {"role": "tool", "content": "...", "tool_call_id": "..."},
    ],
    "tools": [...],  # from CodexToolContext
    "tool_choice": "...",
    "max_tokens": request.max_output_tokens,
    "temperature": request.temperature,
    "top_p": request.top_p,
    "stream": request.stream,
    "stream_options": {"include_usage": True},  # if streaming
    "reasoning_effort": map_reasoning_effort(request.reasoning),  # optional
    "user": request.user,
}
```

### Input Item → Message Mapping

| Responses Item | OpenAI Message |
|---|---|
| `{"type":"message","role":"user","content":"..."}` | `{"role":"user","content":"..."}` |
| `{"type":"message","role":"assistant","content":"..."}` | `{"role":"assistant","content":"..."}` |
| `{"type":"function_call","call_id":"fc_1","name":"...","arguments":"..."}` | `{"role":"assistant","tool_calls":[{"id":"fc_1","type":"function","function":{"name":"...","arguments":"..."}}]}` |
| `{"type":"function_call_output","call_id":"fc_1","output":"..."}` | `{"role":"tool","content":"...","tool_call_id":"fc_1"}` |
| `{"type":"text","text":"..."}` | `{"role":"user","content":"..."}` |

## Response Conversion Pipeline

### Non-Streaming

1. Parse OpenAI `choices[0].message`
2. Build `output` items:
   - Text content → `ResponsesItem(type="message", role="assistant", content=text)`
   - Tool calls → `ResponsesItem(type="function_call", call_id=..., name=..., arguments=...)` for each
3. If tool calls are proxy functions for Codex custom tools, remap back:
   - Proxy `apply_patch_add_file` → back to custom tool name
   - Proxy `mcp__tool_name` → back to namespace + name
4. Map `usage`:
   - `prompt_tokens` → `input_tokens`
   - `completion_tokens` → `output_tokens`
   - `total_tokens` → `total_tokens`
   - `prompt_tokens_details.cached_tokens` → `input_tokens_details.cached_tokens`
   - `completion_tokens_details.reasoning_tokens` → `output_tokens_details.reasoning_tokens`
5. Set `status = "completed"`, generate `id` and `previous_id`

### Streaming

Use a state machine (`CodexStreamState`) per request:

```python
@dataclass
class CodexStreamState:
    response_id: str
    created_at: int
    seq: int = 0
    text_buf: str = ""
    func_args_buf: Dict[int, str] = field(default_factory=dict)
    func_names: Dict[int, str] = field(default_factory=dict)
    func_call_ids: Dict[int, str] = field(default_factory=dict)
    func_item_added: Dict[int, bool] = field(default_factory=dict)
    reasoning_active: bool = False
    reasoning_buf: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    first_chunk: bool = True
```

Event emission per OpenAI SSE delta:

| OpenAI SSE Event | Codex SSE Events |
|---|---|
| First chunk | `response.created` |
| Every chunk | `response.in_progress` |
| `choices[0].delta.content` text | `response.output_text.delta` |
| `choices[0].delta.tool_calls[i]` first appearance | `response.output_item.added` (function_call item) |
| `choices[0].delta.tool_calls[i].function.arguments` | `response.function_call_arguments.delta` |
| `usage` in last chunk | Included in `response.completed` |
| `[DONE]` | `response.completed` (with accumulated usage) |

**Important:** Never emit `partial_json: "{}"` at tool block start. Wait for the first real argument delta, then emit `output_item.added` + arguments delta.

### Error Responses

On upstream error, emit:

```json
{
  "type": "response.failed",
  "error": {
    "type": "api_error",
    "message": "..."
  }
}
```

## Session Management

### `SessionStore`

In-memory dictionary per response ID:

```python
class SessionStore:
    def __init__(self, ttl_seconds: int = 3600):
        self._store: Dict[str, SessionData] = {}
        self._ttl = ttl_seconds

    def get(self, response_id: str) -> Optional[SessionData]: ...
    def put(self, response_id: str, input_items: List[ResponsesItem], output_items: List[ResponsesItem]): ...
    def evict(self): ...  # remove entries older than TTL
```

### Session Data

```python
@dataclass
class SessionData:
    response_id: str
    previous_id: Optional[str]
    input_items: List[ResponsesItem]   # request.input + tools
    output_items: List[ResponsesItem]  # model output
    created_at: float  # monotonic timestamp
```

### Session ID Extraction

Priority:
1. `previous_response_id` from request body → look up session
2. `metadata.conversation_id` or `metadata.user_id`
3. `X-Claude-Code-Session-Id` header
4. Generate fresh UUID

### Multi-Turn Flow

```
Turn 1:
  Client → POST /v1/responses (no previous_response_id)
  Server → generate resp_1 → store (input_1, output_1) under resp_1

Turn 2:
  Client → POST /v1/responses (previous_response_id = resp_1)
  Server → load session → prepend input_1 + output_1 to messages
         → generate resp_2 → store under resp_2
```

## Tool Compatibility Details

### `CodexToolContext`

Parsed from `tools` array:

```python
@dataclass
class CodexToolContext:
    custom_tools: Dict[str, CodexCustomToolSpec]      # name -> spec (name, type, description)
    function_tools: Dict[str, CodexFunctionToolSpec]  # name -> spec (namespace, name, description, parameters)
    has_custom_tools: bool
    has_namespace_tools: bool
```

### Conversion Rules

1. **String tools** (`"exec_command"`):
   - Proxy function name: same string
   - Description: `FREEFORM custom tool`
   - Parameters: single `"input"` (string) or generic `{}`

2. **Custom object tools** (`{"type":"custom","name":"apply_patch"}`):
   - Proxy function names: `{name}_add_file`, `{name}_update_file`, `{name}_delete_file`, `{name}_undo_edit`, `{name}_batch`
   - Description derived from tool or generic
   - On return: remap proxy calls back to `custom_tool_call` items with original name

3. **Namespace tools** (`{"type":"namespace","name":"mcp__","tools":[...]}`):
   - Flatten: `mcp__{tool_name}` → proxy function name
   - Parameters from nested tools
   - On return: unflatten `mcp__tool_name` → `{type:"namespace", name:"mcp__", namespace:"..."}`

4. **Built-in tools**:
   - `web_search` → passthrough to Tavily (if `TAVILY_API_KEY` set) or strip
   - `local_shell` → passthrough or strip
   - `computer_use` → passthrough to existing `computer_use.py` converter

### Tool Choice Mapping

```python
def map_tool_choice(tool_choice: Any, ctx: CodexToolContext) -> Any:
    if tool_choice == "required":
        return "required"
    if tool_choice == "auto" or tool_choice == "none":
        return tool_choice
    # Object form
    if isinstance(tool_choice, dict):
        t = tool_choice.get("type")
        name = tool_choice.get("name", "")
        namespace = tool_choice.get("namespace")
        if namespace:
            return {"type": "function", "function": {"name": f"{namespace}{name}"}}
        if ctx.is_custom_tool(name):
            proxy = ctx.proxy_name_for_custom(name)
            return {"type": "function", "function": {"name": proxy}}
        return {"type": "function", "function": {"name": name}}
    return "auto"
```

## Configuration

New environment variables in `src/core/config.py`:

| Variable | Default | Description |
|---|---|---|
| `CODEX_ENABLED` | `true` | Enable `/v1/responses` endpoint |
| `CODEX_TOOL_COMPAT` | `true` | Enable Codex custom/namespace tool conversion |
| `CODEX_SESSION_TTL_SECONDS` | `3600` | Session TTL in seconds |
| `CODEX_WEBSOCKET_FALLBACK` | `true` | Return 426 on WebSocket upgrade attempts |
| `CODEX_MODEL_MAPPING` | — | Optional JSON mapping: `{"gpt":"big","mini":"small"}` |

### Model Mapping

Default mapping from Codex model names to backend:

```python
def map_codex_model(codex_model: str) -> str:
    lower = codex_model.lower()
    if "mini" in lower:
        return config.small_model
    # "gpt-4", "gpt-5", etc. all map to big model
    return config.big_model
```

Optional `CODEX_MODEL_MAPPING` overrides via JSON prefix matching.

## API Routes

### `POST /v1/responses`

Accepts `ResponsesRequest` JSON body. Returns `ResponsesResponse` (non-streaming) or SSE stream (streaming).

Headers:
- `Authorization: Bearer <key>` or `X-Api-Key: <key>` — validated against `OPENAI_API_KEY`
- `Content-Type: application/json`
- `X-Claude-Code-Session-Id` — optional session hint

### `GET /v1/responses`

Codex CLI first attempts WebSocket. Return:
- If `Upgrade: websocket`: `426 Upgrade Required`
- Else: `405 Method Not Allowed`

This forces Codex CLI to fall back to HTTP POST.

### `POST /v1/responses/:id/input_items` (Future)

Append input items to an existing session. Not required for MVP.

## Observability Integration

Reuse existing `_record_message_observability()` pattern in `endpoints.py`:

- `client_type`: `"codex"`
- `endpoint`: `"/v1/responses"`
- `claude_model`: `request.model` (Codex model name)
- `backend_model`: mapped OpenAI model
- `tool_calls`: extracted from response output items

Dashboard already supports custom `client_type` filtering.

## Reuse Matrix

| Component | Reuse? | How |
|---|---|---|
| `OpenAIClient` | Yes | Same `create_chat_completion()` / `create_chat_completion_stream()` APIs |
| `validate_api_key()` | Yes | Same auth middleware |
| `config` | Yes | Add Codex-specific fields |
| Observability recorder | Yes | Add `client_type="codex"` |
| `server_tools` (Tavily) | Yes | Check search tools in Codex request, run same loop |
| `computer_use.py` | Partial | Wire Codex `computer_use` tool to existing converter |
| Request optimizations | Partial | Some optimizations (title gen skip, etc.) apply to Codex too |
| Cancellation | Yes | Use existing `request_id` + `cancel_event` pattern |
| Retry logic | Yes | Same `_maybe_drop_reasoning_effort`, `_maybe_retrim_context` |
| Prefix fingerprinting | No | Not applicable (different request shape) |

## Open Questions

1. **Session persistence across proxy restarts?** In-memory is fine for single-instance. Load-balanced deployments lose sessions on restart, which is acceptable for v1.
2. **Codex built-in `computer_use` passthrough?** The existing proxy already converts Anthropic `computer_use` tools to OpenAI format. We should wire Codex `computer_use` to the same converter.
3. **Pricing integration?** The dashboard already has `MODEL_PRICES_JSON`. Codex model pricing (e.g. `gpt-4`) can reuse entries if the backend model maps correctly.
4. **Context trimming?** Reuse `_trim_messages_to_fit()` from `request_converter.py` for Sessions that grow too large.

## File Plan

```
src/
  codex/
    __init__.py
    models.py              # Pydantic models
    convert_request.py     # ResponsesRequest → OpenAI Chat request
    convert_response.py    # OpenAI Chat response → ResponsesResponse
    convert_stream.py      # OpenAI SSE → Responses API SSE (state machine)
    tools_compat.py        # Codex tool parsing, conversion, remapping
    session.py             # SessionStore (in-memory, TTL)
tests/
  codex/
    test_models.py
    test_convert_request.py
    test_convert_response.py
    test_convert_stream.py
    test_tools_compat.py
    test_session.py
```

Edits to existing files:
- `src/api/endpoints.py` — add `/v1/responses` route
- `src/core/config.py` — add Codex env vars
- `src/core/model_manager.py` — add `map_codex_model()`
- `CHANGELOG.md` — add entry

## Test Strategy

### Unit Tests

- `test_models.py`: Valid/invalid request/response parsing
- `test_convert_request.py`: Input array → messages, instructions→system, tools→functions, reasoning mapping, tool_choice mapping
- `test_convert_response.py`: Text output, tool_calls→function_call items, usage mapping, custom tool remapping
- `test_convert_stream.py`: State machine events in correct order, text buffering, tool argument buffering, reasoning passthrough, usage on completed
- `test_tools_compat.py`: String tool parsing, custom tool parsing, namespace tool parsing, choice mapping, proxy→original remapping
- `test_session.py`: Create, get, TTL eviction, multi-turn prepend

### E2E Test

```bash
# 1. Start proxy
python start_proxy.py &

# 2. Non-streaming health check
curl -X POST http://localhost:8083/v1/responses \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4","input":"Hello","stream":false}'

# 3. Streaming check
curl -N -X POST http://localhost:8083/v1/responses \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4","input":"Count to 5","stream":true}'

# 4. WebSocket fallback
curl -i -H "Upgrade: websocket" http://localhost:8083/v1/responses
# Expected: HTTP/1.1 426 Upgrade Required
```

### Integration Test

Configure Codex CLI:
```bash
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="http://localhost:8083"
codex "Create a hello.py file"
```

## Milestones

1. **Models + Request Conversion** — Parse and validate Responses requests, convert to OpenAI Chat format
2. **Response Conversion** — Convert OpenAI Chat responses back to Responses format
3. **Streaming** — SSE stream conversion with correct event ordering
4. **Tool Compat** — Handle Codex tool formats, namespace, custom tools
5. **Session + Routes** — Wire endpoint, session manager, auth, observability
6. **Tests + Docs** — Unit tests, E2E tests, documentation

## Risks

1. **Event ordering in streaming** — OpenAI SSE tool call deltas interleave text. The state machine must emit `output_item.added` before first `function_call_arguments.delta` and never emit empty `partial_json: {}`.
2. **Tool name collisions** — Custom tool proxy names might collide. Use deterministic prefixing.
3. **Session memory growth** — Unbounded in-memory storage. Add TTL eviction and max session count.
4. **Cancellation mid-stream** — SSE connections are long-lived. Must propagate cancellation to the upstream OpenAIClient correctly.
5. **Reasoning passthrough** — If upstream returns `reasoning_content`, it must be surfaced in Codex response (as `reasoning` item or stripped, depending on effort). Follow existing `response_converter.py` patterns.
