"""Convert an OpenAI Chat Completions SSE stream into Codex Responses API SSE events."""

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, Optional


@dataclass
class CodexStreamState:
    """Mutable state for one Codex stream conversion."""

    response_id: str
    created_at: int  # epoch seconds
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


_RE_THINKING = re.compile(r"<thinking>|<thinking>", re.IGNORECASE)
_RE_THINKING_CLOSE = re.compile(r"</thinking>|</thinking>", re.IGNORECASE)


def _emit_data(payload: Dict[str, Any]) -> str:
    """Format a single SSE data line."""
    return f"data: {json.dumps(payload)}\n\n"


def _make_event(type_: str, **fields: Any) -> str:
    """Build payload and wrap in SSE data: frame."""
    payload: Dict[str, Any] = {"type": type_}
    payload.update(fields)
    return _emit_data(payload)


def _make_response_id() -> str:
    return f"resp_{uuid.uuid4().hex[:16]}"


async def convert_openai_sse_to_responses_sse(
    openai_sse_stream: AsyncGenerator[str, None],
    request_model: str,
    tool_ctx: Any = None,
) -> AsyncGenerator[str, None]:
    """Consume an OpenAI SSE stream and yield Codex Responses SSE events.

    Args:
        openai_sse_stream: Async generator yielding raw SSE lines from OpenAI.
        request_model: The model name requested by the Codex client.
        tool_ctx: Optional tool-context object with ``remap_tool_calls_back()``.

    Yields:
        SSE-formatted strings (Codex Responses event format).
    """
    state = CodexStreamState(
        response_id=_make_response_id(),
        created_at=int(time.time()),
    )

    reasoning_open = False   # <thinking> currently open

    def _remap_name(name: str) -> str:
        if tool_ctx is not None and hasattr(tool_ctx, "remap_tool_calls_back"):
            remapped = tool_ctx.remap_tool_calls_back(name)
            if remapped:
                return remapped
        return name

    # --- Always emit created first ---
    yield _make_event(
        "response.created",
        response=state.response_id,
        model=request_model,
    )

    async for line in openai_sse_stream:
        if not line.strip():
            continue

        # Only process data: lines
        if not line.startswith("data:"):
            continue

        payload_str = line[len("data:"):].strip()

        # Stream terminator
        if payload_str == "[DONE]":
            break

        try:
            chunk = json.loads(payload_str)
        except json.JSONDecodeError:
            continue

        # --- Every valid chunk: emit in_progress ---
        yield _make_event(
            "response.in_progress",
            response=state.response_id,
            output_index=0,
        )

        choices = chunk.get("choices", [])
        if choices:
            choice = choices[0]
            delta = choice.get("delta", {})
            finish_reason = choice.get("finish_reason")

            # --- Handle text delta ---
            content = delta.get("content")
            if content is not None and content != "":
                text = content

                # Strip <thinking>…</thinking> blocks from visible text
                while True:
                    m_open = _RE_THINKING.search(text)
                    if m_open:
                        before = text[: m_open.start()]
                        after = text[m_open.end():]
                        m_close = _RE_THINKING_CLOSE.search(after)
                        if m_close:
                            text = before + after[m_close.end():]
                            continue
                        else:
                            # Unclosed tag: drop from open to end, mark reasoning active
                            reasoning_open = True
                            state.reasoning_buf += after
                            text = before
                            break
                    else:
                        if reasoning_open:
                            m_close = _RE_THINKING_CLOSE.search(text)
                            if m_close:
                                state.reasoning_buf += text[: m_close.start()]
                                text = text[m_close.end():]
                                reasoning_open = False
                            else:
                                state.reasoning_buf += text
                                text = ""
                        break

                if text:
                    state.text_buf += text
                    yield _make_event(
                        "response.output_text.delta",
                        delta=text,
                        output_index=0,
                    )

            # --- Handle tool call deltas ---
            tool_calls = delta.get("tool_calls")
            if tool_calls and isinstance(tool_calls, list):
                for tc in tool_calls:
                    idx = tc.get("index", 0)

                    func = tc.get("function", {})
                    arguments = func.get("arguments")

                    # Store id / name on first appearance
                    if tc.get("id") and idx not in state.func_call_ids:
                        state.func_call_ids[idx] = tc["id"]
                    if func.get("name") and idx not in state.func_names:
                        state.func_names[idx] = _remap_name(func["name"])

                    # Determine if this chunk carries a real argument delta
                    has_real_args = (
                        arguments is not None and arguments.strip() not in ("", "{}")
                    )

                    if has_real_args:
                        if not state.func_item_added.get(idx, False):
                            # First real argument delta — emit item.added + args delta
                            state.func_item_added[idx] = True
                            call_id = state.func_call_ids.get(idx) or f"fc_{idx}"
                            func_name = state.func_names.get(idx) or "unknown"
                            yield _make_event(
                                "response.output_item.added",
                                output_index=idx,
                                item={
                                    "id": call_id,
                                    "type": "function_call",
                                    "call_id": call_id,
                                    "name": func_name,
                                    "status": "in_progress",
                                },
                            )
                            state.func_args_buf[idx] = arguments
                            yield _make_event(
                                "response.function_call_arguments.delta",
                                output_index=idx,
                                delta=arguments,
                            )
                        else:
                            # Subsequent argument deltas
                            state.func_args_buf[idx] = state.func_args_buf.get(idx, "") + arguments
                            yield _make_event(
                                "response.function_call_arguments.delta",
                                output_index=idx,
                                delta=arguments,
                            )

            # --- Capture usage from final chunk ---
            if finish_reason is not None:
                usage = chunk.get("usage")
                if usage:
                    state.input_tokens = usage.get("prompt_tokens", 0)
                    state.output_tokens = usage.get("completion_tokens", 0)
                    p_details = usage.get("prompt_tokens_details") or {}
                    state.cached_tokens = p_details.get("cached_tokens", 0) or 0
                    c_details = usage.get("completion_tokens_details") or {}
                    state.reasoning_tokens = c_details.get("reasoning_tokens", 0) or 0

    # --- Stream done: emit response.completed ---
    total_tokens = state.input_tokens + state.output_tokens
    yield _make_event(
        "response.completed",
        status="completed",
        output=[],
        usage={
            "input_tokens": state.input_tokens,
            "output_tokens": state.output_tokens,
            "total_tokens": total_tokens,
        },
    )
