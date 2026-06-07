"""Convert an OpenAI Chat Completions SSE stream into Codex Responses API SSE events.

Follows the full Responses API streaming spec — every event carries a
``response`` envelope (a dict, not a bare string) so that the official
OpenAI Rust SDK (used by Codex CLI) can deserialize it.
"""

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional


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


_RE_THINKING = re.compile(r"<thinking>|<thinking>", re.IGNORECASE)
_RE_THINKING_CLOSE = re.compile(r"</thinking>|</thinking>", re.IGNORECASE)


def _response_envelope(
    response_id: str,
    created_at: int,
    *,
    model: str = "",
    status: str = "in_progress",
    output: Optional[List[Dict[str, Any]]] = None,
    usage: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Build the response object that every event carries in the ``response`` key."""
    env: Dict[str, Any] = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "error": None,
        "background": False,
    }
    if model:
        env["model"] = model
    if output is not None:
        env["output"] = output
    if usage is not None:
        env["usage"] = usage
    return env


def _ev(type_: str, *, seq: int, response: Dict[str, Any], **fields: Any) -> str:
    payload: Dict[str, Any] = {"type": type_, "sequence_number": seq, "response": response}
    payload.update(fields)
    return f"data: {json.dumps(payload)}\n\n"


async def convert_openai_sse_to_responses_sse(
    openai_sse_stream: AsyncGenerator[str, None],
    request_model: str,
    tool_ctx: Any = None,
) -> AsyncGenerator[str, None]:
    state = CodexStreamState(
        response_id=f"resp_{uuid.uuid4().hex[:16]}",
        created_at=int(time.time()),
    )
    reasoning_open = False
    msg_id = f"msg_{state.response_id}"

    def _remap_name(name: str) -> str:
        if tool_ctx is not None and hasattr(tool_ctx, "remap_tool_calls_back"):
            remapped = tool_ctx.remap_tool_calls_back(name)
            if remapped:
                return remapped
        return name

    # 1. response.created
    state.seq += 1
    yield _ev(
        "response.created",
        seq=state.seq,
        response=_response_envelope(state.response_id, state.created_at, model=request_model),
    )

    # 2. response.in_progress
    state.seq += 1
    yield _ev(
        "response.in_progress",
        seq=state.seq,
        response=_response_envelope(state.response_id, state.created_at, model=request_model),
    )

    # --- Emit output item for the assistant message ---
    state.seq += 1
    yield _ev(
        "response.output_item.added",
        seq=state.seq,
        response=_response_envelope(state.response_id, state.created_at, model=request_model),
        output_index=0,
        item={
            "id": msg_id,
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
        },
    )

    async for line in openai_sse_stream:
        if not line.strip():
            continue
        if not line.startswith("data:"):
            continue
        payload_str = line[len("data:"):].strip()
        if payload_str == "[DONE]":
            break
        try:
            chunk = json.loads(payload_str)
        except (json.JSONDecodeError, ValueError):
            continue

        choices = chunk.get("choices", [])
        if choices:
            choice = choices[0]
            delta = choice.get("delta", {})
            finish_reason = choice.get("finish_reason")

            # --- Text delta ---
            content = delta.get("content")
            if content is not None and content != "":
                text = content
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
                    state.seq += 1
                    yield _ev(
                        "response.output_text.delta",
                        seq=state.seq,
                        response=_response_envelope(state.response_id, state.created_at, model=request_model),
                        delta=text,
                        output_index=0,
                    )

            # --- Tool call deltas ---
            tool_calls = delta.get("tool_calls")
            if tool_calls and isinstance(tool_calls, list):
                for tc in tool_calls:
                    idx = tc.get("index", 0)
                    func = tc.get("function", {})
                    arguments = func.get("arguments")

                    if tc.get("id") and idx not in state.func_call_ids:
                        state.func_call_ids[idx] = tc["id"]
                    if func.get("name") and idx not in state.func_names:
                        state.func_names[idx] = _remap_name(func["name"])

                    has_real_args = arguments is not None and arguments.strip() not in ("", "{}")

                    if not state.func_item_added.get(idx, False):
                        if has_real_args or tc.get("id"):
                            state.func_item_added[idx] = True
                            call_id = state.func_call_ids.get(idx) or f"fc_{idx}"
                            func_name = state.func_names.get(idx) or "unknown"
                            state.seq += 1
                            yield _ev(
                                "response.output_item.added",
                                seq=state.seq,
                                response=_response_envelope(state.response_id, state.created_at, model=request_model),
                                output_index=idx,
                                item={
                                    "id": call_id,
                                    "type": "function_call",
                                    "call_id": call_id,
                                    "name": func_name,
                                    "status": "in_progress",
                                },
                            )
                            if has_real_args:
                                state.func_args_buf[idx] = arguments
                                state.seq += 1
                                yield _ev(
                                    "response.function_call_arguments.delta",
                                    seq=state.seq,
                                    response=_response_envelope(state.response_id, state.created_at, model=request_model),
                                    output_index=idx,
                                    delta=arguments,
                                )
                    elif has_real_args:
                        state.func_args_buf[idx] = state.func_args_buf.get(idx, "") + arguments
                        state.seq += 1
                        yield _ev(
                            "response.function_call_arguments.delta",
                            seq=state.seq,
                            response=_response_envelope(state.response_id, state.created_at),
                            output_index=idx,
                            delta=arguments,
                        )

            # --- Capture usage from final chunk ---
            if finish_reason is not None:
                usage = chunk.get("usage")
                if usage:
                    state.input_tokens = usage.get("prompt_tokens", 0)
                    state.output_tokens = usage.get("completion_tokens", 0)

    # --- stream done: emit response.completed ---
    total_tokens = state.input_tokens + state.output_tokens
    state.seq += 1
    yield _ev(
        "response.completed",
        seq=state.seq,
        response=_response_envelope(
            state.response_id,
            state.created_at,
            status="completed",
            output=[
                {
                    "type": "message",
                    "id": msg_id,
                    "content": [{"type": "output_text", "text": state.text_buf}],
                    "role": "assistant",
                }
            ],
            usage={
                "input_tokens": state.input_tokens,
                "output_tokens": state.output_tokens,
                "total_tokens": total_tokens,
            },
        ),
    )
