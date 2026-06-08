"""Convert an OpenAI Chat Completions SSE stream into Codex Responses API SSE events.

Follows the full Responses API streaming spec. Every event carries a full
``response`` envelope so the OpenAI Rust SDK (Codex CLI) can deserialize it.
"""

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Codex CLI text-embedded tool call parser (defensive fallback)
# ---------------------------------------------------------------------------
# Some Codex models emit tool calls as literal text blocks when they have
# been trained on the Codex CLI format.  We detect these patterns and
# convert them into proper function_call events on the fly.

_RE_TOOL_CALL_SECTION = re.compile(
    r"<\|tool_calls_section_begin\|>"
    r"(.*?)"
    r"<\|tool_calls_section_end\|>",
    re.DOTALL,
)

_RE_TOOL_CALL = re.compile(
    r"<\|tool_call_begin\|>\s*"
    r"(?:functions\.)?(\w+)(?::(\w+))?\s*"
    r"<\|tool_call_argument_begin\|>\s*"
    r"(.*?)"
    r"<\|tool_call_argument_end\|>\s*"
    r"<\|tool_call_end\|>",
    re.DOTALL,
)

# Inline format without markers (single-line or spread across lines):
#   functions.exec_command:50 {"cmd":"..."}
_RE_INLINE_TOOL_CALL_SINGLE = re.compile(
    r"functions\.(\w+):(\w+)\s+(\{.*?\}\n?)$",
    re.DOTALL,
)
# When the JSON is on its own line but preceded by the function sig:
#   functions.exec_command:50\n{"cmd":"..."}
_RE_INLINE_TOOL_CALL_MULTILINE = re.compile(
    r"^\s*functions\.(\w+):(\w+)\s*\n(\{.*?\})\s*$",
    re.DOTALL | re.MULTILINE,
)


def _add_tool_call(tool_calls: List[Dict[str, Any]], func_name: str, raw_args: str) -> None:
    """Helper: validated append a parsed tool call dict."""
    try:
        json.loads(raw_args)
        args = raw_args
    except (json.JSONDecodeError, ValueError):
        args = json.dumps({"input": raw_args})
    tool_calls.append({
        "index": len(tool_calls),
        "id": f"tc_{uuid.uuid4().hex[:12]}",
        "function": {"name": func_name, "arguments": args},
    })


def _extract_tool_calls_from_text(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Scan *text* for embedded Codex CLI tool call blocks.

    Returns ``(clean_text, tool_calls)`` where *clean_text* has the tool call
    blocks removed and *tool_calls* is a list of dicts shaped like OpenAI
    function calls: {"index": int, "id": str, "function": {"name": str, "arguments": str}}.
    """
    clean_text = text
    tool_calls: List[Dict[str, Any]] = []

    # --- Marked format: <tool_calls_section_begin> ... <tool_calls_section_end> ---
    for section_match in _RE_TOOL_CALL_SECTION.finditer(text):
        section = section_match.group(1)
        for m in _RE_TOOL_CALL.finditer(section):
            func_name = m.group(1)
            raw_args = m.group(3).strip()
            _add_tool_call(tool_calls, func_name, raw_args)
        clean_text = clean_text.replace(section_match.group(0), "")

    # --- Inline single-line: functions.exec_command:50 {"cmd": "..."} ---
    for m in _RE_INLINE_TOOL_CALL_SINGLE.finditer(text):
        func_name = m.group(1)
        raw_args = m.group(3).strip()
        _add_tool_call(tool_calls, func_name, raw_args)
        clean_text = clean_text.replace(m.group(0), "")

    # --- Inline multiline: functions.exec_command:50\n{"cmd": "..."} ---
    for m in _RE_INLINE_TOOL_CALL_MULTILINE.finditer(text):
        func_name = m.group(1)
        raw_args = m.group(3).strip()
        _add_tool_call(tool_calls, func_name, raw_args)
        clean_text = clean_text.replace(m.group(0), "")

    return clean_text, tool_calls


# Some backends (e.g. DeepSeek) wrap reasoning content in <thinking> tags.
_RE_THINKING_OPEN = re.compile(r"<thinking>|<Thinking>", re.IGNORECASE)
_RE_THINKING_CLOSE = re.compile(r"</thinking>|</Thinking>", re.IGNORECASE)


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
    input_tokens: int = 0
    output_tokens: int = 0


def _env(response_id: str, created_at: int, *, model: str = "", status: str = "in_progress") -> Dict[str, Any]:
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
    return env


def _ev(type_: str, *, seq: int, response: Dict[str, Any], **fields: Any) -> str:
    payload: Dict[str, Any] = {"type": type_, "sequence_number": seq, "response": response}
    payload.update(fields)
    return f"data: {json.dumps(payload)}\n\n"


async def convert_openai_sse_to_responses_sse(
    openai_sse_stream: AsyncGenerator[str, None],
    request_model: str,
    tool_ctx: Any = None,
    accumulator: Optional[Dict[str, Any]] = None,
) -> AsyncGenerator[str, None]:
    state = CodexStreamState(
        response_id=f"resp_{uuid.uuid4().hex[:16]}",
        created_at=int(time.time()),
    )
    msg_id = f"msg_{state.response_id}"

    def _remap_name(name: str) -> str:
        if tool_ctx is not None and hasattr(tool_ctx, "remap_tool_calls_back"):
            remapped = tool_ctx.remap_tool_calls_back(name)
            if remapped:
                return remapped
        return name

    def _tool_output_index(openai_index: int) -> int:
        # The assistant message is always emitted as output item 0, so tool
        # call output indices must be offset to avoid colliding with it.
        return openai_index + 1

    # --- 1. created ---
    state.seq += 1
    ev = _env(state.response_id, state.created_at, model=request_model)
    yield _ev("response.created", seq=state.seq, response=ev)

    # --- 2. in_progress ---
    state.seq += 1
    yield _ev("response.in_progress", seq=state.seq, response=ev)

    # --- 3. output_item.added (message) ---
    # content is required on ResponseItem::Message in the Rust SDK; without it
    # the event is silently dropped and the turn never gains an active item.
    state.seq += 1
    yield _ev(
        "response.output_item.added",
        seq=state.seq,
        response=ev,
        output_index=0,
        item={
            "id": msg_id,
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [{"type": "output_text", "text": ""}],
        },
    )

    # --- 4. content_part.added (text) ---
    state.seq += 1
    yield _ev(
        "response.content_part.added",
        seq=state.seq,
        response=ev,
        item_id=msg_id,
        output_index=0,
        content_index=0,
        part={"type": "output_text", "text": ""},
    )

    async for line in openai_sse_stream:
        if not line.strip() or not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except (json.JSONDecodeError, ValueError):
            continue

        choices = chunk.get("choices", [])
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta", {}) or {}
        finish_reason = choice.get("finish_reason")

        # --- Text content ---
        # Provider-specific reasoning fields are internal scratchpad tokens, not
        # assistant-visible Responses output text.
        content = delta.get("content")

        if content is not None and str(content) != "":
            text = str(content)

            # Strip <thinking> blocks (DeepSeek-style reasoning)
            while True:
                m_open = _RE_THINKING_OPEN.search(text)
                if m_open:
                    before = text[: m_open.start()]
                    after = text[m_open.end() :]
                    m_close = _RE_THINKING_CLOSE.search(after)
                    if m_close:
                        text = before + after[m_close.end() :]
                        continue
                    else:
                        state.reasoning_active = True
                        text = before
                        break
                else:
                    if state.reasoning_active:
                        m_close = _RE_THINKING_CLOSE.search(text)
                        if m_close:
                            text = text[m_close.end() :]
                            state.reasoning_active = False
                        else:
                            text = ""
                    break

            # Defensive: some models embed tool calls in text (Codex CLI format).
            # Extract them here and emit as proper function_call events.
            clean_text, extracted_calls = _extract_tool_calls_from_text(text)

            for tc in extracted_calls:
                idx = tc["index"]
                func_name = tc["function"]["name"]
                args = tc["function"]["arguments"]
                call_id = tc["id"]

                state.func_call_ids[idx] = call_id
                state.func_names[idx] = _remap_name(func_name)
                state.func_item_added[idx] = True
                state.func_args_buf[idx] = args

                state.seq += 1
                yield _ev(
                    "response.output_item.added",
                    seq=state.seq,
                    response=ev,
                    output_index=_tool_output_index(idx),
                    item={
                        "id": call_id,
                        "type": "function_call",
                        "call_id": call_id,
                        "name": state.func_names[idx],
                        "status": "in_progress",
                    },
                )
                state.seq += 1
                yield _ev(
                    "response.function_call_arguments.delta",
                    seq=state.seq,
                    response=ev,
                    output_index=_tool_output_index(idx),
                    delta=args,
                )

            if clean_text:
                state.text_buf += clean_text
                state.seq += 1
                yield _ev(
                    "response.output_text.delta",
                    seq=state.seq,
                    response=ev,
                    item_id=msg_id,
                    output_index=0,
                    content_index=0,
                    delta=clean_text,
                )

        # --- Tool call deltas ---
        tool_calls = delta.get("tool_calls")
        if tool_calls and isinstance(tool_calls, list):
            for tc in tool_calls:
                idx = tc.get("index", 0)
                func = tc.get("function", {}) or {}
                arguments = func.get("arguments")

                if tc.get("id") and idx not in state.func_call_ids:
                    state.func_call_ids[idx] = tc["id"]
                if func.get("name") and idx not in state.func_names:
                    state.func_names[idx] = _remap_name(func["name"])

                has_real_args = arguments is not None and str(arguments).strip() not in ("", "{}")

                if not state.func_item_added.get(idx, False):
                    if has_real_args or tc.get("id"):
                        state.func_item_added[idx] = True
                        call_id = state.func_call_ids.get(idx) or f"fc_{idx}"
                        func_name = state.func_names.get(idx) or "unknown"
                        state.seq += 1
                        yield _ev(
                            "response.output_item.added",
                            seq=state.seq,
                            response=ev,
                            output_index=_tool_output_index(idx),
                            item={
                                "id": call_id,
                                "type": "function_call",
                                "call_id": call_id,
                                "name": func_name,
                                "status": "in_progress",
                            },
                        )
                        if has_real_args:
                            state.func_args_buf[idx] = str(arguments)
                            state.seq += 1
                            yield _ev(
                                "response.function_call_arguments.delta",
                                seq=state.seq,
                                response=ev,
                                output_index=_tool_output_index(idx),
                                delta=str(arguments),
                            )
                elif has_real_args:
                    state.func_args_buf[idx] = state.func_args_buf.get(idx, "") + str(arguments)
                    state.seq += 1
                    yield _ev(
                        "response.function_call_arguments.delta",
                        seq=state.seq,
                        response=ev,
                        output_index=_tool_output_index(idx),
                        delta=str(arguments),
                    )

        # --- Capture usage ---
        if finish_reason is not None:
            usage = chunk.get("usage")
            if usage:
                state.input_tokens = usage.get("prompt_tokens", 0)
                state.output_tokens = usage.get("completion_tokens", 0)

    # --- 5. output_text.done ---
    state.seq += 1
    yield _ev(
        "response.output_text.done",
        seq=state.seq,
        response=ev,
        item_id=msg_id,
        output_index=0,
        content_index=0,
        text=state.text_buf,
    )

    # --- 6. content_part.done ---
    state.seq += 1
    yield _ev(
        "response.content_part.done",
        seq=state.seq,
        response=ev,
        item_id=msg_id,
        output_index=0,
        content_index=0,
        part={"type": "output_text", "text": state.text_buf},
    )

    # --- 7. output_item.done (message) ---
    state.seq += 1
    yield _ev(
        "response.output_item.done",
        seq=state.seq,
        response=ev,
        output_index=0,
        item={
            "id": msg_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": state.text_buf}],
        },
    )

    # --- 7b. tool call completion events ---
    # Emit function_call_arguments.done and output_item.done for each tool call
    for idx in sorted(state.func_item_added):
        if not state.func_item_added.get(idx):
            continue
        call_id = state.func_call_ids.get(idx) or f"fc_{idx}"
        func_name = state.func_names.get(idx) or "unknown"
        args = state.func_args_buf.get(idx, "")

        # function_call_arguments.done
        state.seq += 1
        yield _ev(
            "response.function_call_arguments.done",
            seq=state.seq,
            response=ev,
            output_index=_tool_output_index(idx),
            item_id=call_id,
            arguments=args,
        )

        # output_item.done (function_call)
        state.seq += 1
        yield _ev(
            "response.output_item.done",
            seq=state.seq,
            response=ev,
            output_index=_tool_output_index(idx),
            item={
                "id": call_id,
                "type": "function_call",
                "status": "completed",
                "call_id": call_id,
                "name": func_name,
                "arguments": args,
            },
        )

    # --- 7c. Populate accumulator for session saving ---
    if accumulator is not None:
        accumulator["response_id"] = state.response_id
        accumulator["text_buf"] = state.text_buf
        accumulator["tool_calls"] = []
        for idx in sorted(state.func_item_added):
            if state.func_item_added.get(idx):
                accumulator["tool_calls"].append(
                    {
                        "id": state.func_call_ids.get(idx) or f"fc_{idx}",
                        "name": state.func_names.get(idx) or "unknown",
                        "arguments": state.func_args_buf.get(idx, ""),
                    }
                )

    # --- 8. response.completed ---
    # The Codex Rust SDK expects "usage" nested inside the ``response`` object,
    # not as a top-level field — ``usage`` here is silently ignored by the
    # parser and results in zeroed token counts.
    state.seq += 1
    total_tokens = state.input_tokens + state.output_tokens
    output_items = [
        {
            "type": "message",
            "id": msg_id,
            "content": [{"type": "output_text", "text": state.text_buf}],
            "role": "assistant",
            "status": "completed",
        }
    ]
    for idx in sorted(state.func_item_added):
        if not state.func_item_added.get(idx):
            continue
        call_id = state.func_call_ids.get(idx) or f"fc_{idx}"
        output_items.append(
            {
                "id": call_id,
                "type": "function_call",
                "status": "completed",
                "call_id": call_id,
                "name": state.func_names.get(idx) or "unknown",
                "arguments": state.func_args_buf.get(idx, ""),
            }
        )

    yield _ev(
        "response.completed",
        seq=state.seq,
        response={
            **ev,
            "status": "completed",
            "output": output_items,
            "usage": {
                "input_tokens": state.input_tokens,
                "output_tokens": state.output_tokens,
                "total_tokens": total_tokens,
            },
        },
    )


async def codex_response_to_sse(
    response_obj: Any,
    request_model: str,
) -> AsyncGenerator[str, None]:
    """Convert a non-streaming Codex ``ResponsesResponse`` to Codex SSE events.

    Used when ``run_search_loop`` returns a complete response but the client
    requested streaming. Mirrors the event sequence produced by
    ``convert_openai_sse_to_responses_sse`` for a response that has already
    been materialised end-to-end.
    """
    response_id = getattr(response_obj, "id", f"resp_{uuid.uuid4().hex[:16]}")
    created_at = int(time.time())
    seq = 0

    def _next_ev(type_: str, **fields: Any) -> str:
        nonlocal seq
        seq += 1
        env = _env(response_id, created_at, model=request_model)
        return _ev(type_, seq=seq, response=env, **fields)

    # --- 1. response.created ---
    yield _next_ev("response.created")

    # --- 2. response.in_progress ---
    yield _next_ev("response.in_progress")

    # --- 3. Output items ---
    output_items = getattr(response_obj, "output", []) or []
    msg_id = f"msg_{response_id}"

    for idx, item in enumerate(output_items):
        item_type = getattr(item, "type", None)

        if item_type == "message":
            content = getattr(item, "content", "") or ""
            # Content may be a string or a list of blocks; extract text for deltas.
            text = ""
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") in ("output_text", "text"):
                        text += block.get("text", "")

            # output_item.added (message)
            yield _next_ev(
                "response.output_item.added",
                output_index=idx,
                item={
                    "id": msg_id,
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": ""}],
                },
            )

            # content_part.added
            yield _next_ev(
                "response.content_part.added",
                item_id=msg_id,
                output_index=idx,
                content_index=0,
                part={"type": "output_text", "text": ""},
            )

            if text:
                yield _next_ev(
                    "response.output_text.delta",
                    item_id=msg_id,
                    output_index=idx,
                    content_index=0,
                    delta=text,
                )
                yield _next_ev(
                    "response.output_text.done",
                    item_id=msg_id,
                    output_index=idx,
                    content_index=0,
                    text=text,
                )

            # content_part.done
            yield _next_ev(
                "response.content_part.done",
                item_id=msg_id,
                output_index=idx,
                content_index=0,
                part={"type": "output_text", "text": text},
            )

            # output_item.done (message)
            yield _next_ev(
                "response.output_item.done",
                output_index=idx,
                item={
                    "id": msg_id,
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                },
            )

        elif item_type == "function_call":
            call_id = getattr(item, "call_id", "") or f"fc_{idx}"
            name = getattr(item, "name", "")
            args = getattr(item, "arguments", "") or ""

            # output_item.added (function_call)
            yield _next_ev(
                "response.output_item.added",
                output_index=idx,
                item={
                    "id": call_id,
                    "type": "function_call",
                    "status": "in_progress",
                    "call_id": call_id,
                    "name": name,
                },
            )

            # function_call_arguments.delta (emit complete args in one go)
            if args:
                yield _next_ev(
                    "response.function_call_arguments.delta",
                    output_index=idx,
                    delta=args,
                )

            # function_call_arguments.done
            yield _next_ev(
                "response.function_call_arguments.done",
                output_index=idx,
                item_id=call_id,
                arguments=args,
            )

            # output_item.done (function_call)
            yield _next_ev(
                "response.output_item.done",
                output_index=idx,
                item={
                    "id": call_id,
                    "type": "function_call",
                    "status": "completed",
                    "call_id": call_id,
                    "name": name,
                    "arguments": args,
                },
            )

    # --- 4. response.completed ---
    usage = getattr(response_obj, "usage", None)
    if usage is not None:
        in_tok = getattr(usage, "input_tokens", 0) or 0
        out_tok = getattr(usage, "output_tokens", 0) or 0
        total = getattr(usage, "total_tokens", in_tok + out_tok) or (in_tok + out_tok)
    else:
        in_tok = out_tok = total = 0

    # Build completed output items for the final event envelope
    completed_output: List[Dict[str, Any]] = []
    for item in output_items:
        itype = getattr(item, "type", None)
        if itype == "message":
            content = getattr(item, "content", "") or ""
            text = ""
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") in ("output_text", "text"):
                        text += block.get("text", "")
            completed_output.append({
                "type": "message",
                "id": msg_id,
                "content": [{"type": "output_text", "text": text}],
                "role": "assistant",
                "status": "completed",
            })
        elif itype == "function_call":
            call_id = getattr(item, "call_id", "")
            completed_output.append({
                "id": call_id,
                "type": "function_call",
                "status": "completed",
                "call_id": call_id,
                "name": getattr(item, "name", ""),
                "arguments": getattr(item, "arguments", "") or "",
            })

    yield _ev(
        "response.completed",
        seq=seq + 1,
        response={
            **_env(response_id, created_at, model=request_model, status="completed"),
            "status": "completed",
            "output": completed_output,
            "usage": {
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "total_tokens": total,
            },
        },
    )
