"""Tests for src/codex/stream_converter.py.

Covers SSE stream conversion from OpenAI Chat Completions format to Codex
Responses API event format.
"""

import json
from typing import Any, AsyncGenerator, Dict, List

import pytest

from src.codex.stream_converter import (
    CodexStreamState,
    _make_event,
    convert_openai_sse_to_responses_sse,
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

async def mock_openai_sse(*chunks: Dict[str, Any]) -> AsyncGenerator[str, None]:
    """Yield raw SSE lines from dict chunks, ending with [DONE]."""
    for chunk in chunks:
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


async def _collect(stream) -> List[str]:
    """Drain an async generator into a list."""
    return [ev async for ev in stream]


def _parse_event(line: str) -> Dict[str, Any]:
    """Parse the JSON payload of a SSE data: line."""
    prefix = "data: "
    assert line.startswith(prefix), f"Expected SSE data line, got: {line!r}"
    return json.loads(line[len(prefix):])


# --------------------------------------------------------------------------
# 1. Text-only stream
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_text_only_stream():
    stream = mock_openai_sse(
        {"choices": [{"delta": {"content": "Hello, "}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "world!"}, "finish_reason": None}]},
    )
    events = await _collect(convert_openai_sse_to_responses_sse(stream, "gpt-4"))
    types = [_parse_event(ev)["type"] for ev in events]

    assert "response.created" in types
    assert "response.in_progress" in types
    assert types.count("response.output_text.delta") == 2
    assert "response.completed" in types

    # Check first output_text.delta
    deltas = [_parse_event(ev) for ev in events if _parse_event(ev)["type"] == "response.output_text.delta"]
    assert deltas[0]["delta"] == "Hello, "
    assert deltas[0]["output_index"] == 0
    assert deltas[1]["delta"] == "world!"

    # Check completed event has usage
    completed = _parse_event(events[-1])
    assert completed["type"] == "response.completed"
    assert completed["status"] == "completed"


# --------------------------------------------------------------------------
# 2. Tool call stream
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tool_call_stream():
    stream = mock_openai_sse(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_abc",
                                "function": {"name": "exec_command", "arguments": "{\"cmd\": \"ls\"}"},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
    )
    events = await _collect(convert_openai_sse_to_responses_sse(stream, "gpt-4"))
    types = [_parse_event(ev)["type"] for ev in events]

    assert "response.output_item.added" in types
    added = [_parse_event(ev) for ev in events if _parse_event(ev)["type"] == "response.output_item.added"][0]
    assert added["item"]["name"] == "exec_command"
    assert added["item"]["type"] == "function_call"
    assert added["item"]["id"] == "call_abc"

    assert "response.function_call_arguments.delta" in types
    arg_deltas = [_parse_event(ev) for ev in events if _parse_event(ev)["type"] == "response.function_call_arguments.delta"]
    assert len(arg_deltas) == 1
    assert arg_deltas[0]["delta"] == '{"cmd": "ls"}'


# --------------------------------------------------------------------------
# 3. Multiple tool calls in same stream
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_multiple_tool_calls():
    stream = mock_openai_sse(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_a",
                                "function": {"name": "tool_a", "arguments": "{\"x\": 1}"},
                            },
                            {
                                "index": 1,
                                "id": "call_b",
                                "function": {"name": "tool_b", "arguments": "{\"y\": 2}"},
                            },
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
    )
    events = await _collect(convert_openai_sse_to_responses_sse(stream, "gpt-4"))
    types = [_parse_event(ev)["type"] for ev in events]

    assert types.count("response.output_item.added") == 2
    added = [_parse_event(ev) for ev in events if _parse_event(ev)["type"] == "response.output_item.added"]
    assert added[0]["item"]["name"] == "tool_a"
    assert added[1]["item"]["name"] == "tool_b"

    assert types.count("response.function_call_arguments.delta") == 2


# --------------------------------------------------------------------------
# 4. Mixed text + tool calls
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mixed_text_and_tools():
    stream = mock_openai_sse(
        {"choices": [{"delta": {"content": "Let me "}, "finish_reason": None}]},
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {"name": "search", "arguments": "{\"q\": \"x\"}"},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
    )
    events = await _collect(convert_openai_sse_to_responses_sse(stream, "gpt-4"))
    types = [_parse_event(ev)["type"] for ev in events]

    # Event ordering
    idx_created = types.index("response.created")
    idx_text = types.index("response.output_text.delta")
    idx_added = types.index("response.output_item.added")
    idx_completed = types.index("response.completed")

    assert idx_created < idx_text < idx_added < idx_completed


# --------------------------------------------------------------------------
# 5. Empty stream
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_stream():
    stream = mock_openai_sse()
    events = await _collect(convert_openai_sse_to_responses_sse(stream, "gpt-4"))
    types = [_parse_event(ev)["type"] for ev in events]

    assert types[0] == "response.created"
    assert types[-1] == "response.completed"
    completed = _parse_event(events[-1])
    assert completed["usage"] == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


# --------------------------------------------------------------------------
# 6. Usage accumulation from final chunk
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_usage_accumulation():
    stream = mock_openai_sse(
        {"choices": [{"delta": {"content": "hi"}, "finish_reason": None}]},
        {
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "prompt_tokens_details": {"cached_tokens": 2},
                "completion_tokens_details": {"reasoning_tokens": 1},
            },
        },
    )
    events = await _collect(convert_openai_sse_to_responses_sse(stream, "gpt-4"))
    completed = _parse_event(events[-1])
    assert completed["usage"]["input_tokens"] == 10
    assert completed["usage"]["output_tokens"] == 5
    assert completed["usage"]["total_tokens"] == 15


# --------------------------------------------------------------------------
# 7. Tool arg buffering: first real delta creates item, subsequent emits deltas
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tool_arg_buffering_creates_item_on_first_real_delta():
    """Regression: never emit output_item.added on empty {} arguments."""
    stream = mock_openai_sse(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {"name": "my_tool", "arguments": ""},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": "{\"k\": "},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": "\"v\"}"},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
    )
    events = await _collect(convert_openai_sse_to_responses_sse(stream, "gpt-4"))
    types = [_parse_event(ev)["type"] for ev in events]

    assert types.count("response.output_item.added") == 1
    assert types.count("response.function_call_arguments.delta") == 2

    arg_deltas = [_parse_event(ev) for ev in events if _parse_event(ev)["type"] == "response.function_call_arguments.delta"]
    assert arg_deltas[0]["delta"] == '{"k": '
    assert arg_deltas[1]["delta"] == '"v"}'


@pytest.mark.asyncio
async def test_no_item_added_on_empty_braces():
    """Empty {} should be skipped and not trigger item added."""
    stream = mock_openai_sse(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {"name": "my_tool", "arguments": "{}"},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
    )
    events = await _collect(convert_openai_sse_to_responses_sse(stream, "gpt-4"))
    types = [_parse_event(ev)["type"] for ev in events]
    assert "response.output_item.added" not in types


# --------------------------------------------------------------------------
# 8. Reasoning tokens passthrough
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reasoning_tokens_stripped():
    stream = mock_openai_sse(
        {"choices": [{"delta": {"content": "x<thinking>secret</thinking>y"}, "finish_reason": None}]},
    )
    events = await _collect(convert_openai_sse_to_responses_sse(stream, "gpt-4"))
    deltas = [_parse_event(ev) for ev in events if _parse_event(ev)["type"] == "response.output_text.delta"]
    assert len(deltas) == 1
    assert deltas[0]["delta"] == "xy"


@pytest.mark.asyncio
async def test_reasoning_tokens_passthrough_across_chunks():
    """<thinking> opened in one chunk and closed in another."""
    stream = mock_openai_sse(
        {"choices": [{"delta": {"content": "a<thinking>secret"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "more</thinking>b"}, "finish_reason": None}]},
    )
    events = await _collect(convert_openai_sse_to_responses_sse(stream, "gpt-4"))
    deltas = [_parse_event(ev) for ev in events if _parse_event(ev)["type"] == "response.output_text.delta"]
    # First chunk drops everything after <thinking>, yields "a"
    # Second chunk drops up to </thinking>, yields "b"
    assert len(deltas) == 2
    assert deltas[0]["delta"] == "a"
    assert deltas[1]["delta"] == "b"


# --------------------------------------------------------------------------
# 9. Event ordering: created before deltas, completed last
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_event_ordering():
    stream = mock_openai_sse(
        {"choices": [{"delta": {"content": "hi"}, "finish_reason": None}]},
    )
    events = await _collect(convert_openai_sse_to_responses_sse(stream, "gpt-4"))
    types = [_parse_event(ev)["type"] for ev in events]

    assert types[0] == "response.created"
    assert types.count("response.created") == 1
    assert types[-1] == "response.completed"
    assert "response.output_text.delta" in types
    assert types.index("response.created") < types.index("response.output_text.delta")
    assert types.index("response.output_text.delta") < types.index("response.completed")


# --------------------------------------------------------------------------
# 10. Error mid-stream handling
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_error_mid_stream_yields_failed_event():
    """If the upstream stream yields broken JSON, the converter should skip it
    and continue without hanging.  The consumer should still get a completed
    event when the stream ends."""

    async def broken_stream():
        yield "data: {\"choices\": [{\"delta\": {\"content\": \"ok\"}}]}\n\n"
        yield "not a data: line\n\n"
        yield "data: this is not json\n\n"
        yield "data: [DONE]\n\n"

    events = await _collect(convert_openai_sse_to_responses_sse(broken_stream(), "gpt-4"))
    types = [_parse_event(ev)["type"] for ev in events]
    assert "response.created" in types
    assert "response.output_text.delta" in types
    assert "response.completed" in types


# --------------------------------------------------------------------------
# State dataclass sanity
# --------------------------------------------------------------------------

def test_codex_stream_state_defaults():
    state = CodexStreamState(response_id="resp_abc", created_at=123)
    assert state.seq == 0
    assert state.text_buf == ""
    assert state.func_args_buf == {}
    assert state.func_names == {}
    assert state.func_call_ids == {}
    assert state.func_item_added == {}
    assert state.reasoning_active is False
    assert state.reasoning_buf == ""
    assert state.input_tokens == 0
    assert state.output_tokens == 0
    assert state.cached_tokens == 0
    assert state.reasoning_tokens == 0
    assert state.first_chunk is True


# --------------------------------------------------------------------------
# Tool name remapping via tool_ctx
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tool_name_remapped_via_tool_ctx():
    class FakeCtx:
        def remap_tool_calls_back(self, name: str) -> str:
            if name == "proxy_fn":
                return "original_fn"
            return name

    stream = mock_openai_sse(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {
                                    "name": "proxy_fn",
                                    "arguments": "{\"k\": 1}",
                                },
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
    )
    events = await _collect(
        convert_openai_sse_to_responses_sse(stream, "gpt-4", tool_ctx=FakeCtx())
    )
    added = [_parse_event(ev) for ev in events if _parse_event(ev)["type"] == "response.output_item.added"]
    assert added[0]["item"]["name"] == "original_fn"
