"""Tests for src/codex/stream_converter.py.

Covers SSE stream conversion from OpenAI Chat Completions format to Codex
Responses API event format.
"""

import json
from typing import Any, AsyncGenerator, Dict, List

import pytest

from src.codex.stream_converter import (
    CodexStreamState,
    convert_openai_sse_to_responses_sse,
    _response_envelope,
    _ev,
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

    # Check response envelope is a dict, not a string
    created = _parse_event(events[0])
    assert isinstance(created["response"], dict)
    assert created["response"]["object"] == "response"

    # Check deltas
    deltas = [_parse_event(ev) for ev in events if _parse_event(ev)["type"] == "response.output_text.delta"]
    assert deltas[0]["delta"] == "Hello, "
    assert deltas[0]["output_index"] == 0
    assert deltas[1]["delta"] == "world!"

    # Check completed event has usage
    completed = _parse_event(events[-1])
    assert completed["type"] == "response.completed"
    assert completed["response"]["status"] == "completed"
    assert "usage" in completed["response"]


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
    added = [_parse_event(ev) for ev in events if _parse_event(ev)["type"] == "response.output_item.added"]
    tool_added = [a for a in added if a["item"].get("type") == "function_call"]
    assert len(tool_added) == 1
    assert tool_added[0]["item"]["name"] == "exec_command"
    assert tool_added[0]["item"]["id"] == "call_abc"

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

    added = [_parse_event(ev) for ev in events if _parse_event(ev)["type"] == "response.output_item.added"]
    tool_added = [a for a in added if a["item"].get("type") == "function_call"]
    assert len(tool_added) == 2
    assert tool_added[0]["item"]["name"] == "tool_a"
    assert tool_added[1]["item"]["name"] == "tool_b"

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

    idx_created = types.index("response.created")
    idx_text = types.index("response.output_text.delta")
    assert "response.output_item.added" in types
    idx_completed = types.index("response.completed")

    assert idx_created < idx_text < idx_completed


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
    assert "usage" in completed["response"]
    assert completed["response"]["usage"]["total_tokens"] == 0


# --------------------------------------------------------------------------
# 6. Usage accumulation
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_usage_accumulation():
    stream = mock_openai_sse(
        {"choices": [{"delta": {"content": "Hello"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
    )
    events = await _collect(convert_openai_sse_to_responses_sse(stream, "gpt-4"))
    completed = _parse_event(events[-1])
    assert completed["response"]["usage"]["input_tokens"] == 10
    assert completed["response"]["usage"]["output_tokens"] == 5
    assert completed["response"]["usage"]["total_tokens"] == 15


# --------------------------------------------------------------------------
# 7. Tool arg buffering
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tool_arg_buffering():
    stream = mock_openai_sse(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {"name": "fn", "arguments": "{\"a\":"},
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
                                "id": "call_1",
                                "function": {"name": "fn", "arguments": " 1}"},
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

    assert types.count("response.output_item.added") >= 1
    arg_deltas = [_parse_event(ev) for ev in events if _parse_event(ev)["type"] == "response.function_call_arguments.delta"]
    assert len(arg_deltas) == 2
    assert arg_deltas[0]["delta"] == '{"a":'
    assert arg_deltas[1]["delta"] == ' 1}'


# --------------------------------------------------------------------------
# 8. No item added on empty braces (first call delta)
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tool_item_added_once_with_real_args():
    """The tool output_item is added only once — when real arguments arrive."""
    stream = mock_openai_sse(
        # First: empty braces, no name or id — no tool item added yet
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": "{}"},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        # Second: real name + id + args — tool item is added here
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_x",
                                "function": {"name": "my_fn", "arguments": "{\"k\": \"v\"}"},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
    )
    events = await _collect(convert_openai_sse_to_responses_sse(stream, "gpt-4"))
    added = [_parse_event(ev) for ev in events if _parse_event(ev)["type"] == "response.output_item.added"]
    # 1 message item + 1 tool item = 2 total
    assert len(added) == 2  # message item (index 0) + tool item (index 1)


# --------------------------------------------------------------------------
# 9. Reasoning tokens stripped
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reasoning_tokens_stripped():
    stream = mock_openai_sse(
        {"choices": [{"delta": {"content": "<thinking>reasoning</thinking> visible"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": " end"}, "finish_reason": "stop"}]},
    )
    events = await _collect(convert_openai_sse_to_responses_sse(stream, "gpt-4"))
    texts = [_parse_event(ev)["delta"] for ev in events if _parse_event(ev)["type"] == "response.output_text.delta"]
    concat = "".join(texts)
    assert "visible end" in concat
    assert "<thinking>" not in concat


# --------------------------------------------------------------------------
# 10. Event ordering
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_event_ordering():
    stream = mock_openai_sse(
        {"choices": [{"delta": {"content": "a"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "b"}, "finish_reason": "stop"}], "usage": {}},
    )
    events = await _collect(convert_openai_sse_to_responses_sse(stream, "gpt-4"))
    parsed = [_parse_event(ev) for ev in events]

    types = [p["type"] for p in parsed]
    assert types[0] == "response.created"
    assert types[-1] == "response.completed"

    # sequence_numbers should be monotonically increasing
    seqs = [p["sequence_number"] for p in parsed]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)  # all unique


# --------------------------------------------------------------------------
# 11. CodexStreamState fields
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_codex_stream_state_defaults():
    """Verify the dataclass has the expected default field values."""
    s = CodexStreamState(response_id="resp_abc", created_at=123)
    assert s.seq == 0
    assert s.text_buf == ""
    assert s.input_tokens == 0
    assert s.output_tokens == 0
    assert s.cached_tokens == 0
    assert s.reasoning_tokens == 0
    assert s.first_chunk is True


# --------------------------------------------------------------------------
# 12. Tool name remapped via tool_ctx
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tool_name_remapped_via_tool_ctx():
    """If tool_ctx provides a remap function, the emitted name should be the remapped one."""
    class FakeToolCtx:
        def remap_tool_calls_back(self, name):
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
                                "function": {"name": "proxy_fn", "arguments": "{}"},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
    )
    events = await _collect(
        convert_openai_sse_to_responses_sse(stream, "gpt-4", tool_ctx=FakeToolCtx())
    )
    added = [_parse_event(ev) for ev in events if _parse_event(ev)["type"] == "response.output_item.added"]
    tool_added = [a for a in added if a["item"].get("type") == "function_call"]
    assert tool_added[0]["item"]["name"] == "original_fn"
