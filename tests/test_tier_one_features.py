"""Tests for the four Tier-1 proxy upgrades on branch kiran/ToolCallFixes:

1. Prefix-cache discipline   (request_converter)
2. count_tokens endpoint     (api/endpoints + request_converter)
3. Pair-aware auto-truncation (request_converter._trim_messages_to_fit)
4. Tool-call JSON repair + dedup (response_converter, always-on)
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.conversion.request_converter import (
    _canonicalize_schema,
    _compute_prefix_fingerprint,
    _trim_messages_to_fit,
    convert_claude_to_openai,
    count_claude_request_tokens,
)
from src.conversion.response_converter import (
    _finalize_tool_args,
    _sanitize_tool_arguments,
    _try_repair_json,
    convert_openai_streaming_to_claude_with_cancellation,
    convert_openai_to_claude_response,
)
from src.core.constants import Constants
from src.core.model_manager import model_manager
from src.models.claude import (
    ClaudeContentBlockText,
    ClaudeContentBlockToolUse,
    ClaudeContentBlockToolResult,
    ClaudeMessage,
    ClaudeMessagesRequest,
    ClaudeTokenCountRequest,
    ClaudeTool,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _DummyRequest:
    async def is_disconnected(self):
        return False


class _DummyClient:
    def cancel_request(self, _request_id):
        return True


class _DummyLogger:
    def debug(self, *_a, **_k):
        pass

    def info(self, *_a, **_k):
        pass

    def warning(self, *_a, **_k):
        pass

    def error(self, *_a, **_k):
        pass


def _build_request_with_tools():
    """Two requests that share a system prompt + tools should produce the
    same prefix fingerprint, even when called in two different conversations.
    """
    tools = [
        ClaudeTool(
            name="search",
            description="Search the web",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"},
                    "limit": {"type": "integer", "description": "Number of results"},
                },
                "required": ["query"],
            },
        )
    ]
    return ClaudeMessagesRequest(
        model="claude-3-5-sonnet-20241022",
        max_tokens=128,
        system="You are a helpful assistant.",
        messages=[ClaudeMessage(role="user", content="hello")],
        tools=tools,
    )


# ---------------------------------------------------------------------------
# (1) Prefix-cache discipline
# ---------------------------------------------------------------------------


def test_canonicalize_schema_sorts_keys_recursively():
    schema = {
        "type": "object",
        "properties": {
            "z": {"type": "string"},
            "a": {"type": "integer"},
        },
        "required": ["z"],
    }
    canonical = _canonicalize_schema(schema)
    # Top-level keys sorted
    assert list(canonical.keys()) == ["properties", "required", "type"]
    # Nested keys sorted
    assert list(canonical["properties"].keys()) == ["a", "z"]


def test_canonicalize_schema_preserves_lists():
    """List order is semantic in JSON Schema (e.g. enum, required); keep it."""
    schema = {"required": ["b", "a", "c"], "enum": [3, 1, 2]}
    canonical = _canonicalize_schema(schema)
    assert canonical["required"] == ["b", "a", "c"]
    assert canonical["enum"] == [3, 1, 2]


def test_prefix_fingerprint_is_stable_across_two_requests():
    """Two requests with the same system + tools should hash identically.

    This is the property that the Nebius prefix cache relies on. If this
    breaks, we silently lose cache hits.
    """
    req1 = _build_request_with_tools()
    req2 = _build_request_with_tools()
    out1 = convert_claude_to_openai(req1, model_manager)
    out2 = convert_claude_to_openai(req2, model_manager)

    sys1 = out1["messages"][0] if out1["messages"][0]["role"] == "system" else None
    sys2 = out2["messages"][0] if out2["messages"][0]["role"] == "system" else None
    fp1 = _compute_prefix_fingerprint(sys1, out1.get("tools"))
    fp2 = _compute_prefix_fingerprint(sys2, out2.get("tools"))
    assert fp1 == fp2


def test_prefix_fingerprint_changes_when_system_changes():
    req1 = _build_request_with_tools()
    req2 = _build_request_with_tools()
    req2.system = "You are a different assistant."
    out1 = convert_claude_to_openai(req1, model_manager)
    out2 = convert_claude_to_openai(req2, model_manager)
    fp1 = _compute_prefix_fingerprint(out1["messages"][0], out1.get("tools"))
    fp2 = _compute_prefix_fingerprint(out2["messages"][0], out2.get("tools"))
    assert fp1 != fp2


def test_tool_parameters_emitted_with_sorted_keys():
    """Even if the user supplies properties in random order, the wire form
    must be deterministic so prefix cache keys stay stable."""
    req = ClaudeMessagesRequest(
        model="claude-3-5-sonnet-20241022",
        max_tokens=64,
        messages=[ClaudeMessage(role="user", content="hi")],
        tools=[
            ClaudeTool(
                name="x",
                description="x",
                input_schema={
                    "properties": {"z": {"type": "string"}, "a": {"type": "string"}},
                    "type": "object",
                },
            )
        ],
    )
    out = convert_claude_to_openai(req, model_manager)
    params = out["tools"][0]["function"]["parameters"]
    assert list(params["properties"].keys()) == ["a", "z"]


# ---------------------------------------------------------------------------
# (2) count_tokens endpoint backing function
# ---------------------------------------------------------------------------


def test_count_tokens_includes_tool_definitions():
    """The previous count_tokens silently dropped tools. For Claude Code
    that's a 5-10k token undercount. Verify we now include them."""
    base = ClaudeTokenCountRequest(
        model="claude-3-5-sonnet-20241022",
        messages=[ClaudeMessage(role="user", content="hello")],
    )
    with_tools = ClaudeTokenCountRequest(
        model="claude-3-5-sonnet-20241022",
        messages=[ClaudeMessage(role="user", content="hello")],
        tools=[
            ClaudeTool(
                name="search",
                description="Search the web for things",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "the query string"},
                    },
                },
            )
        ],
    )
    n_base = count_claude_request_tokens(base)
    n_with_tools = count_claude_request_tokens(with_tools)
    assert n_with_tools > n_base + 5, (
        f"Tool definition should add tokens; got base={n_base}, with_tools={n_with_tools}"
    )


def test_count_tokens_counts_tool_use_and_tool_result_blocks():
    request = ClaudeTokenCountRequest(
        model="claude-3-5-sonnet-20241022",
        messages=[
            ClaudeMessage(role="user", content="run ls"),
            ClaudeMessage(
                role="assistant",
                content=[
                    ClaudeContentBlockToolUse(
                        type="tool_use",
                        id="tu_1",
                        name="bash",
                        input={"command": "ls -la"},
                    )
                ],
            ),
            ClaudeMessage(
                role="user",
                content=[
                    ClaudeContentBlockToolResult(
                        type="tool_result",
                        tool_use_id="tu_1",
                        content="total 0\ndrwxr-xr-x ...",
                    )
                ],
            ),
        ],
    )
    n = count_claude_request_tokens(request)
    assert n > 5  # non-trivial


def test_count_tokens_returns_at_least_one():
    """Empty request should still return >= 1 (Anthropic shape)."""
    request = ClaudeTokenCountRequest(
        model="claude-3-5-sonnet-20241022",
        messages=[],
    )
    assert count_claude_request_tokens(request) >= 1


def test_count_tokens_handles_schema_less_computer_tool():
    request = ClaudeTokenCountRequest(
        model="claude-3-5-sonnet-20241022",
        messages=[ClaudeMessage(role="user", content="hi")],
        tools=[
            ClaudeTool(
                name="computer",
                type="computer_20251124",
                display_width_px=1024,
                display_height_px=768,
            )
        ],
    )
    n = count_claude_request_tokens(request)
    # Should be larger than a bare "hi" message because the inferred schema
    # for the computer tool is substantial.
    bare = count_claude_request_tokens(
        ClaudeTokenCountRequest(
            model="claude-3-5-sonnet-20241022",
            messages=[ClaudeMessage(role="user", content="hi")],
        )
    )
    assert n > bare + 20


# ---------------------------------------------------------------------------
# (3) Pair-aware auto-truncation
# ---------------------------------------------------------------------------


def test_trim_keeps_assistant_tool_calls_and_their_results_together():
    """If we drop an assistant message that issued tool_calls, we must also
    drop the matching `role=tool` replies — otherwise the upstream backend
    will return 400 for orphan tool results."""
    messages = [
        {"role": "system", "content": "sys"},
        # Old assistant turn issued a tool call that got a result
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_old",
                    "type": "function",
                    "function": {"name": "bash", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_old", "content": "old result"},
        # Many filler messages to force trimming
        *[
            {"role": "user", "content": "x" * 4000}
            for _ in range(20)
        ],
        {"role": "user", "content": "current question"},
    ]
    trimmed, dropped = _trim_messages_to_fit(messages, context_limit=2000, reserve=200)
    # Verify no orphan: every role=tool must be preceded by an assistant
    # with tool_calls in the trimmed output.
    for i, m in enumerate(trimmed):
        if m.get("role") == "tool":
            # find the nearest preceding assistant
            j = i - 1
            while j >= 0 and trimmed[j].get("role") != "assistant":
                j -= 1
            assert j >= 0, f"orphan tool message at index {i}"
            assert trimmed[j].get("tool_calls"), (
                f"tool message at {i} preceded by assistant without tool_calls"
            )
    assert dropped > 0


def test_trim_preserves_system_and_last_message():
    messages = [
        {"role": "system", "content": "you are X"},
        *[{"role": "user", "content": "x" * 5000} for _ in range(10)],
        {"role": "user", "content": "current"},
    ]
    trimmed, dropped = _trim_messages_to_fit(messages, context_limit=1500, reserve=200)
    assert trimmed[0]["role"] == "system"
    assert trimmed[-1]["content"] == "current"
    assert dropped > 0


def test_trim_drops_tool_pair_atomically():
    """Drop should remove the ENTIRE (assistant + tool replies) group, not half."""
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_a",
                    "type": "function",
                    "function": {"name": "bash", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_a", "content": "result A " * 5000},
        {"role": "user", "content": "current question"},
    ]
    trimmed, dropped = _trim_messages_to_fit(messages, context_limit=500, reserve=100)
    # The assistant+tool pair should both be gone, never just one
    has_assistant_with_tools = any(
        m.get("role") == "assistant" and m.get("tool_calls") for m in trimmed
    )
    has_orphan_tool = any(
        m.get("role") == "tool" and not has_assistant_with_tools for m in trimmed
    )
    assert not has_orphan_tool


# ---------------------------------------------------------------------------
# (4) Tool-call JSON repair + dedup (always-on)
# ---------------------------------------------------------------------------


def test_try_repair_json_strips_trailing_commas():
    parsed, repaired = _try_repair_json('{"a": 1, "b": 2,}')
    assert parsed == {"a": 1, "b": 2}
    parsed, repaired = _try_repair_json('[1, 2, 3,]')
    assert parsed == [1, 2, 3]


def test_try_repair_json_escapes_raw_newlines():
    raw = '{"text": "line1\nline2"}'
    parsed, repaired = _try_repair_json(raw)
    assert parsed == {"text": "line1\nline2"}


def test_try_repair_json_returns_none_on_unrecoverable():
    parsed, repaired = _try_repair_json("this is not json at all {{{{")
    assert parsed is None


def test_sanitizer_default_behavior_unchanged():
    """Existing GLM-friendly sanitizer behavior must remain identical."""
    name, args = _sanitize_tool_arguments(
        "Bash",
        "<arg_key>command</arg_key><arg_value>ls -la</arg_value>",
    )
    assert name == "Bash"
    assert json.loads(args) == {"command": "ls -la"}


def test_finalize_tool_args_repairs_trailing_comma():
    """The added _try_repair_json pass should turn near-JSON into valid JSON."""
    name, sanitized, parsed = _finalize_tool_args("bash", '{"command": "ls",}')
    assert parsed == {"command": "ls"}


def test_finalize_tool_args_returns_none_on_unrecoverable():
    """When repair can't fix it, parsed=None and callers wrap in raw_arguments."""
    name, sanitized, parsed = _finalize_tool_args(
        "unknown_tool", "totally not json {{{"
    )
    assert parsed is None


def test_unrepairable_args_fall_back_to_raw_arguments():
    """Historical fallback behavior is preserved end-to-end."""
    request = ClaudeMessagesRequest(
        model="claude-3-5-sonnet-20241022",
        max_tokens=64,
        messages=[ClaudeMessage(role="user", content="x")],
    )
    openai_response = {
        "id": "r",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {
                                "name": "unknown_tool",
                                "arguments": "totally not json {{{",
                            },
                        }
                    ],
                },
            }
        ],
    }
    response = convert_openai_to_claude_response(openai_response, request)
    tool_block = next(b for b in response["content"] if b["type"] == "tool_use")
    assert "raw_arguments" in tool_block["input"]


def test_dedup_drops_duplicate_tool_calls_in_one_turn():
    """Open models (GLM-4.5 in particular) sometimes emit the same tool call
    twice in a single turn. We dedup by (name, sorted-args)."""
    request = ClaudeMessagesRequest(
        model="claude-3-5-sonnet-20241022",
        max_tokens=64,
        messages=[ClaudeMessage(role="user", content="run ls")],
    )
    openai_response = {
        "id": "r",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": '{"command": "ls -la"}',
                            },
                        },
                        {
                            "id": "c2",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": '{"command": "ls -la"}',
                            },
                        },
                    ],
                },
            }
        ],
    }
    response = convert_openai_to_claude_response(openai_response, request)
    tool_blocks = [b for b in response["content"] if b["type"] == "tool_use"]
    assert len(tool_blocks) == 1


# ---------------------------------------------------------------------------
# (4b) Streaming dedup of duplicate tool calls
# ---------------------------------------------------------------------------


async def _dup_tool_stream():
    yield "data: " + json.dumps(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "type": "function",
                                "function": {
                                    "name": "bash",
                                    "arguments": '{"command":"ls"}',
                                },
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        }
    )
    yield "data: " + json.dumps(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 1,
                                "id": "c2",
                                "type": "function",
                                "function": {
                                    "name": "bash",
                                    "arguments": '{"command":"ls"}',
                                },
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        }
    )
    yield "data: " + json.dumps(
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}
    )
    yield "data: [DONE]"


@pytest.mark.asyncio
async def test_streaming_dedups_duplicate_tool_calls():
    request = ClaudeMessagesRequest(
        model="claude-3-5-sonnet-20241022",
        max_tokens=64,
        messages=[ClaudeMessage(role="user", content="ls")],
        stream=True,
    )

    events = []
    async for event in convert_openai_streaming_to_claude_with_cancellation(
        _dup_tool_stream(),
        request,
        _DummyLogger(),
        _DummyRequest(),
        _DummyClient(),
        "req_dedup",
    ):
        events.append(event)

    serialized = "".join(events)
    # The two tool calls have different IDs but identical (name, args).
    # Both content blocks are started (by the time we know it's a dup,
    # the block_start has already gone out), but only one input_json_delta
    # should have been emitted.
    assert serialized.count('"type": "input_json_delta"') == 1
