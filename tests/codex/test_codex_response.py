"""Unit tests for src/codex/response_converter.py."""

import uuid
from typing import Any, List
from unittest.mock import MagicMock

from src.codex.models import ResponsesItem, ResponsesResponse, ResponsesUsage
from src.codex.response_converter import (
    convert_openai_error_to_responses,
    convert_openai_to_responses,
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _make_openai_response(
    content: str = "",
    tool_calls: List[dict] = None,
    usage: dict = None,
) -> dict:
    """Build a minimal OpenAI Chat Completion dict."""
    message: dict = {"role": "assistant"}
    if content:
        message["content"] = content
    if tool_calls:
        message["tool_calls"] = tool_calls

    return {
        "id": "chatcmpl_test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "stop",
            }
        ],
        "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


# --------------------------------------------------------------------------
# 1. Text-only response
# --------------------------------------------------------------------------
def test_text_only_response():
    """A plain text response yields a single assistant message item."""
    openai_resp = _make_openai_response(content="Hello, world!")
    result = convert_openai_to_responses(openai_resp, request_model="gpt-4")

    assert result.status == "completed"
    assert result.model == "gpt-4"
    assert len(result.output) == 1
    item = result.output[0]
    assert item.type == "message"
    assert item.role == "assistant"
    assert item.content == "Hello, world!"
    assert item.status == "completed"


# --------------------------------------------------------------------------
# 2. Tool call response
# --------------------------------------------------------------------------
def test_tool_call_response():
    """A tool_call response yields function_call items and no text item."""
    openai_resp = _make_openai_response(
        content="",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "Paris"}',
                },
            }
        ],
    )
    result = convert_openai_to_responses(openai_resp, request_model="gpt-4")

    assert len(result.output) == 1
    item = result.output[0]
    assert item.type == "function_call"
    assert item.call_id == "call_1"
    assert item.name == "get_weather"
    assert item.arguments == '{"location": "Paris"}'
    assert item.status == "completed"


# --------------------------------------------------------------------------
# 3. Mixed text + tool_calls
# --------------------------------------------------------------------------
def test_mixed_text_and_tool_calls():
    """Text + tool_calls produces a message item followed by function_call items."""
    openai_resp = _make_openai_response(
        content="Let me check the weather.",
        tool_calls=[
            {
                "id": "call_2",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "London"}',
                },
            }
        ],
    )
    result = convert_openai_to_responses(openai_resp, request_model="gpt-4")

    assert len(result.output) == 2
    assert result.output[0].type == "message"
    assert result.output[0].content == "Let me check the weather."
    assert result.output[1].type == "function_call"
    assert result.output[1].call_id == "call_2"


# --------------------------------------------------------------------------
# 4. Usage mapping with all fields
# --------------------------------------------------------------------------
def test_usage_mapping_full():
    """All optional usage fields are mapped correctly."""
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "prompt_tokens_details": {
            "cached_tokens": 10,
            "cache_creation_tokens": 5,
        },
        "completion_tokens_details": {
            "reasoning_tokens": 20,
        },
    }
    openai_resp = _make_openai_response(content="Hi", usage=usage)
    result = convert_openai_to_responses(openai_resp, request_model="gpt-3.5")

    u = result.usage
    assert u.input_tokens == 100
    assert u.output_tokens == 50
    assert u.total_tokens == 150
    assert u.input_tokens_details.cached_tokens == 10
    assert u.output_tokens_details.reasoning_tokens == 20
    assert u.cache_creation_input_tokens == 5
    assert u.cache_read_input_tokens == 10


# --------------------------------------------------------------------------
# 5. Usage mapping with missing optional fields
# --------------------------------------------------------------------------
def test_usage_mapping_minimal():
    """When optional usage details are absent they stay None."""
    usage = {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }
    openai_resp = _make_openai_response(content="Hi", usage=usage)
    result = convert_openai_to_responses(openai_resp, request_model="gpt-4")

    u = result.usage
    assert u.input_tokens == 10
    assert u.output_tokens == 5
    assert u.total_tokens == 15
    assert u.input_tokens_details is None
    assert u.output_tokens_details is None
    assert u.cache_creation_input_tokens is None
    assert u.cache_read_input_tokens is None


# --------------------------------------------------------------------------
# 6. Previous_id propagation
# --------------------------------------------------------------------------
def test_previous_id_propagation():
    """previous_id is forwarded into the resulting ResponsesResponse."""
    openai_resp = _make_openai_response(content="Hello")
    result = convert_openai_to_responses(
        openai_resp, request_model="gpt-4", previous_id="prev_123"
    )
    assert result.previous_id == "prev_123"


# --------------------------------------------------------------------------
# 7. Tool call remapping with custom tools (via tool_ctx)
# --------------------------------------------------------------------------
def test_tool_call_remapping_custom_tools():
    """If tool_ctx.remap_tool_calls_back is present it is invoked."""
    openai_resp = _make_openai_response(
        content="",
        tool_calls=[
            {
                "id": "call_3",
                "type": "function",
                "function": {
                    "name": "apply_patch_add_file",
                    "arguments": '{"path": "/tmp/test.py", "content": "print(1)"}',
                },
            }
        ],
    )

    def remap(items):
        for item in items:
            if item.type == "function_call" and item.name == "apply_patch_add_file":
                item.name = "apply_patch"
        return items

    tool_ctx = MagicMock()
    tool_ctx.remap_tool_calls_back = remap

    result = convert_openai_to_responses(
        openai_resp, request_model="gpt-4", tool_ctx=tool_ctx
    )
    assert result.output[0].name == "apply_patch"


# --------------------------------------------------------------------------
# 8. Tool call remapping with namespace tools
# --------------------------------------------------------------------------
def test_tool_call_remapping_namespace_tools():
    """Namespace tool proxy names are remapped back by tool_ctx."""
    openai_resp = _make_openai_response(
        content="",
        tool_calls=[
            {
                "id": "call_4",
                "type": "function",
                "function": {
                    "name": "mcp__read_file",
                    "arguments": '{"path": "/etc/hosts"}',
                },
            }
        ],
    )

    def remap(items):
        for item in items:
            if item.type == "function_call" and item.name.startswith("mcp__"):
                item.name = item.name.replace("mcp__", "")
                item.namespace = "mcp__"
        return items

    tool_ctx = MagicMock()
    tool_ctx.remap_tool_calls_back = remap

    result = convert_openai_to_responses(
        openai_resp, request_model="gpt-4", tool_ctx=tool_ctx
    )
    assert result.output[0].name == "read_file"
    assert result.output[0].namespace == "mcp__"


# --------------------------------------------------------------------------
# 9. Empty response -> empty message item
# --------------------------------------------------------------------------
def test_empty_response_fallback():
    """When both content and tool_calls are empty/None, a fallback empty message item is created."""
    openai_resp = _make_openai_response(content="")
    result = convert_openai_to_responses(openai_resp, request_model="gpt-4")

    assert len(result.output) == 1
    assert result.output[0].type == "message"
    assert result.output[0].content == ""
    assert result.output[0].role == "assistant"


# --------------------------------------------------------------------------
# 10. Error response conversion
# --------------------------------------------------------------------------
def test_error_response_conversion():
    """convert_openai_error_to_responses produces a failed status response."""
    error = Exception("Upstream timeout")
    result = convert_openai_error_to_responses(error)

    assert result.status == "failed"
    assert result.model == "unknown"
    assert len(result.output) == 1
    assert result.output[0].type == "message"
    assert result.output[0].role == "assistant"
    assert "Upstream timeout" in result.output[0].content
    assert result.usage.input_tokens == 0
    assert result.usage.output_tokens == 0
    assert result.usage.total_tokens == 0


# --------------------------------------------------------------------------
# 11. UUID generation
# --------------------------------------------------------------------------
def test_id_is_uuid():
    """The response id is a valid UUID string."""
    openai_resp = _make_openai_response(content="Yo")
    result = convert_openai_to_responses(openai_resp, request_model="gpt-4")
    # Should not raise
    uuid.UUID(result.id)
