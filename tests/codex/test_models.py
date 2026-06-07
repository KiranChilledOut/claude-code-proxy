"""Tests for Codex Pydantic models."""

import pytest
from pydantic import ValidationError

from src.codex.models import (
    InputTokensDetails,
    OutputTokensDetails,
    ResponsesItem,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesStreamEvent,
    ResponsesUsage,
)


# --------------------------------------------------------------------------
# ResponsesRequest
# --------------------------------------------------------------------------
def test_parse_minimal_request():
    """Parse a minimal ResponsesRequest with string input."""
    req = ResponsesRequest.model_validate({
        "model": "gpt-4",
        "input": "Hello, world!",
    })
    assert req.model == "gpt-4"
    assert req.input == "Hello, world!"
    assert req.stream is False
    assert req.instructions is None


def test_parse_request_with_array_input():
    """Parse a ResponsesRequest with input as array of items."""
    req = ResponsesRequest.model_validate({
        "model": "gpt-4",
        "input": [
            {"type": "message", "role": "user", "content": "Hello"},
            {"type": "message", "role": "assistant", "content": "Hi there"},
        ],
        "stream": True,
        "instructions": "Be helpful",
        "temperature": 0.7,
    })
    assert req.model == "gpt-4"
    assert isinstance(req.input, list)
    assert len(req.input) == 2
    assert req.input[0].type == "message"
    assert req.input[0].role == "user"
    assert req.input[0].content == "Hello"
    assert req.input[1].role == "assistant"
    assert req.instructions == "Be helpful"
    assert req.temperature == 0.7


def test_parse_request_all_fields():
    """Parse a ResponsesRequest with all optional fields populated."""
    req = ResponsesRequest.model_validate({
        "model": "gpt-3.5-turbo",
        "instructions": "System prompt",
        "input": "test",
        "previous_response_id": "resp_123",
        "store": True,
        "max_output_tokens": 1024,
        "temperature": 0.5,
        "top_p": 0.9,
        "stream": False,
        "tools": [{"type": "function", "function": {"name": "search"}}],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "reasoning": {"effort": "medium"},
        "user": "user_42",
        "metadata": {"key": "value"},
    })
    assert req.previous_response_id == "resp_123"
    assert req.max_output_tokens == 1024
    assert req.parallel_tool_calls is True
    assert req.reasoning == {"effort": "medium"}
    assert req.metadata == {"key": "value"}


# --------------------------------------------------------------------------
# ResponsesResponse
# --------------------------------------------------------------------------
def test_parse_response():
    """Parse a ResponsesResponse with output items and usage."""
    data = {
        "id": "resp_abc",
        "model": "gpt-4",
        "output": [
            {"type": "message", "role": "assistant", "content": "Hello!", "status": "completed"},
            {"type": "function_call", "call_id": "fc_1", "name": "search", "arguments": '{"q":"test"}'},
        ],
        "status": "completed",
        "previous_id": "resp_prev",
        "usage": {
            "input_tokens": 50,
            "output_tokens": 20,
            "total_tokens": 70,
            "input_tokens_details": {"cached_tokens": 10},
            "output_tokens_details": {"reasoning_tokens": 5},
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 10,
        },
    }
    resp = ResponsesResponse.model_validate(data)
    assert resp.id == "resp_abc"
    assert resp.status == "completed"
    assert len(resp.output) == 2
    assert resp.output[0].type == "message"
    assert resp.output[0].content == "Hello!"
    assert resp.output[1].type == "function_call"
    assert resp.output[1].call_id == "fc_1"
    assert resp.output[1].arguments == '{"q":"test"}'
    assert resp.usage.input_tokens == 50
    assert resp.usage.total_tokens == 70
    assert resp.usage.input_tokens_details.cached_tokens == 10
    assert resp.usage.output_tokens_details.reasoning_tokens == 5


# --------------------------------------------------------------------------
# ResponsesStreamEvent
# --------------------------------------------------------------------------
def test_parse_stream_event_created():
    """Parse a response.created stream event."""
    event = ResponsesStreamEvent.model_validate({
        "type": "response.created",
        "id": "resp_abc",
        "model": "gpt-4",
    })
    assert event.type == "response.created"
    assert event.id == "resp_abc"
    assert event.model == "gpt-4"


def test_parse_stream_event_text_delta():
    """Parse a response.output_text.delta stream event."""
    event = ResponsesStreamEvent.model_validate({
        "type": "response.output_text.delta",
        "output_index": 0,
        "delta": "Hello",
    })
    assert event.type == "response.output_text.delta"
    assert event.output_index == 0
    assert event.delta == "Hello"


def test_parse_stream_event_item_added():
    """Parse a response.output_item.added stream event."""
    event = ResponsesStreamEvent.model_validate({
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {
            "type": "function_call",
            "call_id": "fc_1",
            "name": "search",
        },
    })
    assert event.type == "response.output_item.added"
    assert event.item is not None
    assert event.item.type == "function_call"
    assert event.item.call_id == "fc_1"


def test_parse_stream_event_completed():
    """Parse a response.completed stream event with usage."""
    event = ResponsesStreamEvent.model_validate({
        "type": "response.completed",
        "id": "resp_abc",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        },
    })
    assert event.type == "response.completed"
    assert event.usage is not None
    assert event.usage.input_tokens == 10


# --------------------------------------------------------------------------
# Forward compat / extra fields
# --------------------------------------------------------------------------
def test_extra_fields_are_ignored():
    """Unknown fields should be silently ignored (forward compat)."""
    req = ResponsesRequest.model_validate({
        "model": "gpt-4",
        "input": "hi",
        "some_future_field": 123,
        "nested_unknown": {"a": "b"},
    })
    assert req.model == "gpt-4"
    assert req.input == "hi"


def test_extra_fields_on_item():
    """Extra fields on ResponsesItem are silently ignored."""
    item = ResponsesItem.model_validate({
        "type": "message",
        "role": "user",
        "content": "hello",
        "future_field": True,
    })
    assert item.type == "message"
    assert item.role == "user"


# --------------------------------------------------------------------------
# Invalid types should raise ValidationError
# --------------------------------------------------------------------------
def test_missing_model_raises():
    """model is required on ResponsesRequest."""
    with pytest.raises(ValidationError):
        ResponsesRequest.model_validate({"input": "hello"})


def test_missing_input_raises():
    """input is required on ResponsesRequest."""
    with pytest.raises(ValidationError):
        ResponsesRequest.model_validate({"model": "gpt-4"})


def test_invalid_temperature_type():
    """temperature must be a float, not a string."""
    with pytest.raises(ValidationError):
        ResponsesRequest.model_validate({
            "model": "gpt-4",
            "input": "hello",
            "temperature": "hot",
        })


def test_invalid_usage_fields():
    """Usage requires numeric token counts."""
    with pytest.raises(ValidationError):
        ResponsesUsage.model_validate({
            "input_tokens": "many",
            "output_tokens": 5,
            "total_tokens": 5,
        })


# --------------------------------------------------------------------------
# ResponsesItem types
# --------------------------------------------------------------------------
def test_item_with_namespace():
    """Parse an item with namespace (used for namespace tools)."""
    item = ResponsesItem.model_validate({
        "type": "function_call",
        "namespace": "mcp__",
        "name": "some_tool",
        "arguments": '{"q":"test"}',
    })
    assert item.namespace == "mcp__"


def test_item_with_output():
    """Parse a function_call_output item."""
    item = ResponsesItem.model_validate({
        "type": "function_call_output",
        "call_id": "fc_1",
        "output": "result data",
    })
    assert item.type == "function_call_output"
    assert item.output == "result data"


def test_item_content_as_list():
    """Parse an item with content as a list of dicts."""
    item = ResponsesItem.model_validate({
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Hello"},
            {"type": "image", "url": "http://example.com/img.png"},
        ],
    })
    assert isinstance(item.content, list)
    assert len(item.content) == 2
    assert item.content[0]["type"] == "text"
