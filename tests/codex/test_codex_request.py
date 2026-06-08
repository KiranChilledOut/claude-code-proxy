"""Tests for Codex ResponsesRequest -> OpenAI Chat Completions conversion."""

import pytest

from src.codex.models import ResponsesItem, ResponsesRequest
from src.codex.request_converter import (
    _map_reasoning_effort,
    _convert_item_to_message,
    _convert_content_blocks,
    _default_map_codex_model,
    convert_responses_to_openai_chat,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_request(**kwargs) -> ResponsesRequest:
    defaults = {"model": "gpt-4", "input": "Hello"}
    defaults.update(kwargs)
    return ResponsesRequest.model_validate(defaults)


# ---------------------------------------------------------------------------
# _convert_content_blocks
# ---------------------------------------------------------------------------
def test_convert_content_string():
    """Pass-through for plain string content."""
    assert _convert_content_blocks("Hello") == "Hello"


def test_convert_content_text_blocks():
    """Concatenate text blocks from a list."""
    blocks = [
        {"type": "text", "text": "Hello "},
        {"type": "text", "text": "world"},
    ]
    assert _convert_content_blocks(blocks) == "Hello world"


def test_convert_content_responses_text_blocks():
    """Codex Responses messages use input_text/output_text block types."""
    blocks = [
        {"type": "input_text", "text": "Hello "},
        {"type": "output_text", "text": "world"},
    ]
    assert _convert_content_blocks(blocks) == "Hello world"


def test_convert_content_converts_images():
    """Image blocks are normalized to OpenAI Chat Completions image_url parts."""
    blocks = [
        {"type": "text", "text": "Look: "},
        {"type": "input_image", "image_url": "http://example.com/img.png", "detail": "high"},
        {"type": "text", "text": "Done"},
    ]
    assert _convert_content_blocks(blocks) == [
        {"type": "text", "text": "Look: "},
        {
            "type": "image_url",
            "image_url": {"url": "http://example.com/img.png", "detail": "high"},
        },
        {"type": "text", "text": "Done"},
    ]


def test_convert_content_converts_base64_image_blocks():
    """Claude-style base64 image blocks are also normalized for Codex history."""
    blocks = [
        {"type": "text", "text": "Look"},
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "abc123"},
        },
    ]
    assert _convert_content_blocks(blocks) == [
        {"type": "text", "text": "Look"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
    ]


def test_convert_content_none():
    """Missing content becomes an empty string."""
    assert _convert_content_blocks(None) == ""


# ---------------------------------------------------------------------------
# _convert_item_to_message
# ---------------------------------------------------------------------------
def test_item_user_message():
    """message + user role -> user message."""
    item = ResponsesItem(type="message", role="user", content="Hi")
    msg = _convert_item_to_message(item)
    assert msg == {"role": "user", "content": "Hi"}


def test_item_assistant_message():
    """message + assistant role -> assistant message."""
    item = ResponsesItem(type="message", role="assistant", content="Sure")
    msg = _convert_item_to_message(item)
    assert msg == {"role": "assistant", "content": "Sure"}


def test_item_system_message_maps_to_system():
    """system role messages are preserved as OpenAI Chat system messages."""
    item = ResponsesItem(type="message", role="system", content="stuff")
    assert _convert_item_to_message(item) == {"role": "system", "content": "stuff"}


def test_item_developer_message_maps_to_system():
    """Codex developer messages carry runtime instructions; do not drop them."""
    item = ResponsesItem(
        type="message",
        role="developer",
        content=[{"type": "input_text", "text": "Follow sandbox rules"}],
    )
    assert _convert_item_to_message(item) == {
        "role": "system",
        "content": "Follow sandbox rules",
    }


def test_item_unknown_message_role_is_dropped():
    """A message without a known role cannot be mapped."""
    item = ResponsesItem(type="message", role="invalid", content="stuff")
    assert _convert_item_to_message(item) is None


def test_item_function_call():
    """function_call -> assistant message with tool_calls."""
    item = ResponsesItem(
        type="function_call",
        call_id="call_123",
        name="search",
        arguments='{"q":"test"}',
    )
    msg = _convert_item_to_message(item)
    assert msg == {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "search",
                    "arguments": '{"q":"test"}',
                },
            }
        ],
    }


def test_item_function_call_output():
    """function_call_output -> tool message."""
    item = ResponsesItem(
        type="function_call_output",
        call_id="call_123",
        output="42",
    )
    msg = _convert_item_to_message(item)
    assert msg == {
        "role": "tool",
        "content": "42",
        "tool_call_id": "call_123",
    }


def test_item_function_call_output_with_image():
    """view_image-style tool output is converted instead of forwarded as input_image."""
    item = ResponsesItem(
        type="function_call_output",
        call_id="call_img",
        output=[
            {
                "type": "input_image",
                "image_url": "data:image/png;base64,abc123",
                "detail": "high",
            }
        ],
    )
    msg = _convert_item_to_message(item)
    assert msg == {
        "role": "tool",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,abc123",
                    "detail": "high",
                },
            }
        ],
        "tool_call_id": "call_img",
    }


def test_item_text():
    """text item type -> user message with the text field."""
    item = ResponsesItem(type="text", text="hello", content=None)
    msg = _convert_item_to_message(item)
    assert msg == {"role": "user", "content": "hello"}


# ---------------------------------------------------------------------------
# _map_reasoning_effort
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "effort,expected",
    [
        ("none", "none"),
        ("auto", "auto"),
        ("minimal", "low"),
        ("low", "low"),
        ("medium", "medium"),
        ("high", "high"),
        ("xhigh", "high"),
        ("unknown", "auto"),
        (None, "auto"),
    ],
)
def test_map_reasoning_effort(effort, expected):
    if effort is None:
        assert _map_reasoning_effort(None) == expected
    else:
        assert _map_reasoning_effort({"effort": effort}) == expected


# ---------------------------------------------------------------------------
# Model mapping fallback
# ---------------------------------------------------------------------------
def test_default_map_mini():
    # Fallback mapping: mini -> small_model -> "zai-org/GLM-4.5" (default)
    assert _default_map_codex_model("o1-mini").lower() == "zai-org/glm-4.5"


def test_default_map_gpt():
    r = _default_map_codex_model("gpt-4")
    assert isinstance(r, str)
    assert len(r) > 0


# ---------------------------------------------------------------------------
# Main conversion: basic cases
# ---------------------------------------------------------------------------
def test_string_input():
    """(1) String input becomes a single user message."""
    request = _make_request(model="gpt-4", input="Hello, world!")
    result = convert_responses_to_openai_chat(request)
    assert result["model"] == "zai-org/GLM-4.5"
    assert result["messages"] == [{"role": "user", "content": "Hello, world!"}]
    assert result["stream"] is False


def test_instructions():
    """(2) Instructions → system message prepended before user message."""
    request = _make_request(
        model="gpt-4",
        instructions="Be helpful",
        input="Help me",
    )
    result = convert_responses_to_openai_chat(request)
    assert result["messages"][0] == {"role": "system", "content": "Be helpful"}
    assert result["messages"][1] == {"role": "user", "content": "Help me"}


def test_array_input_user_message():
    """(3) Array input with user message item."""
    request = _make_request(
        input=[
            {"type": "message", "role": "user", "content": "Turn left"},
        ],
    )
    result = convert_responses_to_openai_chat(request)
    assert result["messages"] == [{"role": "user", "content": "Turn left"}]


def test_array_input_assistant_message():
    """(4) Array input with assistant message item."""
    request = _make_request(
        input=[
            {"type": "message", "role": "assistant", "content": "Sure thing"},
        ],
    )
    result = convert_responses_to_openai_chat(request)
    assert result["messages"] == [{"role": "assistant", "content": "Sure thing"}]


def test_array_input_codex_developer_and_user_input_text():
    """Real Codex CLI requests use developer messages and input_text blocks."""
    request = _make_request(
        input=[
            {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": "Use the repo rules."}],
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "What is today's date?"}],
            },
        ],
    )
    result = convert_responses_to_openai_chat(request)
    assert result["messages"] == [
        {"role": "system", "content": "Use the repo rules."},
        {"role": "user", "content": "What is today's date?"},
    ]


def test_array_input_function_call():
    """(5) Array input with function_call → assistant tool_calls message."""
    request = _make_request(
        input=[
            {
                "type": "function_call",
                "call_id": "fc_1",
                "name": "search",
                "arguments": '{"q":"test"}',
            },
        ],
    )
    result = convert_responses_to_openai_chat(request)
    assert result["messages"][0] == {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "fc_1",
                "type": "function",
                "function": {
                    "name": "search",
                    "arguments": '{"q":"test"}',
                },
            }
        ],
    }


def test_array_input_function_call_output():
    """(6) Array input with function_call_output -> tool message."""
    request = _make_request(
        input=[
            {
                "type": "function_call_output",
                "call_id": "fc_1",
                "output": "42",
            },
        ],
    )
    result = convert_responses_to_openai_chat(request)
    assert result["messages"][0] == {
        "role": "tool",
        "content": "42",
        "tool_call_id": "fc_1",
    }


def test_session_items_prepended():
    """(7) Session items are placed after system but before request.input."""
    request = _make_request(
        model="gpt-4",
        instructions="System prompt",
        input="Now?",
    )
    session = [
        {"role": "user", "content": "First turn"},
        {"role": "assistant", "content": "Got it"},
    ]
    result = convert_responses_to_openai_chat(request, session_items=session)
    assert result["messages"] == [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "First turn"},
        {"role": "assistant", "content": "Got it"},
        {"role": "user", "content": "Now?"},
    ]


def test_model_mapping_with_model_manager():
    """(8) When a model manager is supplied its map_codex_model is called."""

    class FakeManager:
        def map_codex_model(self, name):
            return f"mapped:{name}"

    request = _make_request(model="gpt-4")
    result = convert_responses_to_openai_chat(request, model_manager=FakeManager())
    assert result["model"] == "mapped:gpt-4"


def test_image_request_routes_to_vision_model_and_disables_tools():
    """Codex image requests follow the same vision routing policy as Claude requests."""

    class FakeManager:
        def __init__(self):
            self.config = type("Cfg", (), {"vision_model": "vision-model"})()

        def map_codex_model(self, name):
            return "text-model"

        def contains_image_content(self, messages, latest_user_only=False):
            for message in messages:
                content = message.get("content")
                if isinstance(content, list):
                    if any(part.get("type") == "image_url" for part in content):
                        return True
            return False

    request = _make_request(
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "what is this?"},
                    {"type": "input_image", "image_url": "data:image/png;base64,abc123"},
                ],
            }
        ],
        tools=[{"type": "function", "function": {"name": "noop", "parameters": {}}}],
        tool_choice="auto",
    )

    result = convert_responses_to_openai_chat(request, model_manager=FakeManager())

    assert result["model"] == "vision-model"
    assert "tools" not in result
    assert result["tool_choice"] == "none"
    assert result["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
            ],
        }
    ]


def test_model_mapping_no_manager():
    """(8 cont'd) No model_manager -> fallback mapping kicks in."""
    request = _make_request(model="gpt-4")
    result = convert_responses_to_openai_chat(request)
    # default fallback -> GLM-4.5
    assert result["model"] == "zai-org/GLM-4.5"

    request_mini = _make_request(model="o1-mini")
    result_mini = convert_responses_to_openai_chat(request_mini)
    assert "mini" in result_mini["model"].lower() or "glm" in result_mini["model"].lower()


# ---------------------------------------------------------------------------
# Generation params
# ---------------------------------------------------------------------------
def test_passthrough_params():
    """(9) max_output_tokens, temperature, and top_p are forwarded."""
    request = _make_request(
        max_output_tokens=1024,
        temperature=0.7,
        top_p=0.9,
    )
    result = convert_responses_to_openai_chat(request)
    assert result["max_tokens"] == 1024
    assert result["temperature"] == 0.7
    assert result["top_p"] == 0.9


def test_streaming_and_stream_options():
    """(10) stream=True sets stream_options.include_usage."""
    request = _make_request(stream=True, input="Count to 5")
    result = convert_responses_to_openai_chat(request)
    assert result["stream"] is True
    assert result["stream_options"] == {"include_usage": True}


def test_non_streaming_no_stream_options():
    request = _make_request(stream=False)
    result = convert_responses_to_openai_chat(request)
    assert "stream_options" not in result


# ---------------------------------------------------------------------------
# Reasoning
# ---------------------------------------------------------------------------
def test_reasoning_effort_mapping():
    """(11) reasoning.effort maps to reasoning_effort."""
    request = _make_request(reasoning={"effort": "high"})
    result = convert_responses_to_openai_chat(request)
    assert result["reasoning_effort"] == "high"

    request2 = _make_request(reasoning={"effort": "minimal"})
    result2 = convert_responses_to_openai_chat(request2)
    assert result2["reasoning_effort"] == "low"


def test_reasoning_absent_no_reasoning_effort():
    request = _make_request()
    result = convert_responses_to_openai_chat(request)
    assert "reasoning_effort" not in result


# ---------------------------------------------------------------------------
# Tool choice passthrough
# ---------------------------------------------------------------------------
def test_tool_choice_passthrough():
    """(12) Simple string tool_choice values are passed through."""
    for choice in ("auto", "required", "none"):
        request = _make_request(tool_choice=choice)
        result = convert_responses_to_openai_chat(request)
        assert result["tool_choice"] == choice

    # Object form passthrough when no tool_ctx is present
    request = _make_request(tool_choice={"type": "function", "function": {"name": "foo"}})
    result = convert_responses_to_openai_chat(request)
    assert result["tool_choice"] == {
        "type": "function",
        "function": {"name": "foo"},
    }


# ---------------------------------------------------------------------------
# Minimal / empty request
# ---------------------------------------------------------------------------
def test_empty_request():
    """(13) Just model + input produces valid output with minimal fields."""
    request = ResponsesRequest(model="gpt-4", input="hi")
    result = convert_responses_to_openai_chat(request)
    assert result["model"]
    assert len(result["messages"]) == 1
    assert result["messages"][0] == {"role": "user", "content": "hi"}
    assert "stream" in result


# ---------------------------------------------------------------------------
# User field passthrough
# ---------------------------------------------------------------------------
def test_user_field():
    request = _make_request(user="user_123")
    result = convert_responses_to_openai_chat(request)
    assert result["user"] == "user_123"


# ---------------------------------------------------------------------------
# Session items as ResponseItem objects (not just dicts)
# ---------------------------------------------------------------------------
def test_session_items_as_objects():
    """Session items may be ResponsesItem objects rather than plain dicts."""
    request = _make_request(input="Follow-up")
    session = [
        ResponsesItem(type="message", role="user", content="First"),
        ResponsesItem(type="message", role="assistant", content="Answer"),
    ]
    result = convert_responses_to_openai_chat(request, session_items=session)
    assert result["messages"] == [
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "Answer"},
        {"role": "user", "content": "Follow-up"},
    ]


# ---------------------------------------------------------------------------
# Search tool system supplement injection
# ---------------------------------------------------------------------------
def test_search_supplement_injected():
    """When web_search tool is present, SEARCH_TOOL_SYSTEM_SUPPLEMENT is injected."""
    from unittest.mock import MagicMock

    fake_tool_ctx = MagicMock()
    fake_tool_ctx.has_search_tool = True
    fake_tool_ctx.tools = [{"type": "function", "function": {"name": "web_search"}}]
    fake_tool_ctx.map_tool_choice.return_value = None

    request = _make_request(input="What is the weather?")
    result = convert_responses_to_openai_chat(
        request,
        tool_ctx=fake_tool_ctx,
    )
    first_msg = result["messages"][0]
    assert first_msg["role"] == "system"
    assert "call the web search tool" in first_msg["content"]


def test_search_supplement_not_injected_without_tool():
    """When no search tool is present, no extra system supplement is added."""
    request = _make_request(input="What is 2+2?")
    result = convert_responses_to_openai_chat(request)
    if result["messages"] and result["messages"][0]["role"] == "system":
        assert "call the web search" not in result["messages"][0]["content"]


def test_empty_tools_dropped():
    """When tool_ctx returns an empty tools list, it must not appear in the output
    to avoid 400 from backends that reject empty tool arrays."""
    from unittest.mock import MagicMock

    fake_tool_ctx = MagicMock()
    fake_tool_ctx.tools = []
    fake_tool_ctx.map_tool_choice.return_value = None

    request = _make_request(input="What is 2+2?")
    result = convert_responses_to_openai_chat(request, tool_ctx=fake_tool_ctx)
    assert "tools" not in result
