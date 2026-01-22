import base64
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.conversion.request_converter import convert_claude_to_openai
from src.core.model_manager import model_manager
from src.core.config import config
from src.models.claude import (
    ClaudeMessagesRequest,
    ClaudeMessage,
    ClaudeContentBlockText,
    ClaudeContentBlockImage,
)


def test_image_request_sets_tool_choice_none_when_no_tools():
    """Vision requests should explicitly disable tools to appease providers that reject auto mode."""
    image_data = base64.b64encode(b"fake").decode("utf-8")
    request = ClaudeMessagesRequest(
        model="claude-3-5-sonnet-20241022",
        max_tokens=256,
        messages=[
            ClaudeMessage(
                role="user",
                content=[
                    ClaudeContentBlockText(type="text", text="what's in this image ?"),
                    ClaudeContentBlockImage(
                        type="image",
                        source={
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_data,
                        },
                    ),
                ],
            )
        ],
    )

    openai_request = convert_claude_to_openai(request, model_manager)

    assert openai_request["model"] == config.vision_model
    assert openai_request.get("tool_choice") == "none"
    assert "tools" not in openai_request


def test_text_request_does_not_force_tool_choice_without_tools():
    request = ClaudeMessagesRequest(
        model="claude-3-5-sonnet-20241022",
        max_tokens=64,
        messages=[ClaudeMessage(role="user", content="hello")],
    )

    openai_request = convert_claude_to_openai(request, model_manager)

    assert "tool_choice" not in openai_request
    assert "tools" not in openai_request


def test_image_request_drops_tools_and_sets_none():
    """Image requests should drop tools and force tool_choice none."""
    image_data = base64.b64encode(b"fake").decode("utf-8")
    request = ClaudeMessagesRequest(
        model="claude-3-5-sonnet-20241022",
        max_tokens=256,
        messages=[
            ClaudeMessage(
                role="user",
                content=[
                    ClaudeContentBlockText(type="text", text="what's in this image ?"),
                    ClaudeContentBlockImage(
                        type="image",
                        source={
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_data,
                        },
                    ),
                ],
            )
        ],
        tools=[
            {
                "name": "dummy",
                "description": "ignore",
                "input_schema": {"type": "object"},
            }
        ],
        tool_choice={"type": "auto"},
    )

    openai_request = convert_claude_to_openai(request, model_manager)

    assert openai_request["model"] == config.vision_model
    # Should drop tools and force tool_choice none
    assert "tools" not in openai_request
    assert openai_request.get("tool_choice") == "none"


def test_followup_without_image_keeps_tools_and_model():
    """If the latest user message has no image, keep tools and default model."""
    image_data = base64.b64encode(b"fake").decode("utf-8")
    # Conversation history includes an earlier image, but latest user is text-only
    request = ClaudeMessagesRequest(
        model="claude-3-5-sonnet-20241022",
        max_tokens=128,
        messages=[
            ClaudeMessage(
                role="user",
                content=[
                    ClaudeContentBlockText(type="text", text="what's in this image ?"),
                    ClaudeContentBlockImage(
                        type="image",
                        source={
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_data,
                        },
                    ),
                ],
            ),
            ClaudeMessage(
                role="assistant",
                content="ALL DONE"
            ),
            ClaudeMessage(
                role="user",
                content="create a file called test"
            ),
        ],
        tools=[
            {
                "name": "dummy",
                "description": "ignore",
                "input_schema": {"type": "object"},
            }
        ],
        tool_choice={"type": "auto"},
    )

    openai_request = convert_claude_to_openai(request, model_manager)

    assert openai_request["model"] == config.middle_model
    assert openai_request.get("tools")
    assert openai_request.get("tool_choice") == "auto"
    # No image content should remain
    assert not any(
        isinstance(msg.get("content"), list) and any(part.get("type") == "image_url" for part in msg["content"])
        for msg in openai_request["messages"]
    )
