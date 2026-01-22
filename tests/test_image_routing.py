from src.core.config import config
from src.core.model_manager import ModelManager
from src.models.claude import (
    ClaudeContentBlockImage,
    ClaudeContentBlockText,
    ClaudeMessage,
)


def test_image_routes_to_vision_model():
    model_manager = ModelManager(config)

    text_message = ClaudeMessage(role="user", content="Hello")
    image_message = ClaudeMessage(
        role="user",
        content=[
            ClaudeContentBlockText(type="text", text="What is in this image?"),
            ClaudeContentBlockImage(
                type="image",
                source={
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
                },
            ),
        ],
    )

    assert (
        model_manager.map_claude_model_to_openai("claude-3-sonnet", [text_message])
        == config.middle_model
    )
    assert (
        model_manager.map_claude_model_to_openai("claude-3-sonnet", [image_message])
        == config.vision_model
    )
