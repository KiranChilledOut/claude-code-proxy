#!/usr/bin/env python3
"""Debug script to test image detection and model routing"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv()

from src.core.config import Config
from src.core.model_manager import ModelManager
from src.models.claude import ClaudeMessage, ClaudeContentBlockText, ClaudeContentBlockImage

def main():
    print("=== Debug Image Routing ===")

    # Load config
    config = Config()
    print(f"BIG_MODEL: {config.big_model}")
    print(f"MIDDLE_MODEL: {config.middle_model}")
    print(f"SMALL_MODEL: {config.small_model}")
    print(f"VISION_MODEL: {config.vision_model}")

    # Create model manager
    model_manager = ModelManager(config)

    # Test 1: Text only
    text_message = ClaudeMessage(
        role="user",
        content="Hello, just text"
    )

    model_for_text = model_manager.map_claude_model_to_openai("claude-3-sonnet", [text_message])
    print(f"Text message routes to: {model_for_text}")

    # Test 2: Image
    image_message = ClaudeMessage(
        role="user",
        content=[
            ClaudeContentBlockText(type="text", text="What's in this image?"),
            ClaudeContentBlockImage(
                type="image",
                source={
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "fake_base64_data"
                }
            )
        ]
    )

    model_for_image = model_manager.map_claude_model_to_openai("claude-3-sonnet", [image_message])
    print(f"Image message routes to: {model_for_image}")

    # Test image detection
    has_image = model_manager.contains_image_content([image_message])
    print(f"Image detected in message: {has_image}")

if __name__ == "__main__":
    main()