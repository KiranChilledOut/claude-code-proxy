from src.core.config import config
from src.models.claude import ClaudeMessage

class ModelManager:
    def __init__(self, config):
        self.config = config

    def contains_image_content(self, messages, *, latest_user_only: bool = False) -> bool:
        """Check if any (or just the latest user) message contains image content"""
        iterable = messages
        if latest_user_only:
            # Walk backward to the most recent user message only
            for message in reversed(messages):
                role = message.get("role") if isinstance(message, dict) else message.role
                if role == "user":
                    iterable = [message]
                    break

        for message in iterable:
            if isinstance(message, dict):
                content = message.get("content", [])
            else:
                content = message.content
            if isinstance(content, dict):
                content = [content]

            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get('type') in ('image', 'image_url') or 'image_url' in block:
                            return True
                    else:
                        # Check if it's a ClaudeContentBlockImage object
                        if hasattr(block, 'type') and block.type in ('image', 'image_url'):
                            return True
        return False

    def map_claude_model_to_openai(self, claude_model: str, messages=None) -> str:
        """Map Claude model names to OpenAI model names based on BIG/SMALL pattern"""

        # If messages contain images, route to vision model
        if messages and self.contains_image_content(messages, latest_user_only=True):
            return self.config.vision_model

        # If it's already an OpenAI model, return as-is
        if claude_model.startswith("gpt-") or claude_model.startswith("o1-"):
            return claude_model

        # If it's other supported models (ARK/Doubao/DeepSeek), return as-is
        if (claude_model.startswith("ep-") or claude_model.startswith("doubao-") or
            claude_model.startswith("deepseek-")):
            return claude_model

        # Map based on model naming patterns
        model_lower = claude_model.lower()
        if 'haiku' in model_lower:
            return self.config.small_model
        elif 'sonnet' in model_lower:
            return self.config.middle_model
        elif 'opus' in model_lower:
            return self.config.big_model
        else:
            # Default to big model for unknown models
            return self.config.big_model

model_manager = ModelManager(config)
