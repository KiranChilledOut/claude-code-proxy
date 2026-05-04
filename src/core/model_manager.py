from src.core.config import config
from src.models.claude import ClaudeMessage


def _build_alias_map(cfg) -> dict:
    """Map the short aliases users can type into Claude Code's /model picker
    (e.g. `/model glm`) to the upstream model strings we forward to the
    OpenAI-compatible endpoint."""
    return {
        "opus": cfg.big_model,
        "sonnet": cfg.middle_model,
        "haiku": cfg.small_model,
        "glm": cfg.glm_model,
        "kimi": cfg.kimi_model,
        "gemma": cfg.gemma_model,
        "qwen": cfg.qwen_model,
        "nemotron": cfg.nemotron_model,
        "super": cfg.nemotron_super_model,
        "nano": cfg.nemotron_nano_model,
        "minimax": cfg.minimax_model,
        "hermes": cfg.hermes_model,
        "gpt": cfg.gpt_model,
        "llama": cfg.llama_model,
        "prime": cfg.prime_model,
        "deepseek": cfg.deepseek_model,
    }


class ModelManager:
    def __init__(self, config):
        self.config = config
        self.aliases = _build_alias_map(config)

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
                        if block.get("type") in ("image", "image_url") or "image_url" in block:
                            return True
                    else:
                        # Check if it's a ClaudeContentBlockImage object
                        if hasattr(block, "type") and block.type in ("image", "image_url"):
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
        if (
            claude_model.startswith("ep-")
            or claude_model.startswith("doubao-")
            or claude_model.startswith("deepseek-")
        ):
            return claude_model

        # Token Factory / HF-style ids contain a "/" (e.g.
        # "meta-llama/Llama-3.3-70B-Instruct"). Pass them through verbatim so
        # users can pick any catalog entry directly without falling back to
        # BIG_MODEL.
        if "/" in claude_model:
            return claude_model

        model_lower = claude_model.lower()

        # Exact alias (e.g. `/model glm` -> glm_model). The opus/sonnet/haiku
        # entries here also short-circuit the keyword block below for the
        # bare-alias case, but keyword matching still handles full ids like
        # `claude-opus-4-5`.
        if model_lower in self.aliases:
            return self.aliases[model_lower]

        # Non-Claude keyword aliases inside a longer string (e.g. `glm-5`,
        # `kimi-2.5`). Checked before the haiku/sonnet/opus block so a
        # provider keyword wins over the generic fallback.
        for keyword in ("glm", "kimi", "gemma"):
            if keyword in model_lower:
                return self.aliases[keyword]

        # Map based on model naming patterns
        if "haiku" in model_lower:
            return self.config.small_model
        elif "sonnet" in model_lower:
            return self.config.middle_model
        elif "opus" in model_lower:
            return self.config.big_model
        else:
            # Default to big model for unknown models
            return self.config.big_model


model_manager = ModelManager(config)
