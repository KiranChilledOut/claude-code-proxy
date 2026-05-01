from src.core.config import config
from src.core.model_manager import ModelManager


def test_alias_exact_match_routes_to_provider_model(monkeypatch):
    monkeypatch.setattr(config, "glm_model", "zai-org/GLM-4.5")
    monkeypatch.setattr(config, "kimi_model", "moonshotai/Kimi-K2-Instruct")
    monkeypatch.setattr(config, "gemma_model", "google/gemma-3-27b-it")
    mm = ModelManager(config)

    assert mm.map_claude_model_to_openai("glm") == "zai-org/GLM-4.5"
    assert mm.map_claude_model_to_openai("kimi") == "moonshotai/Kimi-K2-Instruct"
    assert mm.map_claude_model_to_openai("gemma") == "google/gemma-3-27b-it"


def test_alias_keyword_in_longer_id(monkeypatch):
    monkeypatch.setattr(config, "glm_model", "zai-org/GLM-4.5")
    monkeypatch.setattr(config, "kimi_model", "moonshotai/Kimi-K2-Instruct")
    monkeypatch.setattr(config, "gemma_model", "google/gemma-3-27b-it")
    mm = ModelManager(config)

    assert mm.map_claude_model_to_openai("glm-5") == "zai-org/GLM-4.5"
    assert mm.map_claude_model_to_openai("Glm-Beta") == "zai-org/GLM-4.5"
    assert mm.map_claude_model_to_openai("kimi-2.5") == "moonshotai/Kimi-K2-Instruct"


def test_existing_claude_keywords_still_route():
    mm = ModelManager(config)

    assert mm.map_claude_model_to_openai("claude-opus-4-5") == config.big_model
    assert mm.map_claude_model_to_openai("claude-sonnet-4-5") == config.middle_model
    assert mm.map_claude_model_to_openai("claude-haiku-4-5") == config.small_model


def test_native_passthrough_still_wins():
    mm = ModelManager(config)

    # gpt-* and friends must continue to pass through verbatim regardless
    # of any keyword overlap; aliases only kick in for non-native ids.
    assert mm.map_claude_model_to_openai("gpt-4o") == "gpt-4o"
    assert mm.map_claude_model_to_openai("deepseek-chat") == "deepseek-chat"


def test_unknown_model_falls_back_to_big():
    mm = ModelManager(config)

    assert mm.map_claude_model_to_openai("totally-unrecognized-model") == config.big_model
