from src.core.config import config
from src.core.model_manager import ModelManager


def test_alias_exact_match_routes_to_provider_model(monkeypatch):
    monkeypatch.setattr(config, "glm_model", "zai-org/GLM-5")
    monkeypatch.setattr(config, "kimi_model", "moonshotai/Kimi-K2.5")
    monkeypatch.setattr(config, "gemma_model", "google/gemma-3-27b-it")
    mm = ModelManager(config)

    assert mm.map_claude_model_to_openai("glm") == "zai-org/GLM-5"
    assert mm.map_claude_model_to_openai("kimi") == "moonshotai/Kimi-K2.5"
    assert mm.map_claude_model_to_openai("gemma") == "google/gemma-3-27b-it"


def test_alias_keyword_in_longer_id(monkeypatch):
    monkeypatch.setattr(config, "glm_model", "zai-org/GLM-5")
    monkeypatch.setattr(config, "kimi_model", "moonshotai/Kimi-K2.5")
    monkeypatch.setattr(config, "gemma_model", "google/gemma-3-27b-it")
    mm = ModelManager(config)

    assert mm.map_claude_model_to_openai("glm-5") == "zai-org/GLM-5"
    assert mm.map_claude_model_to_openai("Glm-Beta") == "zai-org/GLM-5"
    assert mm.map_claude_model_to_openai("kimi-2.5") == "moonshotai/Kimi-K2.5"


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


def test_slash_id_passes_through_verbatim():
    mm = ModelManager(config)

    # Token Factory ids contain "/"; these must pass through unchanged so
    # users can pick any catalog entry directly without falling back to BIG_MODEL.
    for id_ in (
        "meta-llama/Llama-3.3-70B-Instruct",
        "Qwen/Qwen3-32B",
        "openai/gpt-oss-120b",
        "MiniMaxAI/MiniMax-M2.5",
    ):
        assert mm.map_claude_model_to_openai(id_) == id_


def test_new_aliases_route_to_their_defaults(monkeypatch):
    monkeypatch.setattr(config, "qwen_model", "Qwen/Qwen3.5-397B-A17B")
    monkeypatch.setattr(config, "nemotron_model", "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1")
    monkeypatch.setattr(config, "nemotron_super_model", "nvidia/nemotron-3-super-120b-a12b")
    monkeypatch.setattr(config, "nemotron_nano_model", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B")
    monkeypatch.setattr(config, "minimax_model", "MiniMaxAI/MiniMax-M2.5")
    monkeypatch.setattr(config, "hermes_model", "NousResearch/Hermes-4-405B")
    monkeypatch.setattr(config, "gpt_model", "openai/gpt-oss-120b")
    monkeypatch.setattr(config, "llama_model", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    monkeypatch.setattr(config, "prime_model", "PrimeIntellect/INTELLECT-3")
    monkeypatch.setattr(config, "deepseek_model", "deepseek-ai/DeepSeek-V3.2")
    mm = ModelManager(config)

    assert mm.map_claude_model_to_openai("qwen") == "Qwen/Qwen3.5-397B-A17B"
    assert mm.map_claude_model_to_openai("nemotron") == "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1"
    assert mm.map_claude_model_to_openai("super") == "nvidia/nemotron-3-super-120b-a12b"
    assert mm.map_claude_model_to_openai("nano") == "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
    assert mm.map_claude_model_to_openai("minimax") == "MiniMaxAI/MiniMax-M2.5"
    assert mm.map_claude_model_to_openai("hermes") == "NousResearch/Hermes-4-405B"
    assert mm.map_claude_model_to_openai("gpt") == "openai/gpt-oss-120b"
    assert mm.map_claude_model_to_openai("llama") == "meta-llama/Meta-Llama-3.1-8B-Instruct"
    assert mm.map_claude_model_to_openai("prime") == "PrimeIntellect/INTELLECT-3"
    assert mm.map_claude_model_to_openai("deepseek") == "deepseek-ai/DeepSeek-V3.2"
