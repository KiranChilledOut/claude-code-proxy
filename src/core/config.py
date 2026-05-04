import os
import sys


# Configuration
class Config:
    def __init__(self):
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        # Add Anthropic API key for client validation
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.anthropic_api_key:
            print("Warning: ANTHROPIC_API_KEY not set. Client API key validation will be disabled.")
        self.ignore_client_api_key = os.environ.get("IGNORE_CLIENT_API_KEY", "true").lower() in (
            "1",
            "true",
            "yes",
        )

        self.openai_base_url = os.environ.get(
            "OPENAI_BASE_URL", "https://api.tokenfactory.nebius.com/v1"
        )
        self.azure_api_version = os.environ.get("AZURE_API_VERSION")  # For Azure OpenAI
        self.host = os.environ.get("HOST", "0.0.0.0")
        self.port = int(os.environ.get("PORT", "8083"))
        self.log_level = os.environ.get("LOG_LEVEL", "INFO")
        self.max_tokens_limit = int(os.environ.get("MAX_TOKENS_LIMIT", "4096"))
        self.min_tokens_limit = int(os.environ.get("MIN_TOKENS_LIMIT", "100"))
        # Optional explicit model context limits (tokens). If not set, code falls back to baked-in defaults.
        self.big_model_context_limit = int(os.environ.get("BIG_MODEL_CONTEXT_LIMIT", "0") or 0)
        self.middle_model_context_limit = int(
            os.environ.get("MIDDLE_MODEL_CONTEXT_LIMIT", "0") or 0
        )
        self.small_model_context_limit = int(os.environ.get("SMALL_MODEL_CONTEXT_LIMIT", "0") or 0)
        self.vision_model_context_limit = int(
            os.environ.get("VISION_MODEL_CONTEXT_LIMIT", "0") or 0
        )

        # Connection settings
        self.request_timeout = int(os.environ.get("REQUEST_TIMEOUT", "90"))
        self.max_retries = int(os.environ.get("MAX_RETRIES", "2"))

        # Observability settings. The dashboard stores metadata only by default:
        # model routing, usage, cost estimates, latency, failures, and tool names.
        self.observability_enabled = os.environ.get("OBSERVABILITY_ENABLED", "true").lower() in (
            "1",
            "true",
            "yes",
        )
        self.observability_db_path = os.environ.get(
            "OBSERVABILITY_DB_PATH", "observability.sqlite3"
        )

        # /v1/models surfaces the upstream Token Factory catalog via a cached
        # call. Tune this to taste; default 10 minutes keeps the picker fresh
        # without hammering Token Factory on every Claude Code session start.
        self.models_cache_ttl_seconds = int(os.environ.get("MODELS_CACHE_TTL_SECONDS", "600"))
        self.observability_queue_size = int(os.environ.get("OBSERVABILITY_QUEUE_SIZE", "1000"))
        self.observability_store_tool_args = os.environ.get(
            "OBSERVABILITY_STORE_TOOL_ARGS", "false"
        ).lower() in ("1", "true", "yes")
        self.model_prices_json = os.environ.get("MODEL_PRICES_JSON", "{}")

        # Model settings - BIG and SMALL models
        self.big_model = os.environ.get("BIG_MODEL", "zai-org/GLM-4.5")
        self.middle_model = os.environ.get("MIDDLE_MODEL", self.big_model)
        self.small_model = os.environ.get("SMALL_MODEL", "zai-org/GLM-4.5")
        self.vision_model = os.environ.get("VISION_MODEL", "Qwen/Qwen2.5-VL-72B-Instruct")

        # Aliases for the in-Claude-Code `/model <name>` picker. These let
        # users type a short name (e.g. `/model glm`) and have the proxy route
        # to the right upstream without editing .env. Each alias can be
        # overridden via its env var. Defaults track what's actually live on
        # Nebius's Token Factory today.
        self.glm_model = os.environ.get("GLM_MODEL", "zai-org/GLM-5")
        self.kimi_model = os.environ.get("KIMI_MODEL", "moonshotai/Kimi-K2.5")
        self.gemma_model = os.environ.get("GEMMA_MODEL", "google/gemma-3-27b-it")
        self.qwen_model = os.environ.get("QWEN_MODEL", "Qwen/Qwen3.5-397B-A17B")
        self.nemotron_model = os.environ.get(
            "NEMOTRON_MODEL", "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1"
        )
        self.nemotron_super_model = os.environ.get(
            "NEMOTRON_SUPER_MODEL", "nvidia/nemotron-3-super-120b-a12b"
        )
        self.nemotron_nano_model = os.environ.get(
            "NEMOTRON_NANO_MODEL", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
        )
        self.minimax_model = os.environ.get("MINIMAX_MODEL", "MiniMaxAI/MiniMax-M2.5")
        self.hermes_model = os.environ.get("HERMES_MODEL", "NousResearch/Hermes-4-405B")
        self.gpt_model = os.environ.get("GPT_MODEL", "openai/gpt-oss-120b")
        self.llama_model = os.environ.get(
            "LLAMA_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct"
        )
        self.prime_model = os.environ.get("PRIME_MODEL", "PrimeIntellect/INTELLECT-3")
        self.deepseek_model = os.environ.get("DEEPSEEK_MODEL", "deepseek-ai/DeepSeek-V3.2")

        self.disable_tools = os.environ.get("DISABLE_TOOLS", "false").lower() in (
            "1",
            "true",
            "yes",
        )
        self.strip_image_context = os.environ.get("STRIP_IMAGE_CONTEXT", "true").lower() in (
            "1",
            "true",
            "yes",
        )

        # Ensure bounds are sane even with misconfigured env values.
        if self.max_tokens_limit < 1:
            self.max_tokens_limit = 1
        if self.min_tokens_limit < 1:
            self.min_tokens_limit = 1
        if self.min_tokens_limit > self.max_tokens_limit:
            self.min_tokens_limit = self.max_tokens_limit

    def validate_api_key(self):
        """Basic API key validation"""
        if not self.openai_api_key:
            return False
        # Enforce OpenAI key shape only for official OpenAI endpoint.
        base_url = (self.openai_base_url or "").lower()
        if "api.openai.com" in base_url and not self.openai_api_key.startswith("sk-"):
            return False
        return True

    def validate_client_api_key(self, client_api_key):
        """Validate client's Anthropic API key"""
        # Default behavior: ignore any client-provided API key and rely on server-side OPENAI_API_KEY
        if self.ignore_client_api_key:
            return True

        # If no ANTHROPIC_API_KEY is set in environment, skip validation
        if not self.anthropic_api_key:
            return True

        # Check if the client's API key matches the expected value
        return client_api_key == self.anthropic_api_key

    def get_custom_headers(self):
        """Get custom headers from environment variables"""
        custom_headers = {}

        # Get all environment variables
        env_vars = dict(os.environ)

        # Find CUSTOM_HEADER_* environment variables
        for env_key, env_value in env_vars.items():
            if env_key.startswith("CUSTOM_HEADER_"):
                # Convert CUSTOM_HEADER_KEY to Header-Key
                # Remove 'CUSTOM_HEADER_' prefix and convert to header format
                header_name = env_key[14:]  # Remove 'CUSTOM_HEADER_' prefix

                if header_name:  # Make sure it's not empty
                    # Convert underscores to hyphens for HTTP header format
                    header_name = header_name.replace("_", "-")
                    custom_headers[header_name] = env_value

        return custom_headers


try:
    config = Config()
    print(f"Configuration loaded: API_KEY={'*' * 20}..., BASE_URL='{config.openai_base_url}'")
except Exception as e:
    print(f"Configuration Error: {e}")
    sys.exit(1)
