import sys

import uvicorn
from fastapi import FastAPI

from src.api.endpoints import router as api_router
from src.core.config import config
from src.observability.routes import router as observability_router
from src.observability.store import observability_recorder

app = FastAPI(title="Claude-to-OpenAI API Proxy", version="1.0.0")

app.include_router(api_router)
app.include_router(observability_router)


@app.on_event("startup")
async def startup_event():
    await observability_recorder.start()


@app.on_event("shutdown")
async def shutdown_event():
    await observability_recorder.stop()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Claude-to-OpenAI API Proxy v1.0.0")
        print("")
        print("Usage: python start_proxy.py")
        print("")
        print("Required environment variables:")
        print("  OPENAI_API_KEY - Your provider API key")
        print("")
        print("Optional environment variables:")
        print("  ANTHROPIC_API_KEY - Expected Anthropic API key for client validation")
        print("                      If set, clients must provide this exact API key")
        print("  IGNORE_CLIENT_API_KEY - Ignore/drop client API key headers (default: true)")
        print(
            f"  OPENAI_BASE_URL - OpenAI-compatible API base URL (default: {config.openai_base_url})"
        )
        print(f"  BIG_MODEL - Model for opus requests (default: {config.big_model})")
        print(f"  MIDDLE_MODEL - Model for sonnet requests (default: {config.middle_model})")
        print(f"  SMALL_MODEL - Model for haiku requests (default: {config.small_model})")
        print(f"  VISION_MODEL - Model for image requests (default: {config.vision_model})")
        print(f"  HOST - Server host (default: {config.host})")
        print(f"  PORT - Server port (default: {config.port})")
        print(f"  LOG_LEVEL - Logging level (default: {config.log_level})")
        print(f"  MAX_TOKENS_LIMIT - Token limit (default: {config.max_tokens_limit})")
        print(
            f"  MIN_TOKENS_LIMIT - Fallback token limit for invalid requests (default: {config.min_tokens_limit})"
        )
        print(f"  REQUEST_TIMEOUT - Request timeout in seconds (default: {config.request_timeout})")
        print(
            f"  MAX_RETRIES - Retry attempts for provider requests (default: {config.max_retries})"
        )
        print("")
        print("Model mapping:")
        print(f"  Claude haiku models -> {config.small_model}")
        print(f"  Claude sonnet models -> {config.middle_model}")
        print(f"  Claude opus models -> {config.big_model}")
        print(f"  Requests with images -> {config.vision_model}")
        sys.exit(0)

    # Configuration summary
    print("🚀 Claude-to-OpenAI API Proxy v1.0.0")
    print(f"✅ Configuration loaded successfully")
    print(f"   OpenAI Base URL: {config.openai_base_url}")
    print(f"   Big Model (opus): {config.big_model}")
    print(f"   Middle Model (sonnet): {config.middle_model}")
    print(f"   Small Model (haiku): {config.small_model}")
    print(f"   Vision Model (images): {config.vision_model}")
    print(f"   Max Tokens Limit: {config.max_tokens_limit}")
    print(f"   Request Timeout: {config.request_timeout}s")
    print(f"   Server: {config.host}:{config.port}")
    validation_enabled = bool(config.anthropic_api_key and not config.ignore_client_api_key)
    print(f"   Client API Key Validation: {'Enabled' if validation_enabled else 'Disabled'}")
    print(
        f"   Ignore Client API Key Headers: {'Enabled' if config.ignore_client_api_key else 'Disabled'}"
    )
    print(
        f"   Observability: {'Enabled' if config.observability_enabled else 'Disabled'} "
        f"({config.observability_db_path})"
    )
    print("")

    # Parse log level - extract just the first word to handle comments
    log_level = config.log_level.split()[0].lower()

    # Validate and set default if invalid
    valid_levels = ["debug", "info", "warning", "error", "critical"]
    if log_level not in valid_levels:
        log_level = "info"

    # Start server
    uvicorn.run(
        "src.main:app",
        host=config.host,
        port=config.port,
        log_level=log_level,
        reload=False,
    )


if __name__ == "__main__":
    main()
