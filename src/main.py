import os
import signal
import subprocess
import sys
import time
from pathlib import Path

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


# --- Daemon helpers (start/stop/status) -------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
PID_FILE = REPO_ROOT / ".proxy.pid"
LOG_FILE = REPO_ROOT / "proxy.log"


def _read_pid():
    if not PID_FILE.exists():
        return None
    try:
        return int(PID_FILE.read_text().strip())
    except (ValueError, OSError):
        return None


def _process_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def daemon_start():
    pid = _read_pid()
    if pid is not None and _process_alive(pid):
        print(f"proxy already running (pid {pid})")
        return 0

    log = open(LOG_FILE, "a")
    proc = subprocess.Popen(
        [sys.executable, str(REPO_ROOT / "start_proxy.py")],
        stdout=log,
        stderr=subprocess.STDOUT,
        cwd=str(REPO_ROOT),
        start_new_session=True,
    )
    PID_FILE.write_text(f"{proc.pid}\n")
    print(f"proxy started (pid {proc.pid}); logs: {LOG_FILE}")
    return 0


def daemon_stop():
    pid = _read_pid()
    if pid is None:
        print("proxy not running (no PID file)")
        return 0
    if not _process_alive(pid):
        print(f"proxy not running (stale PID file for pid {pid}); cleaning up")
        PID_FILE.unlink(missing_ok=True)
        return 0
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        PID_FILE.unlink(missing_ok=True)
        return 0
    for _ in range(20):
        if not _process_alive(pid):
            break
        time.sleep(0.25)
    else:
        print("proxy did not exit on SIGTERM, sending SIGKILL")
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    PID_FILE.unlink(missing_ok=True)
    print(f"proxy stopped (pid {pid})")
    return 0


def daemon_status():
    pid = _read_pid()
    if pid is None:
        print("proxy not running")
        return 1
    if _process_alive(pid):
        print(f"proxy running (pid {pid})")
        return 0
    print(f"proxy not running (stale PID file for pid {pid})")
    PID_FILE.unlink(missing_ok=True)
    return 1


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("start", "stop", "status"):
        cmd = sys.argv[1]
        if cmd == "start":
            sys.exit(daemon_start())
        if cmd == "stop":
            sys.exit(daemon_stop())
        if cmd == "status":
            sys.exit(daemon_status())

    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Claude-to-OpenAI API Proxy v1.0.0")
        print("")
        print("Usage: python start_proxy.py [start|stop|status|--help]")
        print("")
        print("  (no args)    Run the proxy in the foreground (Ctrl-C to stop).")
        print("  start        Start the proxy as a detached background process.")
        print("               PID is recorded in .proxy.pid; logs go to proxy.log.")
        print("  stop         Stop a background-running proxy via .proxy.pid.")
        print("  status       Print whether a background proxy is currently running.")
        print("Usage: python start_proxy.py [--help|--selftest]")
        print("")
        print("  --selftest   Hit /test-connection in-process and exit 0 if it")
        print("               succeeds, non-zero otherwise. Intended for CI and")
        print("               install scripts.")
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
        print(
            "  ENABLE_REQUEST_OPTIMIZATIONS - Answer Claude Code housekeeping "
            f"requests locally (default: {config.enable_request_optimizations})"
        )
        print("")
        print("Model mapping:")
        print(f"  Claude haiku models -> {config.small_model}")
        print(f"  Claude sonnet models -> {config.middle_model}")
        print(f"  Claude opus models -> {config.big_model}")
        print(f"  Requests with images -> {config.vision_model}")
        sys.exit(0)

    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        # In-process smoke test: invoke /test-connection via Starlette's
        # TestClient (no uvicorn, no port binding) and exit 0/1 based on
        # the result. We deliberately don't enter TestClient as a context
        # manager so FastAPI lifespan handlers (which would open the
        # observability sqlite database) do not fire.
        import json

        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/test-connection")
        try:
            result = response.json()
        except ValueError:
            status_code = response.status_code
            message = f"selftest: non-JSON response (status {status_code}): {response.text}"
            print(message, file=sys.stderr)
            sys.exit(2)
        json.dump(result, sys.stdout, indent=2)
        sys.stdout.write("\n")
        sys.exit(0 if result.get("status") == "success" else 1)

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
    print(
        "   Request Optimizations: "
        f"{'Enabled' if config.enable_request_optimizations else 'Disabled'}"
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
