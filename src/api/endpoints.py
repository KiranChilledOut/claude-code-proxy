import json
import os
import time
import traceback
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from src.codex.models import ResponsesItem, ResponsesRequest, ResponsesResponse, ResponsesUsage
from src.codex.request_converter import convert_responses_to_openai_chat
from src.codex.response_converter import convert_openai_to_responses
from src.codex.session import SessionStore
from src.codex.stream_converter import codex_response_to_sse, convert_openai_sse_to_responses_sse
from src.codex.tools_compat import parse_codex_tools
from src.api.optimization_handlers import optimized_response_to_sse, try_local_optimization
from src.conversion.request_converter import (
    _count_tokens_text,
    _estimate_prompt_tokens,
    convert_claude_to_openai,
    count_claude_request_tokens,
)
from src.conversion import server_tools
from src.conversion.response_converter import (
    claude_response_to_sse,
    convert_openai_streaming_to_claude_with_cancellation,
    convert_openai_to_claude_response,
)
from src.core.client import OpenAIClient
from src.core.config import config
from src.core.logging import logger
from src.core.model_manager import model_manager
from src.models.claude import ClaudeMessagesRequest, ClaudeTokenCountRequest
from src.observability.store import observability_recorder

router = APIRouter()

# Get custom headers from config
custom_headers = config.get_custom_headers()

openai_client = OpenAIClient(
    config.openai_api_key,
    config.openai_base_url,
    config.request_timeout,
    api_version=config.azure_api_version,
    custom_headers=custom_headers,
    max_retries=config.max_retries,
)

codex_session_store = SessionStore(config.codex_session_ttl_seconds)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_tool_calls_from_codex_output(output_items: list) -> list:
    """Extract tool-call metadata from Codex output items for observability."""
    tool_calls = []
    for item in output_items:
        if isinstance(item, dict):
            if item.get("type") == "function_call":
                tool_calls.append({
                    "tool_id": item.get("call_id", ""),
                    "tool_name": item.get("name", ""),
                    "arguments": item.get("arguments", ""),
                    "status": item.get("status", "emitted"),
                    "sanitized": False,
                })
        elif hasattr(item, "type") and item.type == "function_call":
            tool_calls.append({
                "tool_id": getattr(item, "call_id", "") or "",
                "tool_name": getattr(item, "name", "") or "",
                "arguments": getattr(item, "arguments", "") or "",
                "status": getattr(item, "status", "emitted") or "emitted",
                "sanitized": False,
            })
    return tool_calls


def _extract_tool_calls_from_claude_response(claude_response: dict) -> list:
    tool_calls = []
    for block in claude_response.get("content", []) or []:
        if block.get("type") != "tool_use":
            continue
        tool_calls.append(
            {
                "tool_id": block.get("id"),
                "tool_name": block.get("name"),
                "arguments": block.get("input"),
                "status": "emitted",
                "sanitized": False,
            }
        )
    return tool_calls


def _has_token_usage(usage: Optional[dict]) -> bool:
    if not usage:
        return False
    return any(
        int(usage.get(key) or 0) > 0
        for key in (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        )
    )


def _stream_usage_with_fallback(stream_metrics: dict, estimated_input_tokens: int) -> dict:
    usage = dict(stream_metrics.get("usage") or {})
    if _has_token_usage(usage):
        usage.setdefault("source", "provider")
        return usage

    return {
        "input_tokens": estimated_input_tokens,
        "output_tokens": int(stream_metrics.get("estimated_output_tokens") or 0),
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "source": "estimated",
    }


def _record_message_observability(
    *,
    request_id: str,
    session_id: Optional[str] = None,
    session_name: Optional[str] = None,
    started_at: str,
    started_at_unix: float,
    start_monotonic: float,
    request: ClaudeMessagesRequest,
    backend_model: Optional[str],
    stream: bool,
    status: str,
    http_status: Optional[int],
    usage: Optional[dict] = None,
    stop_reason: Optional[str] = None,
    error_type: Optional[str] = None,
    error_message: Optional[str] = None,
    tool_calls: Optional[list] = None,
) -> None:
    observability_recorder.record_request(
        request_id=request_id,
        session_id=session_id,
        session_name=session_name,
        started_at=started_at,
        started_at_unix=started_at_unix,
        completed_at=_utc_now_iso(),
        base_url=config.openai_base_url,
        claude_model=request.model,
        backend_model=backend_model,
        stream=stream,
        status=status,
        http_status=http_status,
        latency_ms=(time.monotonic() - start_monotonic) * 1000,
        usage=usage,
        stop_reason=stop_reason,
        error_type=error_type,
        error_message=error_message,
        tool_calls=tool_calls,
    )


async def validate_api_key(
    x_api_key: Optional[str] = Header(None), authorization: Optional[str] = Header(None)
):
    """Validate the client's API key from either x-api-key header or Authorization header."""
    # Default behavior for this proxy: drop/ignore any client-supplied API key.
    # The proxy always uses server-side OPENAI_API_KEY for upstream calls.
    if config.ignore_client_api_key:
        if x_api_key or authorization:
            logger.debug("Client API key header received and ignored by proxy policy")
        return

    client_api_key = None

    # Extract API key from headers
    if x_api_key:
        client_api_key = x_api_key
    elif authorization and authorization.startswith("Bearer "):
        client_api_key = authorization.replace("Bearer ", "")

    # Skip validation if ANTHROPIC_API_KEY is not set in the environment
    if not config.anthropic_api_key:
        return

    # Validate the client API key
    if not client_api_key or not config.validate_client_api_key(client_api_key):
        logger.warning(f"Invalid API key provided by client")
        raise HTTPException(
            status_code=401, detail="Invalid API key. Please provide a valid Anthropic API key."
        )


# ===========================================================================
# Codex /v1/responses endpoints
# ===========================================================================

@router.get("/v1/responses")
async def codex_responses_get(http_request: Request):
    """Codex CLI first tries WebSocket; return 426 to force HTTP fallback."""
    if config.codex_websocket_fallback and http_request.headers.get("upgrade", "").lower() == "websocket":
        return Response(status_code=426)
    return Response(status_code=405)


@router.post("/v1/responses")
async def create_response(
    request: ResponsesRequest, http_request: Request, _: None = Depends(validate_api_key)
):
    """Codex Responses API endpoint."""
    request_id = str(uuid.uuid4())
    started_at = _utc_now_iso()
    started_at_unix = time.time()
    start_monotonic = time.monotonic()
    backend_model = None

    # Session ID from request or headers
    session_id = None
    if request.previous_response_id:
        session_id = request.previous_response_id

    try:
        # DEBUG: log full request payload
        logger.info(f"Codex request input: {request.input!r}, model={request.model}, stream={request.stream}")
        # Look up session if previous_response_id provided
        session = None
        session_items: list = []
        if request.previous_response_id:
            session = await codex_session_store.get(request.previous_response_id)
            if session:
                session_items = session.input_items + session.output_items

        # Parse tools
        tool_ctx = None
        if request.tools and config.codex_tool_compat:
            tool_ctx = parse_codex_tools(request.tools)

        # Convert request
        openai_request = convert_responses_to_openai_chat(
            request,
            session_items=session_items,
            tool_ctx=tool_ctx,
            model_manager=model_manager,
        )
        backend_model = openai_request.get("model")

        # Server-side web search: mirror the Claude Code path for Codex.
        # When a search tool is offered and Tavily is configured, the proxy
        # executes the search itself in a bounded loop so results are available
        # to the model (Codex's built-in search can't run against a non-Anthropic
        # backend). Only engaged when a search tool is present.
        needs_search = (
            tool_ctx is not None
            and getattr(tool_ctx, "has_search_tool", False)
            and config.tavily_api_key
            and config.server_search_enabled
        )

        if needs_search:
            # The search loop is non-streaming by design (multiple chat.completions
            # calls are run internally).
            openai_request["stream"] = False
            openai_request.pop("stream_options", None)
            openai_response = await server_tools.run_search_loop(
                openai_request, openai_client, request_id
            )
            codex_response = convert_openai_to_responses(
                openai_response,
                request_model=backend_model or request.model,
                previous_id=session_id,
                tool_ctx=tool_ctx,
            )

            # Observability
            usage_dict = {}
            if hasattr(codex_response.usage, "model_dump"):
                usage_dict = codex_response.usage.model_dump()
            else:
                usage_dict = {
                    "input_tokens": codex_response.usage.input_tokens,
                    "output_tokens": codex_response.usage.output_tokens,
                    "total_tokens": codex_response.usage.total_tokens,
                }
            observability_recorder.record_request(
                request_id=request_id,
                session_id=session_id,
                session_name=None,
                started_at=started_at,
                started_at_unix=started_at_unix,
                completed_at=_utc_now_iso(),
                base_url=config.openai_base_url,
                claude_model=request.model,
                backend_model=backend_model,
                stream=request.stream,
                status="success",
                http_status=200,
                latency_ms=(time.monotonic() - start_monotonic) * 1000,
                usage=usage_dict,
                tool_calls=_extract_tool_calls_from_codex_output(codex_response.output),
            )

            # Save session (same logic as non-streaming path)
            if isinstance(request.input, list):
                input_items = list(request.input)
            else:
                input_items = [ResponsesItem(type="text", text=request.input)]
            await codex_session_store.put(
                codex_response.id,
                input_items,
                codex_response.output,
                previous_id=session_id,
            )

            if not request.stream:
                return codex_response

            # Streaming: synthesise SSE from the complete response
            async def search_sse_stream():
                async for event in codex_response_to_sse(
                    codex_response, request_model=backend_model or request.model
                ):
                    yield event

            return StreamingResponse(
                search_sse_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        # Non-streaming
        if not request.stream:
            openai_response = await openai_client.create_chat_completion(openai_request, request_id)
            codex_response = convert_openai_to_responses(
                openai_response,
                request_model=backend_model or request.model,
                previous_id=session_id,
                tool_ctx=tool_ctx,
            )

            # Observability
            usage_dict = {}
            if hasattr(codex_response.usage, "model_dump"):
                usage_dict = codex_response.usage.model_dump()
            else:
                usage_dict = {
                    "input_tokens": codex_response.usage.input_tokens,
                    "output_tokens": codex_response.usage.output_tokens,
                    "total_tokens": codex_response.usage.total_tokens,
                }
            observability_recorder.record_request(
                request_id=request_id,
                session_id=session_id,
                session_name=None,
                started_at=started_at,
                started_at_unix=started_at_unix,
                completed_at=_utc_now_iso(),
                base_url=config.openai_base_url,
                claude_model=request.model,
                backend_model=backend_model,
                stream=False,
                status="success",
                http_status=200,
                latency_ms=(time.monotonic() - start_monotonic) * 1000,
                usage=usage_dict,
                tool_calls=_extract_tool_calls_from_codex_output(codex_response.output),
            )

            # Save session (non-streaming only)
            if isinstance(request.input, list):
                input_items = list(request.input)
            else:
                input_items = [ResponsesItem(type="text", text=request.input)]
            await codex_session_store.put(
                codex_response.id,
                input_items,
                codex_response.output,
                previous_id=session_id,
            )

            return codex_response

        # Streaming
        openai_stream = openai_client.create_chat_completion_stream(openai_request, request_id)

        async def codex_sse_stream():
            stream_status = "success"
            stream_error = None
            accumulator: Dict[str, Any] = {}
            try:
                async for event in convert_openai_sse_to_responses_sse(
                    openai_stream,
                    request_model=backend_model or request.model,
                    tool_ctx=tool_ctx,
                    accumulator=accumulator,
                ):
                    yield event
            except Exception as exc:
                stream_status = "error"
                stream_error = str(exc)
                raise
            finally:
                observability_recorder.record_request(
                    request_id=request_id,
                    session_id=session_id,
                    session_name=None,
                    started_at=started_at,
                    started_at_unix=started_at_unix,
                    completed_at=_utc_now_iso(),
                    base_url=config.openai_base_url,
                    claude_model=request.model,
                    backend_model=backend_model,
                    stream=True,
                    status=stream_status,
                    http_status=200 if stream_status == "success" else 500,
                    latency_ms=(time.monotonic() - start_monotonic) * 1000,
                    error_type=None if stream_status == "success" else type(stream_error).__name__,
                    error_message=stream_error,
                )
                # Save session for next turn (same logic as non-streaming)
                if accumulator.get("response_id"):
                    if isinstance(request.input, list):
                        input_items = list(request.input)
                    else:
                        input_items = [ResponsesItem(type="text", text=request.input)]
                    output_items: List[Any] = []
                    if accumulator.get("text_buf", ""):
                        output_items.append(
                            ResponsesItem(
                                type="message",
                                role="assistant",
                                content=accumulator["text_buf"],
                                status="completed",
                            )
                        )
                    for tc in accumulator.get("tool_calls", []):
                        output_items.append(
                            ResponsesItem(
                                type="function_call",
                                call_id=tc.get("id", ""),
                                name=tc.get("name", ""),
                                arguments=tc.get("arguments", ""),
                                status="completed",
                            )
                        )
                    await codex_session_store.put(
                        accumulator["response_id"],
                        input_items,
                        output_items,
                        previous_id=session_id,
                    )

        return StreamingResponse(
            codex_sse_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Codex response error: {e}")
        logger.error(traceback.format_exc())
        error_response = ResponsesResponse(
            id=str(uuid.uuid4()),
            model=request.model,
            output=[ResponsesItem(type="message", role="assistant", content=f"Error: {e}")],
            status="failed",
            usage=ResponsesUsage(input_tokens=0, output_tokens=0, total_tokens=0),
        )
        observability_recorder.record_request(
            request_id=request_id,
            session_id=session_id,
            session_name=None,
            started_at=started_at,
            started_at_unix=started_at_unix,
            completed_at=_utc_now_iso(),
            base_url=config.openai_base_url,
            claude_model=request.model,
            backend_model=backend_model,
            stream=False,
            status="error",
            http_status=500,
            latency_ms=(time.monotonic() - start_monotonic) * 1000,
            error_type=type(e).__name__,
            error_message=str(e),
        )
        return error_response


@router.post("/v1/messages")
async def create_message(
    request: ClaudeMessagesRequest, http_request: Request, _: None = Depends(validate_api_key)
):
    request_id = str(uuid.uuid4())
    started_at = _utc_now_iso()
    started_at_unix = time.time()
    start_monotonic = time.monotonic()
    backend_model = None
    try:
        # Log anthropic-beta header if present (for computer use, etc.)
        beta_header = http_request.headers.get("anthropic-beta", "")
        if beta_header:
            logger.info(f"anthropic-beta header: {beta_header}")

        session_id = http_request.headers.get("x-claude-code-session-id")
        session_name = http_request.headers.get("x-session-name")
        if session_id:
            logger.debug(f"x-claude-code-session-id: {session_id}")

        logger.debug(f"Incoming headers: dict({dict(http_request.headers)})")

        # Check if client disconnected before doing either local work or upstream calls.
        if await http_request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        optimized = try_local_optimization(request)
        if optimized is not None:
            backend_model = f"local/{optimized.kind}"
            observability_usage = dict(optimized.response.get("usage") or {})
            observability_usage["source"] = "local_optimization"
            _record_message_observability(
                request_id=request_id,
                session_id=session_id,
                session_name=session_name,
                started_at=started_at,
                started_at_unix=started_at_unix,
                start_monotonic=start_monotonic,
                request=request,
                backend_model=backend_model,
                stream=bool(request.stream),
                status="success",
                http_status=200,
                usage=observability_usage,
                stop_reason=optimized.response.get("stop_reason"),
                tool_calls=[],
            )
            if request.stream:
                return StreamingResponse(
                    optimized_response_to_sse(optimized.response),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Headers": "*",
                    },
                )
            return optimized.response

        # Convert Claude request to OpenAI format
        openai_request = convert_claude_to_openai(request, model_manager)
        backend_model = openai_request.get("model")
        estimated_input_tokens = _estimate_prompt_tokens(
            openai_request.get("messages", []), include_safety_buffer=False
        )

        # Server-side web search: when a search tool is offered and Tavily is
        # configured, the proxy executes the search itself in a bounded loop and
        # returns the final answer (Claude Code's search can't run behind a
        # non-Anthropic backend). Only engaged when a search tool is present, so
        # all other requests stay on the normal streaming path untouched.
        if server_tools.request_has_search_tool(request):
            openai_response = await server_tools.run_search_loop(
                openai_request, openai_client, request_id
            )
            claude_response = convert_openai_to_claude_response(openai_response, request)
            _record_message_observability(
                request_id=request_id,
                session_id=session_id,
                session_name=session_name,
                started_at=started_at,
                started_at_unix=started_at_unix,
                start_monotonic=start_monotonic,
                request=request,
                backend_model=backend_model,
                stream=bool(request.stream),
                status="success",
                http_status=200,
                usage=claude_response.get("usage"),
                stop_reason=claude_response.get("stop_reason"),
                tool_calls=_extract_tool_calls_from_claude_response(claude_response),
            )
            if request.stream:
                return StreamingResponse(
                    claude_response_to_sse(claude_response),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Headers": "*",
                    },
                )
            return claude_response

        if request.stream:
            # Streaming response - wrap in error handling
            try:
                openai_stream = openai_client.create_chat_completion_stream(
                    openai_request, request_id
                )
                stream_metrics = {
                    "usage": {},
                    "tool_calls": [],
                    "stop_reason": None,
                    "status": "success",
                }

                async def observed_stream():
                    stream_status = "success"
                    stream_error = None
                    try:
                        async for event in convert_openai_streaming_to_claude_with_cancellation(
                            openai_stream,
                            request,
                            logger,
                            http_request,
                            openai_client,
                            request_id,
                            observability_context=stream_metrics,
                        ):
                            yield event
                        stream_status = stream_metrics.get("status") or "success"
                        stream_error = stream_metrics.get("error_message")
                    except Exception as exc:
                        stream_status = "error"
                        stream_error = str(exc)
                        raise
                    finally:
                        _record_message_observability(
                            request_id=request_id,
                            session_id=session_id,
                            session_name=session_name,
                            started_at=started_at,
                            started_at_unix=started_at_unix,
                            start_monotonic=start_monotonic,
                            request=request,
                            backend_model=backend_model,
                            stream=True,
                            status=stream_status,
                            http_status=200 if stream_status == "success" else 500,
                            usage=_stream_usage_with_fallback(
                                stream_metrics, estimated_input_tokens
                            ),
                            stop_reason=stream_metrics.get("stop_reason"),
                            error_type=stream_metrics.get("error_type"),
                            error_message=stream_error,
                            tool_calls=stream_metrics.get("tool_calls"),
                        )

                return StreamingResponse(
                    observed_stream(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Headers": "*",
                    },
                )
            except HTTPException as e:
                # Convert to proper error response for streaming
                logger.error(f"Streaming error: {e.detail}")
                import traceback

                logger.error(traceback.format_exc())
                error_message = openai_client.classify_openai_error(e.detail)
                error_response = {
                    "type": "error",
                    "error": {"type": "api_error", "message": error_message},
                }
                _record_message_observability(
                    request_id=request_id,
                    session_id=session_id,
                    session_name=session_name,
                    started_at=started_at,
                    started_at_unix=started_at_unix,
                    start_monotonic=start_monotonic,
                    request=request,
                    backend_model=backend_model,
                    stream=True,
                    status="error",
                    http_status=e.status_code,
                    error_type="HTTPException",
                    error_message=error_message,
                )
                return JSONResponse(status_code=e.status_code, content=error_response)
        else:
            # Non-streaming response
            openai_response = await openai_client.create_chat_completion(openai_request, request_id)
            claude_response = convert_openai_to_claude_response(openai_response, request)
            _record_message_observability(
                request_id=request_id,
                session_id=session_id,
                session_name=session_name,
                started_at=started_at,
                started_at_unix=started_at_unix,
                start_monotonic=start_monotonic,
                request=request,
                backend_model=backend_model,
                stream=False,
                status="success",
                http_status=200,
                usage=claude_response.get("usage"),
                stop_reason=claude_response.get("stop_reason"),
                tool_calls=_extract_tool_calls_from_claude_response(claude_response),
            )
            return claude_response
    except HTTPException as e:
        _record_message_observability(
            request_id=request_id,
            session_id=session_id,
            session_name=session_name,
            started_at=started_at,
            started_at_unix=started_at_unix,
            start_monotonic=start_monotonic,
            request=request,
            backend_model=backend_model,
            stream=bool(request.stream),
            status="cancelled" if e.status_code == 499 else "error",
            http_status=e.status_code,
            error_type="HTTPException",
            error_message=str(e.detail),
        )
        raise
    except Exception as e:
        import traceback

        logger.error(f"Unexpected error processing request: {e}")
        logger.error(traceback.format_exc())
        error_message = openai_client.classify_openai_error(str(e))
        _record_message_observability(
            request_id=request_id,
            session_id=session_id,
            session_name=session_name,
            started_at=started_at,
            started_at_unix=started_at_unix,
            start_monotonic=start_monotonic,
            request=request,
            backend_model=backend_model,
            stream=bool(request.stream),
            status="error",
            http_status=500,
            error_type=type(e).__name__,
            error_message=error_message,
        )
        raise HTTPException(status_code=500, detail=error_message)


@router.post("/v1/messages/count_tokens")
async def count_tokens(request: ClaudeTokenCountRequest, _: None = Depends(validate_api_key)):
    """Anthropic-compatible token-counting endpoint.

    Returns {"input_tokens": N} matching the shape Claude Code expects.
    Counts system + every message (text / image / tool_use / tool_result)
    + every tool definition, including schema-less computer/bash/text_editor
    tools. Tool definitions are the largest part of most Claude Code
    requests — the prior implementation silently omitted them.
    """
    try:
        return {"input_tokens": count_claude_request_tokens(request)}
    except Exception as e:
        logger.error(f"Error counting tokens: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "openai_api_configured": bool(config.openai_api_key),
        "api_key_valid": config.validate_api_key(),
        "client_api_key_validation": bool(
            config.anthropic_api_key and not config.ignore_client_api_key
        ),
        "client_api_key_ignored": bool(config.ignore_client_api_key),
    }


@router.get("/test-connection")
async def test_connection():
    """Test API connectivity to OpenAI"""
    try:
        # Simple test request to verify API connectivity
        test_response = await openai_client.create_chat_completion(
            {
                "model": config.small_model,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5,
            }
        )

        return {
            "status": "success",
            "message": "Successfully connected to OpenAI API",
            "model_used": config.small_model,
            "timestamp": datetime.now().isoformat(),
            "response_id": test_response.get("id", "unknown"),
        }

    except Exception as e:
        logger.error(f"API connectivity test failed: {e}")
        msg = str(e)
        msg_l = msg.lower()

        if "404" in msg or "not found" in msg_l or "does not exist" in msg_l:
            suggestions = [
                f"The configured model '{config.small_model}' may not be available on this provider — "
                f"verify against GET {config.openai_base_url.rstrip('/')}/models",
                "Check BIG_MODEL, MIDDLE_MODEL, SMALL_MODEL, and VISION_MODEL in your .env",
                "Token-factory providers like Nebius rotate model availability",
            ]
        elif "401" in msg or "403" in msg or "unauthorized" in msg_l or "forbidden" in msg_l:
            suggestions = [
                "Check your OPENAI_API_KEY is valid",
                "Verify your API key has the necessary permissions",
            ]
        elif "429" in msg or "rate" in msg_l:
            suggestions = [
                "Check if you have reached rate limits",
                "Wait and retry, or contact your provider about quota",
            ]
        else:
            suggestions = [
                "Check your OPENAI_API_KEY is valid",
                "Verify your API key has the necessary permissions",
                "Check if you have reached rate limits",
            ]

        return JSONResponse(
            status_code=503,
            content={
                "status": "failed",
                "error_type": "API Error",
                "message": msg,
                "timestamp": datetime.now().isoformat(),
                "suggestions": suggestions,
            },
        )


def rotate_log_file(log_file_path: str, max_size_mb: int = 10):
    """Rotate log file if it exceeds max_size_mb"""
    try:
        if os.path.exists(log_file_path):
            file_size = os.path.getsize(log_file_path)
            max_size_bytes = max_size_mb * 1024 * 1024

            if file_size > max_size_bytes:
                # Create backup
                backup_path = f"{log_file_path}.bak"
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                os.rename(log_file_path, backup_path)
                logger.info(f"Rotated log file: {log_file_path} -> {backup_path}")
    except Exception as e:
        logger.error(f"Error rotating log file: {e}")


async def parse_flexible_events(request: Request):
    """
    Parse events from request body in flexible formats:
    - JSON array: [{"event": "data"}, ...]
    - Single object: {"event": "data"}
    - Invalid JSON wrapped in array context
    """
    try:
        # Get raw body and try to parse
        body = await request.body()

        if not body:
            return []

        # Try to parse as JSON
        try:
            data = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON received: {e}")
            # Try to fix common JSON issues and parse again
            text = body.decode("utf-8")

            # Try to fix unquoted property names (common issue)
            import re

            # Replace unquoted property names with quoted ones
            fixed_text = re.sub(r"([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)", r'\1"\2"\3', text)

            try:
                data = json.loads(fixed_text)
                logger.info("Successfully parsed JSON after fixing unquoted properties")
            except json.JSONDecodeError:
                logger.error("Could not parse JSON even after attempted fixes")
                return []

        # Handle different input formats
        if isinstance(data, list):
            # Already an array - use as-is
            return data
        elif isinstance(data, dict):
            # Single object - wrap in array
            return [data]
        else:
            # Other types (string, number, etc.) - wrap in array as event
            return [{"raw_data": data}]

    except Exception as e:
        logger.error(f"Error parsing request body: {e}")
        return []


@router.post("/api/event_logging/batch")
async def event_logging_batch(request: Request, _: None = Depends(validate_api_key)):
    """
    Flexible event logging endpoint that appends JSON lines to Claude-proxy.log
    Accepts various input formats:
    - JSON array: [{"event_type": "...", "data": {...}}, ...]
    - Single object: {"event_type": "...", "data": {...}}
    - Invalid JSON with unquoted properties (auto-fixed)

    Includes request timestamp and client IP
    Implements log rotation at 10MB
    """
    try:
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"

        # Get current timestamp
        timestamp = datetime.now().isoformat()

        # Parse events with flexible format handling
        events = await parse_flexible_events(request)

        # If no events could be parsed, still return 200 but with 0 events
        if not events:
            logger.warning(f"No events could be parsed from request from {client_ip}")
            return JSONResponse(
                status_code=200,
                content={
                    "status": "success",
                    "message": "No valid events found in request",
                    "timestamp": timestamp,
                    "events_logged": 0,
                    "note": "Request body may be malformed",
                },
            )

        # Define log file path
        log_file_path = "Claude-proxy.log"

        # Rotate log file if needed
        rotate_log_file(log_file_path, max_size_mb=10)

        # Append each event as JSON line with timestamp and client IP
        with open(log_file_path, "a", encoding="utf-8") as f:
            for event in events:
                log_entry = {"timestamp": timestamp, "client_ip": client_ip, "event": event}
                # Write as JSON line
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        logger.info(f"Processed batch of {len(events)} events from {client_ip}")

        # Return 200 OK
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": f"Processed {len(events)} events",
                "timestamp": timestamp,
                "events_logged": len(events),
            },
        )

    except Exception as e:
        logger.error(f"Error in event logging: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e), "timestamp": datetime.now().isoformat()},
        )


@router.get("/v1/models")
async def list_models(_: None = Depends(validate_api_key)):
    """List available models — Claude aliases + upstream-discovered models.

    Returns an Anthropic-compatible models listing that also includes
    every model available on the upstream provider (e.g. Nebius) at
    query time. This lets clients discover both proxied aliases and raw
    backend models for explicit routing.
    """
    model_entries = []
    seen = set()

    # Define model tiers with their backend mappings and multiple ID variants
    # The tier_name maps to the pattern in ModelManager (haiku->small, sonnet->middle, opus->big)
    model_tiers = [
        {
            "tier": "haiku",
            "backend": config.small_model,
            "variants": [
                ("claude-haiku-4-5-20251001", "Claude Haiku 4.5 (proxied)"),
                ("claude-haiku-4-5", "Claude Haiku 4.5 (proxied)"),
                ("claude-3-5-haiku-20241022", "Claude 3.5 Haiku (proxied)"),
            ],
        },
        {
            "tier": "sonnet",
            "backend": config.middle_model,
            "variants": [
                ("claude-sonnet-4-6", "Claude Sonnet 4.6 (proxied)"),
                ("claude-sonnet-4-5-20250929", "Claude Sonnet 4.5 (proxied)"),
                ("claude-sonnet-4-20250514", "Claude Sonnet 4 (proxied)"),
                ("claude-3-5-sonnet-20241022", "Claude 3.5 Sonnet (proxied)"),
            ],
        },
        {
            "tier": "opus",
            "backend": config.big_model,
            "variants": [
                ("claude-opus-4-7", "Claude Opus 4.7 (proxied)"),
                ("claude-opus-4-6", "Claude Opus 4.6 (proxied)"),
                ("claude-opus-4-5-20251101", "Claude Opus 4.5 (proxied)"),
                ("claude-opus-4-20250514", "Claude Opus 4 (proxied)"),
            ],
        },
        {
            "tier": "vision",
            "backend": config.vision_model,
            "variants": [
                ("claude-haiku-4-5-20251001", "Claude Haiku 4.5 Vision (proxied)"),
            ],
        },
    ]

    for tier_config in model_tiers:
        backend = tier_config["backend"]
        for claude_id, display_name in tier_config["variants"]:
            if claude_id not in seen:
                seen.add(claude_id)
                model_entries.append(
                    {
                        "id": claude_id,
                        "object": "model",
                        "created": 1700000000,
                        "owned_by": "anthropic-proxy",
                        "display_name": f"{display_name.replace(' (proxied)', '')} → {backend}",
                        "backend_model": backend,
                    }
                )

    # Also include any custom model configurations from env
    if config.big_model:
        custom_models = [
            (config.big_model, "BIG model"),
            (config.middle_model, "MIDDLE model"),
            (config.small_model, "SMALL model"),
            (config.vision_model, "VISION model"),
        ]
        for model_id, model_type in custom_models:
            if model_id and model_id not in seen:
                seen.add(model_id)
                model_entries.append(
                    {
                        "id": model_id,
                        "object": "model",
                        "created": 1700000000,
                        "owned_by": "anthropic-proxy",
                        "display_name": f"Custom {model_type} → {model_id}",
                        "backend_model": model_id,
                    }
                )

    # Fetch upstream models dynamically and append any not already listed
    upstream_models = openai_client.list_models()
    for m in upstream_models:
        model_id = m.get("id")
        if model_id and model_id not in seen:
            seen.add(model_id)
            model_entries.append(
                {
                    "id": model_id,
                    "object": "model",
                    "created": m.get("created") or 1700000000,
                    "owned_by": m.get("owned_by") or "upstream",
                    "display_name": m.get("id"),
                    "backend_model": model_id,
                }
            )

    return {
        "object": "list",
        "data": model_entries,
    }


@router.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Claude-to-OpenAI API Proxy v1.0.0",
        "status": "running",
        "config": {
            "openai_base_url": config.openai_base_url,
            "max_tokens_limit": config.max_tokens_limit,
            "api_key_configured": bool(config.openai_api_key),
            "client_api_key_validation": bool(config.anthropic_api_key),
            "big_model": config.big_model,
            "small_model": config.small_model,
        },
        "endpoints": {
            "messages": "/v1/messages",
            "responses": "/v1/responses",
            "models": "/v1/models",
            "count_tokens": "/v1/messages/count_tokens",
            "health": "/health",
            "test_connection": "/test-connection",
            "event_logging_batch": "/api/event_logging/batch",
        },
    }
