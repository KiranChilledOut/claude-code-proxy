import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import HTTPException
from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai._exceptions import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)

logger = logging.getLogger(__name__)

# Models that returned a 400 specifically because of `reasoning_effort`. Once a
# model rejects it, we stop sending it (avoids repeating the failed call). This
# lets effort be forwarded dynamically without permanent latency on backends
# that don't support the parameter.
_EFFORT_UNSUPPORTED_MODELS: set = set()


def reasoning_effort_supported(model: str) -> bool:
    """False once a backend model has rejected `reasoning_effort`."""
    return model not in _EFFORT_UNSUPPORTED_MODELS


def _maybe_drop_reasoning_effort(req: Dict[str, Any], error: Exception) -> bool:
    """If a 400 was caused by `reasoning_effort`, drop it (and remember the
    model can't take it) so the caller can retry once. Returns True if dropped."""
    if "reasoning_effort" not in req:
        return False
    if "reasoning_effort" not in str(error).lower():
        return False
    model = req.get("model")
    if model:
        _EFFORT_UNSUPPORTED_MODELS.add(model)
    req.pop("reasoning_effort", None)
    logger.warning(
        "Backend rejected reasoning_effort for model %s; retrying without it "
        "and disabling it for this model.",
        model,
    )
    return True


_CONTEXT_LEN_MARKERS = (
    "context length",
    "context_length",
    "maximum context",
    "context window",
    "reduce the length",
    "too many tokens",
    "too long",
)


def _is_context_length_error(error: Exception) -> bool:
    detail = str(error).lower()
    return any(marker in detail for marker in _CONTEXT_LEN_MARKERS)


def _maybe_retrim_context(req: Dict[str, Any], error: Exception) -> bool:
    """On an upstream context-length 400, drop the oldest messages (using the
    existing safe trimmer, which preserves the system message, the latest turn,
    and atomic tool pairs) and signal a single retry. Returns True if trimmed.

    Bounded by the caller's retry loop and the `dropped > 0` guard, so it cannot
    loop forever: once nothing more can be dropped, the error is surfaced.
    """
    if not _is_context_length_error(error):
        return False
    messages = req.get("messages")
    if not isinstance(messages, list) or len(messages) <= 2:
        return False
    # Imported lazily to avoid a request_converter <-> client import cycle.
    from src.conversion.request_converter import (
        _estimate_prompt_tokens,
        _trim_messages_to_fit,
    )

    estimate = _estimate_prompt_tokens(messages)
    target = max(int(estimate * 0.7), 1)  # force shedding ~30% of the prompt
    trimmed, dropped = _trim_messages_to_fit(messages, target, reserve=2048)
    if dropped <= 0:
        return False
    req["messages"] = trimmed
    logger.warning(
        "Upstream rejected for context length; dropped %d oldest message(s) and retrying.",
        dropped,
    )
    return True


class OpenAIClient:
    """Async OpenAI client with cancellation support."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout: int = 90,
        api_version: Optional[str] = None,
        custom_headers: Optional[Dict[str, str]] = None,
        max_retries: int = 2,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.custom_headers = custom_headers or {}
        self.max_retries = max(0, max_retries)
        self.retry_backoff_seconds = 0.5

        # Prepare default headers
        default_headers = {"Content-Type": "application/json", "User-Agent": "claude-proxy/1.0.0"}

        # Merge custom headers with default headers
        all_headers = {**default_headers, **self.custom_headers}

        # Detect if using Azure and instantiate the appropriate client
        if api_version:
            self.client = AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=base_url,
                api_version=api_version,
                timeout=timeout,
                default_headers=all_headers,
            )
        else:
            self.client = AsyncOpenAI(
                api_key=api_key, base_url=base_url, timeout=timeout, default_headers=all_headers
            )
        self.active_requests: Dict[str, asyncio.Event] = {}

    def _should_retry(self, error: Exception) -> bool:
        if isinstance(error, (RateLimitError, APIConnectionError, APITimeoutError)):
            return True
        if isinstance(error, APIError):
            status_code = getattr(error, "status_code", None)
            return status_code is None or status_code >= 500 or status_code in (408, 429)
        return False

    async def _sleep_before_retry(self, attempt: int) -> None:
        delay = self.retry_backoff_seconds * (2**attempt)
        await asyncio.sleep(delay)

    async def create_chat_completion(
        self, request: Dict[str, Any], request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send chat completion to OpenAI API with cancellation support."""

        # Create cancellation token if request_id provided
        if request_id:
            cancel_event = asyncio.Event()
            self.active_requests[request_id] = cancel_event

        try:
            for attempt in range(self.max_retries + 1):
                try:
                    # Create task that can be cancelled
                    completion_task = asyncio.create_task(
                        self.client.chat.completions.create(**request)
                    )

                    if request_id:
                        # Wait for either completion or cancellation
                        cancel_task = asyncio.create_task(cancel_event.wait())
                        done, pending = await asyncio.wait(
                            [completion_task, cancel_task], return_when=asyncio.FIRST_COMPLETED
                        )

                        # Cancel pending tasks
                        for task in pending:
                            task.cancel()
                            try:
                                await task
                            except asyncio.CancelledError:
                                pass

                        # Check if request was cancelled
                        if cancel_task in done:
                            completion_task.cancel()
                            raise HTTPException(
                                status_code=499, detail="Request cancelled by client"
                            )

                        completion = await completion_task
                    else:
                        completion = await completion_task

                    # Convert to dict format that matches the original interface
                    return completion.model_dump()
                except HTTPException:
                    raise
                except AuthenticationError as e:
                    self._log_openai_error(e)
                    raise HTTPException(status_code=401, detail=self.classify_openai_error(str(e)))
                except BadRequestError as e:
                    if _maybe_drop_reasoning_effort(request, e):
                        continue
                    if _maybe_retrim_context(request, e):
                        continue
                    self._log_openai_error(e)
                    raise HTTPException(status_code=400, detail=self.classify_openai_error(str(e)))
                except (RateLimitError, APIConnectionError, APITimeoutError, APIError) as e:
                    self._log_openai_error(e)
                    if self._should_retry(e) and attempt < self.max_retries:
                        logger.warning(
                            "OpenAI request failed (%s), retrying (%d/%d)",
                            type(e).__name__,
                            attempt + 1,
                            self.max_retries,
                        )
                        await self._sleep_before_retry(attempt)
                        continue
                    status_code = getattr(e, "status_code", None)
                    if isinstance(e, RateLimitError):
                        status_code = 429
                    raise HTTPException(
                        status_code=status_code or 500,
                        detail=self.classify_openai_error(str(e)),
                    )
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
            raise HTTPException(status_code=500, detail="Request failed after retries")

        finally:
            # Clean up active request tracking
            if request_id and request_id in self.active_requests:
                del self.active_requests[request_id]

    async def create_chat_completion_stream(
        self, request: Dict[str, Any], request_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Send streaming chat completion to OpenAI API with cancellation support."""

        # Create cancellation token if request_id provided
        if request_id:
            cancel_event = asyncio.Event()
            self.active_requests[request_id] = cancel_event

        try:
            # Ensure stream is enabled
            stream_request = dict(request)
            stream_request["stream"] = True
            stream_options = dict(stream_request.get("stream_options") or {})
            stream_options["include_usage"] = True
            stream_request["stream_options"] = stream_options

            streaming_completion = None
            for attempt in range(self.max_retries + 1):
                try:
                    streaming_completion = await self.client.chat.completions.create(
                        **stream_request
                    )
                    break
                except AuthenticationError as e:
                    self._log_openai_error(e)
                    raise HTTPException(status_code=401, detail=self.classify_openai_error(str(e)))
                except BadRequestError as e:
                    if _maybe_drop_reasoning_effort(stream_request, e):
                        continue
                    if _maybe_retrim_context(stream_request, e):
                        continue
                    self._log_openai_error(e)
                    raise HTTPException(status_code=400, detail=self.classify_openai_error(str(e)))
                except (RateLimitError, APIConnectionError, APITimeoutError, APIError) as e:
                    self._log_openai_error(e)
                    if self._should_retry(e) and attempt < self.max_retries:
                        logger.warning(
                            "OpenAI stream setup failed (%s), retrying (%d/%d)",
                            type(e).__name__,
                            attempt + 1,
                            self.max_retries,
                        )
                        await self._sleep_before_retry(attempt)
                        continue
                    status_code = getattr(e, "status_code", None)
                    if isinstance(e, RateLimitError):
                        status_code = 429
                    raise HTTPException(
                        status_code=status_code or 500,
                        detail=self.classify_openai_error(str(e)),
                    )

            if streaming_completion is None:
                raise HTTPException(status_code=500, detail="Stream setup failed after retries")

            async for chunk in streaming_completion:
                # Check for cancellation before yielding each chunk
                if request_id and request_id in self.active_requests:
                    if self.active_requests[request_id].is_set():
                        raise HTTPException(status_code=499, detail="Request cancelled by client")

                # Convert chunk to SSE format matching original HTTP client format
                chunk_dict = chunk.model_dump()
                chunk_json = json.dumps(chunk_dict, ensure_ascii=False)
                yield f"data: {chunk_json}"

            # Signal end of stream
            yield "data: [DONE]"
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

        finally:
            # Clean up active request tracking
            if request_id and request_id in self.active_requests:
                del self.active_requests[request_id]

    def classify_openai_error(self, error_detail: Any) -> str:
        """Provide specific error guidance for common OpenAI API issues."""
        error_str = str(error_detail).lower()

        # Region/country restrictions
        if (
            "unsupported_country_region_territory" in error_str
            or "country, region, or territory not supported" in error_str
        ):
            return "OpenAI API is not available in your region. Consider using a VPN or Azure OpenAI service."

        # API key issues
        if "invalid_api_key" in error_str or "unauthorized" in error_str:
            return "Invalid API key. Please check your OPENAI_API_KEY configuration."

        # Rate limiting
        if "rate_limit" in error_str or "quota" in error_str:
            return "Rate limit exceeded. Please wait and try again, or upgrade your API plan."

        # Model not found
        if "model" in error_str and ("not found" in error_str or "does not exist" in error_str):
            return "Model not found. Please check your BIG_MODEL and SMALL_MODEL configuration."

        # Billing issues
        if "billing" in error_str or "payment" in error_str:
            return "Billing issue. Please check your OpenAI account billing status."

        # Context window exceeded (after the proxy already tried to shed messages)
        if any(m in error_str for m in _CONTEXT_LEN_MARKERS):
            return (
                "Input exceeds the model's context window. Lower the conversation "
                "size, or set the correct <ROLE>_MODEL_CONTEXT_LIMIT for this backend."
            )

        # Default: return original message
        return str(error_detail)

    def _log_openai_error(self, error: Exception) -> None:
        response = getattr(error, "response", None)
        if response is not None:
            try:
                logger.error("OpenAI API error response body: %s", response.text)
            except Exception:
                logger.error("OpenAI API error response body: <unreadable>")
        body = getattr(error, "body", None)
        if body:
            logger.error("OpenAI API error body parsed: %s", body)

    def list_models(self) -> list[dict]:
        """Fetch available models from the upstream provider's /v1/models endpoint.

        Uses raw urllib (not the OpenAI SDK) to mirror how the TUI discovers models,
        sending a simple GET to BASE_URL + /models with Authorization: Bearer <key>.
        """
        import ssl
        import urllib.error
        import urllib.request

        endpoint = self.base_url.rstrip("/") + "/models"
        req = urllib.request.Request(
            endpoint, headers={"Authorization": f"Bearer {self.api_key}"}
        )
        ctx = ssl.create_default_context()
        try:
            with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
                data = json.load(resp)
                return [{"id": m["id"]} for m in data.get("data", []) if m.get("id")]
        except urllib.error.HTTPError as e:
            logger.warning("Upstream /models returned HTTP %d: %s", e.code, e.read().decode()[:200])
            return []
        except Exception as e:
            logger.warning("Upstream /models request failed: %s", e)
            return []

    def cancel_request(self, request_id: str) -> bool:
        """Cancel an active request by request_id."""
        if request_id in self.active_requests:
            self.active_requests[request_id].set()
            return True
        return False
