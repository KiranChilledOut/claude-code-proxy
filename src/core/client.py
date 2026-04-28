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

    def cancel_request(self, request_id: str) -> bool:
        """Cancel an active request by request_id."""
        if request_id in self.active_requests:
            self.active_requests[request_id].set()
            return True
        return False
