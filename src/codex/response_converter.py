"""Convert OpenAI Chat Completions responses to Codex Responses API format."""

import uuid
from typing import Any, Dict, List, Optional, Union

from src.codex.models import (
    InputTokensDetails,
    OutputTokensDetails,
    ResponsesItem,
    ResponsesResponse,
    ResponsesUsage,
)


def _build_usage(usage_raw: Any) -> ResponsesUsage:
    """Map OpenAI usage fields into a ResponsesUsage model."""
    if not isinstance(usage_raw, dict):
        # Handle Pydantic model objects (OpenAI SDK v1/v2 sometimes returns these)
        usage_raw = usage_raw.model_dump() if hasattr(usage_raw, "model_dump") else (
            usage_raw.dict() if hasattr(usage_raw, "dict") else {})

    input_tokens_details = None
    if "prompt_tokens_details" in usage_raw:
        prompt_details = usage_raw["prompt_tokens_details"]
        if not isinstance(prompt_details, dict):
            prompt_details = prompt_details.model_dump() if hasattr(prompt_details, "model_dump") else {}
        input_tokens_details = InputTokensDetails(
            cached_tokens=prompt_details.get("cached_tokens", 0)
        )

    output_tokens_details = None
    if "completion_tokens_details" in usage_raw:
        completion_details = usage_raw["completion_tokens_details"]
        if not isinstance(completion_details, dict):
            completion_details = completion_details.model_dump() if hasattr(completion_details, "model_dump") else {}
        output_tokens_details = OutputTokensDetails(
            reasoning_tokens=completion_details.get("reasoning_tokens", 0)
        )

    cache_creation = None
    cache_read = None
    if "prompt_tokens_details" in usage_raw:
        ptd = usage_raw["prompt_tokens_details"]
        if isinstance(ptd, dict):
            cache_creation = ptd.get("cache_creation_tokens", 0) or None
            cache_read = ptd.get("cached_tokens", 0) or None

    return ResponsesUsage(
        input_tokens=usage_raw.get("prompt_tokens", 0),
        output_tokens=usage_raw.get("completion_tokens", 0),
        total_tokens=usage_raw.get("total_tokens", 0),
        input_tokens_details=input_tokens_details,
        output_tokens_details=output_tokens_details,
        cache_creation_input_tokens=cache_creation,
        cache_read_input_tokens=cache_read,
    )


def _build_output_items(
    message: Dict[str, Any], tool_ctx: Any = None
) -> List[ResponsesItem]:
    """Build ResponsesItem list from an OpenAI message dict."""
    items: List[ResponsesItem] = []

    # Text content
    content = message.get("content")
    if content:
        items.append(
            ResponsesItem(
                type="message",
                role="assistant",
                content=content,
                status="completed",
            )
        )

    # Tool calls
    tool_calls = message.get("tool_calls")
    if tool_calls:
        for tool_call in tool_calls:
            tc = tool_call if isinstance(tool_call, dict) else {
                "id": tool_call.id,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            func = tc.get("function", {}) if isinstance(tc, dict) else tc.function
            items.append(
                ResponsesItem(
                    type="function_call",
                    call_id=tc.get("id") if isinstance(tc, dict) else tc.id,
                    name=func.get("name") if isinstance(func, dict) else func.name,
                    arguments=func.get("arguments") if isinstance(func, dict) else func.arguments,
                    status="in_progress",
                )
            )

    # Remap tool calls if a tool context is available
    if tool_ctx is not None and hasattr(tool_ctx, "remap_tool_calls_back"):
        items = tool_ctx.remap_tool_calls_back(items)

    # Guard: never return an empty output array
    if not items:
        items = [
            ResponsesItem(
                type="message",
                role="assistant",
                content="",
                status="completed",
            )
        ]

    return items


def convert_openai_to_responses(
    openai_response: Dict[str, Any],
    request_model: str,
    previous_id: Optional[str] = None,
    tool_ctx: Any = None,
) -> ResponsesResponse:
    """Convert a non-streaming OpenAI Chat Completion response to Codex Responses API format.

    Args:
        openai_response: A dict shaped like ``completion.model_dump()``.
        request_model: The model name to surface in the Codex response.
        previous_id: Optional previous response id.
        tool_ctx: Optional tool context that can remap proxy names back to original names.

    Returns:
        A ``ResponsesResponse`` Pydantic model.
    """
    choices = openai_response.get("choices", [])
    if choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message", {})
        else:
            message = getattr(first, "message", None) or {}
    else:
        message = {}

    if isinstance(message, dict):
        message_dict = message
    else:
        message_dict = {
            "content": getattr(message, "content", None),
            "tool_calls": getattr(message, "tool_calls", None),
        }

    output_items = _build_output_items(message_dict, tool_ctx=tool_ctx)

    usage_raw = openai_response.get("usage", {})
    usage = _build_usage(usage_raw)

    return ResponsesResponse(
        id=str(uuid.uuid4()),
        model=request_model,
        output=output_items,
        status="completed",
        previous_id=previous_id,
        usage=usage,
    )


def convert_openai_error_to_responses(error: Any) -> ResponsesResponse:
    """Convert a non-streaming upstream error into a failed ``ResponsesResponse``."""
    return ResponsesResponse(
        id=str(uuid.uuid4()),
        model="unknown",
        output=[
            ResponsesItem(
                type="message",
                role="assistant",
                content=f"Error: {error}",
            )
        ],
        status="failed",
        usage=ResponsesUsage(input_tokens=0, output_tokens=0, total_tokens=0),
    )
