"""Convert Codex Responses API requests to OpenAI Chat Completions format."""

from typing import Any, Dict, List, Optional, Union

from src.codex.models import ResponsesItem, ResponsesRequest


# ---------------------------------------------------------------------------
# Reasoning effort mapping
# ---------------------------------------------------------------------------
_REASONING_EFFORT_MAP = {
    "none": "none",
    "auto": "auto",
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
}


def _map_reasoning_effort(reasoning: Optional[Dict[str, Any]]) -> str:
    """Map Codex reasoning.effort to OpenAI reasoning_effort.

    Defaults to "auto" when reasoning is absent or the effort value is unknown.
    """
    if not reasoning:
        return "auto"
    effort = reasoning.get("effort", "")
    if not effort:
        return "auto"
    return _REASONING_EFFORT_MAP.get(effort, "auto")


# ---------------------------------------------------------------------------
# Content helpers
# ---------------------------------------------------------------------------
def _convert_content_blocks(content: Any) -> Union[str, List[Dict[str, Any]]]:
    """Convert ResponsesItem content to OpenAI Chat Completions message content.

    String content is returned as-is.  A list of content blocks is reduced to a
    plain string (text blocks concatenated; image blocks are skipped).
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))
                # Image blocks are skipped for now (OpenAI Chat Completions
                # supports vision via a different shape, not used here).
            elif isinstance(block, str):
                texts.append(block)
        return "".join(texts)

    return "" if content is None else str(content)


def _convert_item_to_message(item: ResponsesItem) -> Optional[Dict[str, Any]]:
    """Convert a single ResponsesItem to an OpenAI Chat Completions message dict.

    Returns ``None`` for unsupported item types so callers can simply drop them.
    """
    item_type = getattr(item, "type", None)

    if item_type == "message":
        role = getattr(item, "role", "")
        if role not in ("user", "assistant"):
            # Malformed message — no valid OpenAI role to map to.
            return None
        content = _convert_content_blocks(getattr(item, "content", None))
        return {"role": role, "content": content}

    if item_type == "function_call":
        call_id = getattr(item, "call_id", "")
        name = getattr(item, "name", "")
        arguments = getattr(item, "arguments", "")
        return {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments if arguments is not None else "",
                    },
                }
            ],
        }

    if item_type == "function_call_output":
        call_id = getattr(item, "call_id", "")
        output = getattr(item, "output", None)
        return {
            "role": "tool",
            "content": output if output is not None else "",
            "tool_call_id": call_id,
        }

    if item_type == "text":
        text = getattr(item, "text", "")
        if text is None:
            text = getattr(item, "content", "")
            text = text if text is not None else ""
        return {"role": "user", "content": text}

    # Unsupported type (e.g. "image")
    return None


# ---------------------------------------------------------------------------
# Model mapping fallback
# ---------------------------------------------------------------------------
def _default_map_codex_model(codex_model: str) -> str:
    """Fallback model mapping when no ModelManager is provided."""
    lower = codex_model.lower()
    if "mini" in lower:
        return "zai-org/GLM-4.5"  # same default as Config.small_model
    return "zai-org/GLM-4.5"  # same default as Config.big_model


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------
def convert_responses_to_openai_chat(
    request: ResponsesRequest,
    session_items: Optional[List[ResponsesItem]] = None,
    tool_ctx: Any = None,
    model_manager: Any = None,
) -> Dict[str, Any]:
    """Convert a Codex ``ResponsesRequest`` into an OpenAI Chat Completions request dict.

    :param request: Codex request model.
    :param session_items: Previous turn messages already in OpenAI format (optional).
    :param tool_ctx: CodexToolContext or ``None``; provides tool conversion.
    :param model_manager: ``ModelManager`` instance or ``None`` for fallback mapping.
    :return: Dict suitable for passing to an OpenAI-compatible ``chat.completions.create`` call.
    """
    # --- Model ---
    if model_manager is not None:
        mapped_model = model_manager.map_codex_model(request.model)
    else:
        mapped_model = _default_map_codex_model(request.model)

    # --- Messages ---
    messages: List[Dict[str, Any]] = []

    # 1. System message from instructions
    if request.instructions is not None:
        messages.append({"role": "system", "content": request.instructions})

    # 2. Session items (previous turn history)
    if session_items is not None:
        for item in session_items:
            if isinstance(item, dict):
                messages.append(item)
            else:
                msg = _convert_item_to_message(item)
                if msg is not None:
                    messages.append(msg)

    # 3. Convert request.input
    if isinstance(request.input, str):
        messages.append({"role": "user", "content": request.input})
    else:
        for item in request.input:
            msg = _convert_item_to_message(item)
            if msg is not None:
                messages.append(msg)

    # --- Build output dict ---
    result: Dict[str, Any] = {
        "model": mapped_model,
        "messages": messages,
        "stream": request.stream,
    }

    # --- Tools ---
    if tool_ctx is not None:
        # When tool_ctx is present, let it decide what tools/tool_choice look like
        if hasattr(tool_ctx, "tools") and tool_ctx.tools is not None:
            result["tools"] = tool_ctx.tools
        if hasattr(tool_ctx, "map_tool_choice") and request.tool_choice is not None:
            result["tool_choice"] = tool_ctx.map_tool_choice(request.tool_choice)
    elif request.tools is not None:
        # Simple passthrough — Codex "tools" may already be OpenAI functions,
        # just pass them through when there is no conversion layer.
        result["tools"] = request.tools
    elif request.tools is not None and tool_ctx is None:
        # same path, but ensure tool_choice passthrough when no tool_ctx
        pass

    # Tool choice passthrough when tool_ctx is absent
    if tool_ctx is None and request.tool_choice is not None:
        if isinstance(request.tool_choice, str):
            if request.tool_choice in ("auto", "required", "none"):
                result["tool_choice"] = request.tool_choice
        else:
            # Object form — passthrough as-is when not handled by tool_ctx
            result["tool_choice"] = request.tool_choice

    # --- Generation parameters ---
    if request.max_output_tokens is not None:
        result["max_tokens"] = request.max_output_tokens
    if request.temperature is not None:
        result["temperature"] = request.temperature
    if request.top_p is not None:
        result["top_p"] = request.top_p
    if request.user is not None:
        result["user"] = request.user

    # --- Stream options ---
    if request.stream:
        result["stream_options"] = {"include_usage": True}

    # --- Reasoning effort ---
    if request.reasoning is not None:
        result["reasoning_effort"] = _map_reasoning_effort(request.reasoning)

    return result
