"""Convert Codex Responses API requests to OpenAI Chat Completions format."""

from typing import Any, Dict, List, Optional, Tuple, Union

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
_TEXT_BLOCK_TYPES = {"text", "input_text", "output_text"}
_IMAGE_BLOCK_TYPES = {"image", "input_image", "image_url"}


def _image_url_part(url: str, detail: Optional[str] = None) -> Dict[str, Any]:
    image_url: Dict[str, Any] = {"url": url}
    if detail:
        image_url["detail"] = detail
    return {"type": "image_url", "image_url": image_url}


def _normalize_image_url_payload(
    payload: Any, detail: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Normalize Responses/OpenAI-ish image URL payloads to Chat Completions shape."""
    if isinstance(payload, str):
        if not payload:
            return None
        return _image_url_part(payload, detail=detail)

    if isinstance(payload, dict):
        image_url = dict(payload)
        if "url" not in image_url:
            return None
        if detail and "detail" not in image_url:
            image_url["detail"] = detail
        return {"type": "image_url", "image_url": image_url}

    return None


def _convert_image_block(block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    block_type = block.get("type")
    detail = block.get("detail")

    if block_type in ("input_image", "image_url") or "image_url" in block:
        return _normalize_image_url_payload(block.get("image_url") or block.get("url"), detail)

    if block_type == "image":
        source = block.get("source")
        if isinstance(source, dict):
            if source.get("type") == "base64" and source.get("media_type") and source.get("data"):
                return _image_url_part(
                    f"data:{source['media_type']};base64,{source['data']}",
                    detail=detail,
                )
            if source.get("type") == "url":
                return _normalize_image_url_payload(source.get("url"), detail)
        return _normalize_image_url_payload(block.get("url"), detail)

    return None


def _convert_content_block(block: Any) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    if isinstance(block, str):
        return block, None

    if not isinstance(block, dict):
        return None, None

    block_type = block.get("type")
    if block_type in _TEXT_BLOCK_TYPES:
        return block.get("text", ""), None

    if block_type in _IMAGE_BLOCK_TYPES or "image_url" in block:
        return None, _convert_image_block(block)

    return None, None


def _convert_content_blocks(content: Any) -> Union[str, List[Dict[str, Any]]]:
    """Convert ResponsesItem content to OpenAI Chat Completions message content.

    String content is returned as-is. Text-only block lists are reduced to a
    plain string. When image blocks are present, content is returned in the
    OpenAI Chat Completions multimodal shape.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, dict):
        content = [content]

    if isinstance(content, list):
        texts: List[str] = []
        parts: List[Dict[str, Any]] = []
        has_image = False
        for block in content:
            text, image_part = _convert_content_block(block)
            if image_part is not None:
                has_image = True
                if texts:
                    parts.append({"type": "text", "text": "".join(texts)})
                    texts = []
                parts.append(image_part)
            elif text is not None:
                texts.append(text)

        if has_image:
            if texts:
                parts.append({"type": "text", "text": "".join(texts)})
            return parts

        return "".join(texts)

    return "" if content is None else str(content)


def _convert_item_to_message(item: ResponsesItem) -> Optional[Dict[str, Any]]:
    """Convert a single ResponsesItem to an OpenAI Chat Completions message dict.

    Returns ``None`` for unsupported item types so callers can simply drop them.
    """
    item_type = getattr(item, "type", None)

    if item_type == "message":
        role = getattr(item, "role", "")
        if role in ("developer", "system"):
            role = "system"
        if role not in ("system", "user", "assistant"):
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
            "content": _convert_content_blocks(output),
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

    # --- Model ---
    if model_manager is not None:
        mapped_model = model_manager.map_codex_model(request.model)
    else:
        mapped_model = _default_map_codex_model(request.model)

    has_image = bool(
        model_manager
        and hasattr(model_manager, "contains_image_content")
        and model_manager.contains_image_content(messages)
    )
    if has_image:
        vision_model = getattr(getattr(model_manager, "config", None), "vision_model", None)
        if vision_model:
            mapped_model = vision_model

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
    # Tool choice passthrough when tool_ctx is absent
    if tool_ctx is None and request.tool_choice is not None:
        if isinstance(request.tool_choice, str):
            if request.tool_choice in ("auto", "required", "none"):
                result["tool_choice"] = request.tool_choice
        else:
            # Object form — passthrough as-is when not handled by tool_ctx
            result["tool_choice"] = request.tool_choice

    # Vision endpoints commonly reject tool use; mirror the Claude image path.
    if has_image:
        result.pop("tools", None)
        result["tool_choice"] = "none"

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

    # --- Search nudge ---
    if tool_ctx is not None and getattr(tool_ctx, "has_search_tool", False):
        from src.conversion.server_tools import SEARCH_TOOL_SYSTEM_SUPPLEMENT
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] += "\n\n" + SEARCH_TOOL_SYSTEM_SUPPLEMENT
        else:
            messages.insert(0, {"role": "system", "content": SEARCH_TOOL_SYSTEM_SUPPLEMENT})

    # --- Reasoning effort ---
    if request.reasoning is not None:
        result["reasoning_effort"] = _map_reasoning_effort(request.reasoning)

    return result
