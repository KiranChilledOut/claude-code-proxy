import hashlib
import json
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from src.conversion.computer_use import (
    convert_schema_less_tools,
    is_computer_use_tool,
)
from src.conversion.server_tools import (
    SEARCH_TOOL_PARAMETERS,
    SEARCH_TOOL_SYSTEM_SUPPLEMENT,
    is_search_tool,
)
from src.core.client import reasoning_effort_supported
from src.core.config import config
from src.core.constants import Constants
from src.models.claude import ClaudeMessage, ClaudeMessagesRequest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prefix-cache discipline (Tier-1)
# ---------------------------------------------------------------------------
# vLLM / SGLang (the engines behind Nebius token-factory deployments) cache
# the KV state of any *byte-identical* prompt prefix. The proxy maximises
# cache hits for Claude Code's huge, repeated system + tool block by:
#   1) emitting tool-parameter JSON Schema with deterministic key ordering, and
#   2) logging a fingerprint of the cacheable prefix so operators can verify
#      reuse from logs without inspecting payloads.
# We do NOT reorder the tools list itself or the messages list — the model
# would observe those changes and we must not alter request semantics.


def _canonicalize_schema(node: Any) -> Any:
    """Recursively sort dict keys inside a JSON Schema sub-tree.

    JSON Schema does not assign meaning to property ordering, so sorting keys
    inside `parameters` produces a canonical wire form that gives prefix
    caches deterministic hits across requests, even when upstream serializers
    happen to emit keys in different orders.
    """
    if isinstance(node, dict):
        return {k: _canonicalize_schema(node[k]) for k in sorted(node.keys())}
    if isinstance(node, list):
        return [_canonicalize_schema(item) for item in node]
    return node


def _compute_prefix_fingerprint(
    system_message: Optional[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]]
) -> str:
    """Return a short sha256 hex digest of the cacheable prefix.

    The prefix here is the system message + the tools list — the part of the
    request that Claude Code sends nearly identically on every turn. A stable
    fingerprint across requests in the same session implies the upstream
    prefix cache will hit. Returns the first 12 hex chars (96 bits) — plenty
    for collision-resistance in operator logs.
    """
    payload = {
        "system": system_message,
        "tools": tools or [],
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]

# ---------------------------------------------------------------------------
# Tiktoken-based token counting (Feature 4)
# ---------------------------------------------------------------------------
# We use cl100k_base as a reasonable cross-model approximation. It's the
# encoding used by GPT-4 / GPT-3.5 and is close enough for context-window
# guard-rails even when the actual backend model uses a different tokenizer.
# If tiktoken fails to load (rare), we fall back to the old chars/4 heuristic.

_tiktoken_encoding = None
_tiktoken_available = False

try:
    import tiktoken

    _tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
    _tiktoken_available = True
    logger.info("tiktoken loaded successfully — using cl100k_base for token estimation")
except Exception as e:
    logger.warning(f"tiktoken not available, falling back to char-based estimation: {e}")

# Rough per-model context limits (tokens). Used to downscale max_tokens when
# prompts get close to the window. Configurable via env overrides in config.
# If no override is provided, we fall back to the safe default below.
DEFAULT_CONTEXT_LIMIT = 128000
# Extra safety buffer beyond the reserve passed to trimming.
TOKEN_ESTIMATE_BUFFER = 512


def _get_context_limit(model_name: str) -> int:
    """Get the context limit for a given model name.

    Checks config for model-specific context limit overrides (big, middle,
    small, vision). Falls back to DEFAULT_CONTEXT_LIMIT if no override.
    """
    # Per-role overrides from config
    if model_name == config.big_model and config.big_model_context_limit:
        return config.big_model_context_limit
    if model_name == config.middle_model and config.middle_model_context_limit:
        return config.middle_model_context_limit
    if model_name == config.small_model and config.small_model_context_limit:
        return config.small_model_context_limit
    if model_name == config.vision_model and config.vision_model_context_limit:
        return config.vision_model_context_limit

    # No prefix match; use safe default
    return DEFAULT_CONTEXT_LIMIT


def _map_client_effort(effort) -> Optional[str]:
    """Map an Anthropic effort level (output_config.effort) to a backend
    reasoning_effort. Backends generally accept low/medium/high, so the
    Anthropic-only `xhigh`/`max` levels collapse to `high`."""
    if not isinstance(effort, str):
        return None
    e = effort.strip().lower()
    if e in ("max", "xhigh", "high"):
        return "high"
    if e == "medium":
        return "medium"
    if e == "low":
        return "low"
    return None


def _client_effort(claude_request) -> Optional[str]:
    """Read output_config.effort from the request, if present."""
    oc = getattr(claude_request, "output_config", None)
    if isinstance(oc, dict):
        return oc.get("effort")
    if oc is not None:
        return getattr(oc, "effort", None)
    return None


def _resolve_reasoning_effort(claude_request) -> Optional[str]:
    """Forward the effort the user chose in Claude Code (output_config.effort)
    to the backend as reasoning_effort — mirroring how Claude Code's /effort
    flows through to Anthropic. Returns None when the client sent no effort."""
    return _map_client_effort(_client_effort(claude_request))


def _count_tokens_text(text: str) -> int:
    """Count tokens in a string using tiktoken or fallback."""
    if _tiktoken_available and _tiktoken_encoding is not None:
        return len(_tiktoken_encoding.encode(text, disallowed_special=()))
    # Fallback: chars / 4 with a conservative 1.35x bias
    return int(math.ceil(len(text) / 4 * 1.35))


def _estimate_prompt_tokens(
    messages: List[Dict[str, Any]], *, include_safety_buffer: bool = True
) -> int:
    """Estimate total prompt tokens using tiktoken (or char-based fallback).

    Accounts for text content, image tokens, and tool call arguments.
    Adds per-message overhead (role tokens, separators) consistent with
    the OpenAI chat format. Context-window guard rails include a safety
    buffer; observability estimates should disable it to avoid inflating cost.
    """
    total_tokens = 0
    image_bonus = 0
    PER_MESSAGE_OVERHEAD = 4  # <|start|>role\n ... <|end|>

    for msg in messages:
        total_tokens += PER_MESSAGE_OVERHEAD
        content = msg.get("content")
        if content is None:
            pass
        elif isinstance(content, str):
            total_tokens += _count_tokens_text(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        total_tokens += _count_tokens_text(block.get("text", ""))
                    elif block.get("type") == "image_url":
                        image_bonus += 400  # conservative per-image estimate
        # assistant tool calls
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                total_tokens += _count_tokens_text(fn.get("name", ""))
                total_tokens += _count_tokens_text(fn.get("arguments", ""))

    total_tokens += image_bonus
    if include_safety_buffer:
        total_tokens += TOKEN_ESTIMATE_BUFFER
    return total_tokens


def count_claude_request_tokens(claude_request) -> int:
    """Count tokens for a Claude-format request (system + messages + tools).

    This is the implementation that backs `POST /v1/messages/count_tokens`.
    It mirrors how the Claude API itself counts: system, every message
    (text / image / tool_use / tool_result), and every tool definition
    (name + description + input_schema). Tools matter a lot in practice —
    Claude Code's tool block alone is 5–10k tokens, and the previous
    estimator silently dropped them.

    Returns an integer ≥ 1.
    """
    total = 0

    # ---- system ----
    system = getattr(claude_request, "system", None)
    if isinstance(system, str):
        total += _count_tokens_text(system)
    elif isinstance(system, list):
        for block in system:
            text = (
                block.get("text")
                if isinstance(block, dict)
                else getattr(block, "text", None)
            )
            if text:
                total += _count_tokens_text(text)

    # ---- messages ----
    PER_MESSAGE_OVERHEAD = 4
    for msg in getattr(claude_request, "messages", []) or []:
        total += PER_MESSAGE_OVERHEAD
        content = getattr(msg, "content", None)
        if content is None:
            continue
        if isinstance(content, str):
            total += _count_tokens_text(content)
            continue
        if isinstance(content, list):
            for block in content:
                # Resolve type and fields whether block is dict or pydantic model
                if isinstance(block, dict):
                    btype = block.get("type")
                    btext = block.get("text")
                    bname = block.get("name")
                    binput = block.get("input")
                    bcontent = block.get("content")
                else:
                    btype = getattr(block, "type", None)
                    btext = getattr(block, "text", None)
                    bname = getattr(block, "name", None)
                    binput = getattr(block, "input", None)
                    bcontent = getattr(block, "content", None)

                if btype == Constants.CONTENT_TEXT and btext:
                    total += _count_tokens_text(btext)
                elif btype == Constants.CONTENT_IMAGE:
                    total += 400  # conservative per-image estimate
                elif btype == Constants.CONTENT_TOOL_USE:
                    # tool name + serialized input
                    if bname:
                        total += _count_tokens_text(bname)
                    if binput is not None:
                        try:
                            total += _count_tokens_text(
                                json.dumps(binput, ensure_ascii=False)
                            )
                        except (TypeError, ValueError):
                            total += _count_tokens_text(str(binput))
                elif btype == Constants.CONTENT_TOOL_RESULT:
                    if isinstance(bcontent, str):
                        total += _count_tokens_text(bcontent)
                    elif isinstance(bcontent, list):
                        for item in bcontent:
                            # Handle dict, string, or pydantic model objects
                            if isinstance(item, dict):
                                if item.get("type") == "text":
                                    total += _count_tokens_text(item.get("text", ""))
                                else:
                                    try:
                                        total += _count_tokens_text(
                                            json.dumps(item, ensure_ascii=False)
                                        )
                                    except (TypeError, ValueError):
                                        total += _count_tokens_text(str(item))
                            elif isinstance(item, str):
                                total += _count_tokens_text(item)
                            else:
                                # Pydantic model object — extract text field or serialize
                                item_text = getattr(item, "text", None)
                                if item_text:
                                    total += _count_tokens_text(item_text)
                                else:
                                    try:
                                        total += _count_tokens_text(
                                            json.dumps(item, ensure_ascii=False)
                                        )
                                    except (TypeError, ValueError):
                                        total += _count_tokens_text(str(item))
                    elif isinstance(bcontent, dict):
                        try:
                            total += _count_tokens_text(
                                json.dumps(bcontent, ensure_ascii=False)
                            )
                        except (TypeError, ValueError):
                            total += _count_tokens_text(str(bcontent))
                    else:
                        # Pydantic model object (e.g., ClaudeContentBlockToolResult)
                        bcontent_text = getattr(bcontent, "content", None)
                        if bcontent_text:
                            if isinstance(bcontent_text, str):
                                total += _count_tokens_text(bcontent_text)
                            elif isinstance(bcontent_text, list):
                                for sub_item in bcontent_text:
                                    if isinstance(sub_item, str):
                                        total += _count_tokens_text(sub_item)
                                    elif isinstance(sub_item, dict):
                                        try:
                                            total += _count_tokens_text(
                                                json.dumps(sub_item, ensure_ascii=False)
                                            )
                                        except (TypeError, ValueError):
                                            total += _count_tokens_text(str(sub_item))
                                    else:
                                        # Pydantic model
                                        sub_text = getattr(sub_item, "text", None)
                                        if sub_text:
                                            total += _count_tokens_text(sub_text)

    # ---- tools (the part the old estimator dropped on the floor) ----
    tools = getattr(claude_request, "tools", None)
    if tools:
        # Anthropic's count_tokens charges for tool name, description, and
        # the full JSON schema. We approximate by serializing the same
        # function-tool form the proxy emits upstream.
        for tool in tools:
            tname = getattr(tool, "name", None) or ""
            tdesc = getattr(tool, "description", None) or ""
            tschema = getattr(tool, "input_schema", None)
            ttype = getattr(tool, "type", None)

            if tname:
                total += _count_tokens_text(tname)
            if tdesc:
                total += _count_tokens_text(tdesc)
            if tschema:
                try:
                    total += _count_tokens_text(json.dumps(tschema, ensure_ascii=False))
                except (TypeError, ValueError):
                    total += _count_tokens_text(str(tschema))
            elif ttype:
                # Schema-less Anthropic tool (computer/bash/text_editor).
                # Use the same baked-in schema we'll send to the backend.
                from src.conversion.computer_use import get_schema_for_tool

                inferred = get_schema_for_tool(tool)
                if inferred is not None:
                    total += _count_tokens_text(json.dumps(inferred, ensure_ascii=False))

    return max(total, 1)


def _group_messages_by_tool_pair(
    messages: List[Dict[str, Any]],
) -> List[List[Dict[str, Any]]]:
    """Bundle each assistant-with-tool_calls together with its tool replies.

    OpenAI-compatible backends (and Claude itself) reject conversations where
    a `role=tool` message has no preceding assistant with a matching
    tool_call_id. To trim safely we treat (assistant + immediately-following
    tool replies) as one atomic group that can only be dropped together.
    Every other message is its own group.
    """
    groups: List[List[Dict[str, Any]]] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get("role") == Constants.ROLE_ASSISTANT and msg.get("tool_calls"):
            group = [msg]
            j = i + 1
            while j < len(messages) and messages[j].get("role") == Constants.ROLE_TOOL:
                group.append(messages[j])
                j += 1
            groups.append(group)
            i = j
        else:
            groups.append([msg])
            i += 1
    return groups


def _trim_messages_to_fit(
    messages: List[Dict[str, Any]], context_limit: int, reserve: int = 2048
) -> Tuple[List[Dict[str, Any]], int]:
    """Drop oldest messages until the prompt fits, preserving tool pairs.

    Rules:
      * The system message (if any) is always preserved.
      * The most recent group is always preserved (so the user always sees a
        response to their latest turn, and any in-flight tool exchange stays
        intact).
      * An assistant message with `tool_calls` and its following `role=tool`
        replies are dropped together — never half-dropped, which would
        produce an orphan tool_result and a 400 from the backend.

    Returns (trimmed_messages, dropped_count) where dropped_count is the
    number of *messages* removed (not groups), to keep observability simple.
    """
    if not messages:
        return messages, 0

    groups = _group_messages_by_tool_pair(messages)
    dropped = 0

    while groups:
        flat = [m for g in groups for m in g]
        est = _estimate_prompt_tokens(flat)
        if est <= max(context_limit - reserve, 1):
            break

        drop_idx = None
        for k, g in enumerate(groups):
            # Never drop the most recent group — it's the in-flight turn.
            if k == len(groups) - 1:
                continue
            # Never drop a system message.
            if any(m.get("role") == Constants.ROLE_SYSTEM for m in g):
                continue
            drop_idx = k
            break

        if drop_idx is None:
            # Nothing safe left to drop. Bail out and let downstream see
            # the current size — the upstream may still accept it.
            break

        dropped += len(groups[drop_idx])
        groups.pop(drop_idx)

    flat = [m for g in groups for m in g]
    return flat, dropped


def convert_claude_to_openai(
    claude_request: ClaudeMessagesRequest, model_manager
) -> Dict[str, Any]:
    """Convert Claude API request format to OpenAI format."""
    allow_tools = not config.disable_tools
    # Only treat the latest user message as image-bearing for routing/tool decisions
    has_image = bool(
        model_manager
        and model_manager.contains_image_content(claude_request.messages, latest_user_only=True)
    )

    # Map model
    openai_model = model_manager.map_claude_model_to_openai(
        claude_request.model, claude_request.messages
    )
    logger.info(
        f"Selected model: {openai_model} for request with {len(claude_request.messages)} messages"
    )

    # Convert messages
    openai_messages = []

    # Special handling for image requests: to avoid blowing the smaller vision
    # model's context window, send only the latest user turn that carries the
    # image (plus an optional short system prompt when allowed). The rest of the
    # conversation stays on the Claude side and resumes with the text model.
    if has_image:
        # Optional system message (only when we are not stripping image context)
        if claude_request.system and not config.strip_image_context:
            system_text = ""
            if isinstance(claude_request.system, str):
                system_text = claude_request.system
            elif isinstance(claude_request.system, list):
                text_parts = []
                for block in claude_request.system:
                    if hasattr(block, "type") and block.type == Constants.CONTENT_TEXT:
                        text_parts.append(block.text)
                    elif isinstance(block, dict) and block.get("type") == Constants.CONTENT_TEXT:
                        text_parts.append(block.get("text", ""))
                system_text = "\n\n".join(text_parts)

            if system_text.strip():
                openai_messages.append(
                    {"role": Constants.ROLE_SYSTEM, "content": system_text.strip()}
                )

        # Find the most recent user message that contains an image and only send that
        latest_image_msg = None
        for message in reversed(claude_request.messages):
            if message.role == Constants.ROLE_USER and model_manager.contains_image_content(
                [message]
            ):
                latest_image_msg = message
                break

        if latest_image_msg:
            openai_messages.append(convert_claude_user_message(latest_image_msg, allow_images=True))
    else:
        # Original multi-turn handling for text-only flow
        # Add system message if present
        if claude_request.system:
            system_text = ""
            if isinstance(claude_request.system, str):
                system_text = claude_request.system
            elif isinstance(claude_request.system, list):
                text_parts = []
                for block in claude_request.system:
                    if hasattr(block, "type") and block.type == Constants.CONTENT_TEXT:
                        text_parts.append(block.text)
                    elif isinstance(block, dict) and block.get("type") == Constants.CONTENT_TEXT:
                        text_parts.append(block.get("text", ""))
                system_text = "\n\n".join(text_parts)

            if system_text.strip():
                openai_messages.append(
                    {"role": Constants.ROLE_SYSTEM, "content": system_text.strip()}
                )

        # Process Claude messages
        i = 0
        while i < len(claude_request.messages):
            msg = claude_request.messages[i]

            if msg.role == Constants.ROLE_SYSTEM:
                # Inline system message (e.g. from SessionStart hooks). Flatten
                # to text and append as an OpenAI system turn.
                sys_text = ""
                if isinstance(msg.content, str):
                    sys_text = msg.content
                elif isinstance(msg.content, list):
                    parts = []
                    for block in msg.content:
                        if hasattr(block, "type") and block.type == Constants.CONTENT_TEXT:
                            parts.append(block.text)
                        elif isinstance(block, dict) and block.get("type") == Constants.CONTENT_TEXT:
                            parts.append(block.get("text", ""))
                    sys_text = "\n\n".join(parts)
                if sys_text.strip():
                    openai_messages.append(
                        {"role": Constants.ROLE_SYSTEM, "content": sys_text.strip()}
                    )
            elif msg.role == Constants.ROLE_USER:
                openai_message = convert_claude_user_message(msg, allow_images=has_image)
                openai_messages.append(openai_message)
            elif msg.role == Constants.ROLE_ASSISTANT:
                openai_message = convert_claude_assistant_message(msg, allow_tools=allow_tools)
                openai_messages.append(openai_message)

                # Check if next message contains tool results
                if allow_tools and i + 1 < len(claude_request.messages):
                    next_msg = claude_request.messages[i + 1]
                    if (
                        next_msg.role == Constants.ROLE_USER
                        and isinstance(next_msg.content, list)
                        and any(
                            block.type == Constants.CONTENT_TOOL_RESULT
                            for block in next_msg.content
                            if hasattr(block, "type")
                        )
                    ):
                        # Process tool results
                        i += 1  # Skip to tool result message
                        tool_results = convert_claude_tool_results(next_msg)
                        openai_messages.extend(tool_results)

            i += 1

    # Build OpenAI request
    # Context trimming + max_tokens guard
    context_limit = _get_context_limit(openai_model)
    openai_messages, dropped = _trim_messages_to_fit(openai_messages, context_limit, reserve=2048)
    if dropped:
        logger.warning(
            f"Trimmed {dropped} oldest messages to fit context window for model {openai_model}"
        )

    prompt_estimate = _estimate_prompt_tokens(openai_messages)
    available = max(context_limit - prompt_estimate - 2048, 1)
    # Respect client intent; treat MIN_TOKENS_LIMIT as a fallback for missing/invalid
    # values instead of forcing an oversized floor.
    requested = claude_request.max_tokens
    if not isinstance(requested, int) or requested < 1:
        requested = config.min_tokens_limit

    safe_max_tokens = min(requested, config.max_tokens_limit, available)

    openai_request = {
        "model": openai_model,
        "messages": openai_messages,
        "max_tokens": safe_max_tokens,
        "temperature": claude_request.temperature,
        "stream": claude_request.stream,
    }
    # Opt-in reasoning passthrough so reasoning-capable backends actually think.
    reasoning_effort = _resolve_reasoning_effort(claude_request)
    if reasoning_effort and reasoning_effort_supported(openai_model):
        openai_request["reasoning_effort"] = reasoning_effort
    logger.debug(
        f"Converted Claude request to OpenAI format: {json.dumps(openai_request, indent=2, ensure_ascii=False)}"
    )
    # Add optional parameters
    if claude_request.stop_sequences:
        openai_request["stop"] = claude_request.stop_sequences
    if claude_request.top_p is not None:
        openai_request["top_p"] = claude_request.top_p

    # Convert tools — handles both standard and schema-less (computer use) tools
    if allow_tools and claude_request.tools:
        # First pass: detect and convert any schema-less Anthropic tools
        cu_converted, cu_system_supplement, has_computer_use = convert_schema_less_tools(
            claude_request.tools
        )

        # If there are computer-use tools, inject the environment description
        # into the system prompt so the (non-Claude) model knows about the display.
        if cu_system_supplement:
            # Prepend to existing system message or add a new one
            if openai_messages and openai_messages[0].get("role") == Constants.ROLE_SYSTEM:
                openai_messages[0]["content"] += "\n" + cu_system_supplement
            else:
                openai_messages.insert(
                    0,
                    {
                        "role": Constants.ROLE_SYSTEM,
                        "content": cu_system_supplement.strip(),
                    },
                )

        # If a web-search tool is offered and server-side search is configured,
        # nudge the model to call it on its own turn — batched search tool calls
        # can't be executed server-side (the backend turn can't be completed
        # without results for the sibling client tools).
        if config.tavily_api_key and any(is_search_tool(t) for t in claude_request.tools):
            if openai_messages and openai_messages[0].get("role") == Constants.ROLE_SYSTEM:
                openai_messages[0]["content"] += "\n\n" + SEARCH_TOOL_SYSTEM_SUPPLEMENT
            else:
                openai_messages.insert(
                    0,
                    {"role": Constants.ROLE_SYSTEM, "content": SEARCH_TOOL_SYSTEM_SUPPLEMENT},
                )

        openai_tools = []
        for idx, tool in enumerate(claude_request.tools):
            if not tool.name or not tool.name.strip():
                continue

            if cu_converted[idx] is not None:
                # Schema-less tool already converted to function format —
                # canonicalize its parameters so the wire bytes are stable
                # across requests (prefix-cache discipline).
                cu_tool = cu_converted[idx]
                cu_params = cu_tool.get(Constants.TOOL_FUNCTION, {}).get("parameters")
                if isinstance(cu_params, dict):
                    cu_tool[Constants.TOOL_FUNCTION]["parameters"] = _canonicalize_schema(
                        cu_params
                    )
                openai_tools.append(cu_tool)
            else:
                # Standard function tool. We canonicalize the parameters
                # schema (key-sort recursively) so that minor reordering
                # upstream cannot break the Nebius/vLLM prefix cache.
                # NOTE: We deliberately do NOT sort the outer tools list —
                # tool order is observable by the model.
                # Search tools (web_search/WebSearch) are executed by the proxy
                # (server_tools); force a real {query} schema so the backend emits
                # a clean call instead of an empty / control-token blob.
                if is_search_tool(tool):
                    params = SEARCH_TOOL_PARAMETERS
                else:
                    params = tool.input_schema or {"type": "object", "properties": {}}
                openai_tools.append(
                    {
                        "type": Constants.TOOL_FUNCTION,
                        Constants.TOOL_FUNCTION: {
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": _canonicalize_schema(params),
                        },
                    }
                )
        if openai_tools:
            openai_request["tools"] = openai_tools

    # Convert tool choice only when tools are present
    if allow_tools and claude_request.tool_choice and openai_request.get("tools"):
        choice_type = claude_request.tool_choice.get("type")
        if choice_type == "auto":
            openai_request["tool_choice"] = "auto"
        elif choice_type == "any":
            # Claude "any" = forced tool use → OpenAI "required"
            openai_request["tool_choice"] = "required"
        elif choice_type == "tool" and "name" in claude_request.tool_choice:
            openai_request["tool_choice"] = {
                "type": Constants.TOOL_FUNCTION,
                Constants.TOOL_FUNCTION: {"name": claude_request.tool_choice["name"]},
            }
        else:
            openai_request["tool_choice"] = "auto"

    # Vision endpoints commonly reject tool use; force no tools for image requests
    if has_image:
        openai_request.pop("tools", None)
        openai_request["tool_choice"] = "none"

    # --- Prefix-cache fingerprint (Tier-1) ---
    # Hash only the cacheable prefix: system message + tools. If this digest
    # stays stable across calls in the same Claude Code session, the
    # upstream KV cache is being reused.
    system_msg_for_fp: Optional[Dict[str, Any]] = None
    if openai_messages and openai_messages[0].get("role") == Constants.ROLE_SYSTEM:
        system_msg_for_fp = openai_messages[0]
    fingerprint = _compute_prefix_fingerprint(system_msg_for_fp, openai_request.get("tools"))
    logger.info(f"prefix_cache_fingerprint={fingerprint} model={openai_model}")

    return openai_request


def convert_claude_user_message(msg: ClaudeMessage, *, allow_images: bool) -> Dict[str, Any]:
    """Convert Claude user message to OpenAI format."""
    if msg.content is None:
        return {"role": Constants.ROLE_USER, "content": ""}

    if isinstance(msg.content, str):
        return {"role": Constants.ROLE_USER, "content": msg.content}

    # Handle multimodal content
    openai_content = []
    text_blocks = []
    image_blocks = []
    has_image = False
    for block in msg.content:
        # Normalize block access
        if isinstance(block, dict):
            block_type = block.get("type")
        else:
            block_type = getattr(block, "type", None)

        # Text blocks
        if block_type == Constants.CONTENT_TEXT:
            text_value = (
                block.get("text") if isinstance(block, dict) else getattr(block, "text", "")
            )
            text_blocks.append(text_value or "")

        # Base64 image blocks (Claude style)
        elif block_type == Constants.CONTENT_IMAGE and allow_images:
            source = (
                block.get("source") if isinstance(block, dict) else getattr(block, "source", {})
            )
            if (
                isinstance(source, dict)
                and source.get("type") == "base64"
                and "media_type" in source
                and "data" in source
            ):
                has_image = True
                image_blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{source['media_type']};base64,{source['data']}"
                        },
                    }
                )

        # Pre-encoded image_url blocks (OpenAI style) - pass through
        elif block_type == "image_url" and allow_images:
            image_url_payload = (
                block.get("image_url")
                if isinstance(block, dict)
                else getattr(block, "image_url", None)
            )
            if image_url_payload:
                has_image = True
                image_blocks.append({"type": "image_url", "image_url": image_url_payload})

    # Always strip/trim when an image is present to protect the vision model context,
    # regardless of the STRIP_IMAGE_CONTEXT flag. This keeps image hops lightweight.
    if has_image:
        text_to_keep = ""
        for text in reversed(text_blocks):
            stripped = text.strip()
            if not stripped:
                continue
            if stripped.startswith("<system-reminder>"):
                continue
            if stripped.lower().startswith("[image:"):
                continue
            text_to_keep = text
            break

        MAX_VISION_TEXT_CHARS = 1500
        if text_to_keep and len(text_to_keep) > MAX_VISION_TEXT_CHARS:
            text_to_keep = text_to_keep[-MAX_VISION_TEXT_CHARS:]

        if text_to_keep:
            openai_content = [{"type": "text", "text": text_to_keep}] + image_blocks
        else:
            openai_content = image_blocks
    else:
        for text in text_blocks:
            openai_content.append({"type": "text", "text": text})
        openai_content.extend(image_blocks)

    if len(openai_content) == 1 and openai_content[0]["type"] == "text":
        return {"role": Constants.ROLE_USER, "content": openai_content[0]["text"]}
    else:
        return {"role": Constants.ROLE_USER, "content": openai_content}


def convert_claude_assistant_message(msg: ClaudeMessage, *, allow_tools: bool) -> Dict[str, Any]:
    """Convert Claude assistant message to OpenAI format."""
    text_parts = []
    tool_calls = []

    if msg.content is None:
        return {"role": Constants.ROLE_ASSISTANT, "content": None}

    if isinstance(msg.content, str):
        return {"role": Constants.ROLE_ASSISTANT, "content": msg.content}

    for block in msg.content:
        btype = getattr(block, "type", None)
        if btype in ("thinking", "redacted_thinking"):
            # Extended-thinking blocks echoed back by the client. OpenAI-compatible
            # backends cannot ingest them, so we deliberately drop them here. They
            # are still accepted at the model layer so the request does not 422.
            continue
        if block.type == Constants.CONTENT_TEXT:
            text_parts.append(block.text)
        elif allow_tools and block.type == Constants.CONTENT_TOOL_USE:
            tool_calls.append(
                {
                    "id": block.id,
                    "type": Constants.TOOL_FUNCTION,
                    Constants.TOOL_FUNCTION: {
                        "name": block.name,
                        "arguments": json.dumps(block.input, ensure_ascii=False),
                    },
                }
            )

    openai_message = {"role": Constants.ROLE_ASSISTANT}

    # Set content
    if text_parts:
        openai_message["content"] = "".join(text_parts)
    else:
        openai_message["content"] = None

    # Set tool calls
    if tool_calls:
        openai_message["tool_calls"] = tool_calls

    return openai_message


def convert_claude_tool_results(msg: ClaudeMessage) -> List[Dict[str, Any]]:
    """Convert Claude tool results to OpenAI format."""
    tool_messages = []

    if isinstance(msg.content, list):
        for block in msg.content:
            if block.type == Constants.CONTENT_TOOL_RESULT:
                content = parse_tool_result_content(block.content)
                tool_messages.append(
                    {
                        "role": Constants.ROLE_TOOL,
                        "tool_call_id": block.tool_use_id,
                        "content": content,
                    }
                )

    return tool_messages


def parse_tool_result_content(content):
    """Parse and normalize tool result content into a string format."""
    if content is None:
        return "No content provided"

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        result_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == Constants.CONTENT_TEXT:
                result_parts.append(item.get("text", ""))
            elif isinstance(item, str):
                result_parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    result_parts.append(item.get("text", ""))
                else:
                    try:
                        result_parts.append(json.dumps(item, ensure_ascii=False))
                    except:
                        result_parts.append(str(item))
        return "\n".join(result_parts).strip()

    if isinstance(content, dict):
        if content.get("type") == Constants.CONTENT_TEXT:
            return content.get("text", "")
        try:
            return json.dumps(content, ensure_ascii=False)
        except:
            return str(content)

    try:
        return str(content)
    except:
        return "Unparseable content"
