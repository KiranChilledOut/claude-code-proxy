import asyncio
import json
import logging
import re
import time
import traceback
import uuid
from typing import Optional

from fastapi import HTTPException, Request

from src.conversion.request_converter import _count_tokens_text
from src.core.config import config
from src.core.constants import Constants
from src.models.claude import ClaudeMessagesRequest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool-call JSON repair (Tier-1)
# ---------------------------------------------------------------------------
# Open models often emit tool-call arguments that are *almost* JSON but
# trip strict parsers: trailing commas, single quotes, control characters
# inside strings. We attempt a small set of conservative repairs before
# giving up. We deliberately do NOT pull in a heavyweight JSON5 parser —
# that would be a behavior change risk for existing Nebius users.

_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _try_repair_json(raw: str) -> tuple:
    """Try to coerce a near-JSON string into valid JSON.

    Returns (parsed_obj_or_None, repaired_string). If parsed_obj is None,
    the string could not be repaired into valid JSON; callers wrap the
    raw text in `{"raw_arguments": ...}` so the model can re-prompt on
    the next turn (Claude Code handles this naturally).
    """
    if not raw or not raw.strip():
        return {}, "{}"

    # Fast path: already valid JSON.
    try:
        return json.loads(raw), raw
    except json.JSONDecodeError:
        pass

    # Repair pass 1: strip trailing commas before } or ]
    fixed = _TRAILING_COMMA_RE.sub(r"\1", raw)
    if fixed != raw:
        try:
            return json.loads(fixed), fixed
        except json.JSONDecodeError:
            pass

    # Repair pass 2: escape literal newlines/tabs inside string values.
    # We do this only if the un-escaped versions caused the parse to fail —
    # naive escape would corrupt valid JSON. Heuristic: try replacing only
    # raw newlines with \n and re-parse.
    candidate = fixed.replace("\r\n", "\n").replace("\n", "\\n").replace("\t", "\\t")
    if candidate != fixed:
        try:
            return json.loads(candidate), candidate
        except json.JSONDecodeError:
            pass

    return None, raw


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sse(event: str, data: dict) -> str:
    """Format a single SSE frame."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _map_finish_reason(finish_reason: Optional[str]) -> str:
    """Map OpenAI finish reason to Claude stop reason constants.

    Translates 'stop', 'length', 'tool_calls', 'function_call' to their
    Claude equivalents (end_turn, max_tokens, tool_use).
    """
    return {
        "stop": Constants.STOP_END_TURN,
        "length": Constants.STOP_MAX_TOKENS,
        "tool_calls": Constants.STOP_TOOL_USE,
        "function_call": Constants.STOP_TOOL_USE,
        # Provider safety stop -> Claude "refusal" so the CLI does not treat a
        # blocked completion as a normal end_turn.
        "content_filter": Constants.STOP_REFUSAL,
    }.get(finish_reason or "stop", Constants.STOP_END_TURN)


def _extract_usage(usage_raw: Optional[dict]) -> dict:
    """Build a Claude-style usage dict from OpenAI usage data.

    Covers prompt_tokens, completion_tokens, cached_tokens and also
    returns cache_creation_input_tokens (Feature 5).
    """
    if not usage_raw:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }

    cache_read = 0
    cache_creation = 0
    prompt_details = usage_raw.get("prompt_tokens_details") or {}
    if prompt_details:
        cache_read = prompt_details.get("cached_tokens", 0) or 0
    # Some providers report cache_creation separately
    completion_details = usage_raw.get("completion_tokens_details") or {}
    # Approximate cache_creation as total prompt minus cached portion when the
    # provider doesn't expose it explicitly — this gives Claude Code a usable
    # number for cost display without breaking anything.
    prompt_tokens = usage_raw.get("prompt_tokens", 0) or 0
    cache_creation = max(prompt_tokens - cache_read, 0) if cache_read else 0

    return {
        "input_tokens": prompt_tokens,
        "output_tokens": usage_raw.get("completion_tokens", 0) or 0,
        "cache_creation_input_tokens": cache_creation,
        "cache_read_input_tokens": cache_read,
    }


# ---------------------------------------------------------------------------
# Thinking-tag parser  (Feature 1)
# ---------------------------------------------------------------------------

_THINK_OPEN = re.compile(r"<think>", re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)

# Regex for XML-style arg encoding: <arg_key>name</arg_key><arg_value>value</arg_value>
_XML_ARG_PATTERN = re.compile(
    r"<arg_key>\s*(\w+)\s*</arg_key>\s*<arg_value>(.*?)</arg_value>",
    re.DOTALL,
)

# Broader XML pattern: also matches <tool_call> and mismatched tags
_XML_BROAD_PATTERN = re.compile(
    r"(?:<tool_call>|<arg_key>)\s*(\w+)\s*(?:</arg_key>|</tool_call>)\s*<arg_value>(.*?)</arg_value>",
    re.DOTALL,
)


# Kimi-K2 native tool-call control tokens, e.g.:
#   <|tool_calls_section_begin|> <|tool_call_begin|> functions.web_search:0
#   <|tool_call_argument_begin|> {"query": "..."} <|tool_call_end|> <|tool_calls_section_end|>
# These leak into `arguments` when a tool is forwarded to Kimi without a real
# parameter schema (e.g. Anthropic server tools like web_search). Extract the
# inner JSON and the real tool name.
_KIMI_ARG_PATTERN = re.compile(
    r"<\|tool_call_argument_begin\|>\s*(\{.*?\})\s*<\|tool_call_end\|>", re.DOTALL
)
_KIMI_NAME_PATTERN = re.compile(r"functions\.([A-Za-z0-9_.\-]+):\d+")
_KIMI_TOKEN_PATTERN = re.compile(
    r"<\|tool_calls?_section_(?:begin|end)\|>|<\|tool_call_(?:begin|end)\|>|"
    r"<\|tool_call_argument_begin\|>|functions\.[A-Za-z0-9_.\-]+:\d+"
)


def _clean_tool_name(raw_name: str) -> str:
    """Extract clean tool name by stripping any XML tags, trailing parens, etc."""
    # Strip from first XML-like tag onwards
    cut = re.split(r"<(?:arg_key|tool_call|arg_value|/)", raw_name, maxsplit=1)[0].strip()
    # Strip trailing open paren (model sometimes appends it)
    cut = cut.rstrip("(").strip()
    return cut if cut else raw_name


def _sanitize_tool_arguments(name: str, arguments_str: str) -> tuple:
    """Sanitize malformed tool call arguments from non-standard models.

    Handles known GLM-4.5 patterns:
    1. XML-style: <arg_key>command</arg_key><arg_value>ls -la</arg_value>
    2. XML in function name: Bash<tool_call>command</arg_key><arg_value>...
    3. Hybrid JSON+XML keys: {"command=value</arg_value><arg_key>description":"desc"}
    4. Args embedded in function name: bash(command="ls -la")
    5. Args in parameter names with spaces: {"command ls -la": ""}
    6. Raw strings that aren't valid JSON

    Returns (clean_name, clean_arguments_json_str).
    """
    raw_name = name or ""
    raw_args = arguments_str or ""

    # ── Step 0a: Kimi-K2 native control-token format ──
    if "<|tool_call" in raw_args or "<|tool_call" in raw_name:
        combined = raw_args if "<|tool_call" in raw_args else raw_name
        name_m = _KIMI_NAME_PATTERN.search(combined)
        kimi_name = _clean_tool_name(raw_name)
        if name_m and not re.fullmatch(r"[A-Za-z0-9_]+", kimi_name or ""):
            kimi_name = name_m.group(1)
        arg_m = _KIMI_ARG_PATTERN.search(combined)
        if arg_m:
            inner = arg_m.group(1).strip()
            try:
                json.loads(inner)
                logger.info(f"[SANITIZE] Kimi control tokens stripped: name={kimi_name}")
                return kimi_name, inner
            except json.JSONDecodeError:
                pass
        # Fallback: strip all Kimi tokens and recover the first JSON object.
        stripped = _KIMI_TOKEN_PATTERN.sub(" ", combined).strip()
        jm = re.search(r"\{.*\}", stripped, re.DOTALL)
        if jm:
            try:
                json.loads(jm.group(0))
                return kimi_name, jm.group(0)
            except json.JSONDecodeError:
                pass

    # ── Step 0: Try XML extraction on ALL sources (args, name, combined) ──
    # Check args string, name, and combined for XML arg patterns
    for source_label, source_text in [
        ("args", raw_args),
        ("name", raw_name),
        ("combined", raw_name + raw_args),
    ]:
        for pattern in [_XML_ARG_PATTERN, _XML_BROAD_PATTERN]:
            xml_matches = pattern.findall(source_text)
            if xml_matches:
                parsed = {}
                for key, val in xml_matches:
                    parsed[key.strip()] = val.strip()
                if parsed:
                    clean_name = _clean_tool_name(raw_name)
                    logger.info(
                        f"[SANITIZE] XML args from {source_label}: name={clean_name} args={parsed}"
                    )
                    return clean_name, json.dumps(parsed)

    # ── Step 1: Clean up the tool name ──
    clean_name = _clean_tool_name(raw_name)

    # Default args
    clean_args = raw_args if raw_args.strip() else "{}"

    # ── Step 2: Args in function name via parentheses: name({...}) or name(k="v") ──
    paren_idx = clean_name.find("(")
    if paren_idx > 0 and clean_name.endswith(")"):
        embedded_args = clean_name[paren_idx + 1 : -1].strip()
        clean_name = clean_name[:paren_idx].strip()
        if embedded_args and clean_args.strip() in ("", "{}"):
            try:
                json.loads(embedded_args)
                clean_args = embedded_args
            except json.JSONDecodeError:
                pairs = {}
                for match in re.finditer(r'(\w+)\s*=\s*["\']([^"\']*)["\']', embedded_args):
                    pairs[match.group(1)] = match.group(2)
                if pairs:
                    clean_args = json.dumps(pairs)
            logger.info(f"[SANITIZE] Args from name parens: name={clean_name} args={clean_args}")
            return clean_name, clean_args

    # ── Step 3: Parse JSON and fix mangled keys ──
    # Handles: {"command=value</arg_value><arg_key>description": "desc"}
    #          {"command ls -la": ""}
    #          {"command=\"value\"</arg_value><arg_key>description": "desc"}
    try:
        parsed = json.loads(clean_args)
        if isinstance(parsed, dict):
            needs_fix = any(" " in k or "<" in k or ">" in k or "=" in k for k in parsed)

            if needs_fix:
                fixed = {}
                for key, val in parsed.items():
                    # First, split key at XML boundaries to extract multiple params
                    # e.g. "command=value</arg_value><arg_key>description" → two params
                    key_parts = re.split(r"</arg_value>\s*<arg_key>", key)

                    for kp in key_parts:
                        # Strip remaining XML tags
                        clean_kp = re.sub(r"</?[\w_]+>", "", kp).strip()
                        clean_kp = clean_kp.strip('"').strip("'")

                        if not clean_kp:
                            continue

                        # Try key=value pattern
                        eq_match = re.match(r"^(\w+)\s*=\s*(.+)$", clean_kp, re.DOTALL)
                        if eq_match:
                            pname = eq_match.group(1)
                            pval = eq_match.group(2).strip().strip('"').strip("'")
                            fixed[pname] = pval
                            continue

                        # Try "key value" pattern (space-separated)
                        parts = clean_kp.split(None, 1)
                        if len(parts) == 2 and parts[0].isidentifier():
                            fixed[parts[0]] = parts[1]
                            continue

                        # Simple identifier — this is the KEY, use the JSON value
                        if re.match(r"^\w+$", clean_kp):
                            # Only use the original val for the LAST key fragment
                            if kp == key_parts[-1]:
                                fixed[clean_kp] = val
                            continue

                if fixed:
                    logger.info(f"[SANITIZE] Fixed mangled keys: {fixed}")
                    return clean_name, json.dumps(fixed)

            # JSON is valid and keys look normal — pass through
            return clean_name, clean_args
    except (json.JSONDecodeError, TypeError):
        pass

    # ── Step 4: Raw string (not JSON at all) ──
    if clean_args.strip() and clean_args.strip()[0] not in ("{", "[", '"'):
        raw_val = clean_args.strip()
        lower_name = clean_name.lower()
        if lower_name == "bash":
            clean_args = json.dumps({"command": raw_val})
            logger.info(f"[SANITIZE] Wrapped raw bash arg: {clean_args}")
        elif lower_name == "computer":
            clean_args = json.dumps({"action": raw_val})
            logger.info(f"[SANITIZE] Wrapped raw computer arg: {clean_args}")

    return clean_name, clean_args


def _finalize_tool_args(name: str, raw_args: str) -> tuple:
    """Sanitize + JSON-validate tool arguments for the final emit.

    Returns (clean_name, args_json_str, parsed_dict_or_None).

    Pipeline: run the existing sanitizer (XML, embedded args, mangled
    keys, raw bash strings) and then a small JSON-repair pass for
    near-JSON survivors (trailing commas, raw newlines). If the bytes
    still don't parse, parsed_dict is None — callers wrap in
    `{"raw_arguments": ...}` exactly as the proxy has done historically.
    Claude Code's natural next-turn re-prompt handles those cases fine.
    """
    clean_name, clean_args = _sanitize_tool_arguments(name, raw_args)
    parsed, repaired = _try_repair_json(clean_args)
    return clean_name, repaired, parsed


def _split_thinking_and_text(text: str):
    """Split text containing <think>…</think> into thinking and text parts.

    Returns a list of tuples: [("thinking", str), ("text", str), …]
    Handles multiple or nested think blocks and leftover text.
    """
    parts = []
    pos = 0
    while pos < len(text):
        m_open = _THINK_OPEN.search(text, pos)
        if not m_open:
            remainder = text[pos:]
            if remainder:
                parts.append(("text", remainder))
            break
        # Text before <think>
        before = text[pos : m_open.start()]
        if before:
            parts.append(("text", before))
        # Find closing tag
        m_close = _THINK_CLOSE.search(text, m_open.end())
        if m_close:
            thinking_content = text[m_open.end() : m_close.start()]
            if thinking_content:
                parts.append(("thinking", thinking_content))
            pos = m_close.end()
        else:
            # Unclosed think tag — treat rest as thinking
            thinking_content = text[m_open.end() :]
            if thinking_content:
                parts.append(("thinking", thinking_content))
            break
    return parts


def _strip_think_tags(text: str) -> str:
    """Drop <think>…</think> spans (including an unclosed trailing one),
    returning only the visible text. Used when thinking must not be surfaced."""
    return "".join(v for kind, v in _split_thinking_and_text(text) if kind == "text")


def _should_surface_thinking(original_request) -> bool:
    """Decide whether the backend's thinking *text* is surfaced to the client.

    Honors the operator THINKING_DISPLAY_OVERRIDE, then the request's `display`
    / mode default (adaptive -> omitted, matching Opus 4.7/4.8). When thinking
    is disabled, never surface.
    """
    thinking = getattr(original_request, "thinking", None)
    if not (thinking and thinking.is_enabled()):
        return False
    override = getattr(config, "thinking_display_override", "")
    if override == "summarized":
        return True
    if override == "omitted":
        return False
    return thinking.surfaces_text()


# ---------------------------------------------------------------------------
# Non-streaming response converter
# ---------------------------------------------------------------------------


def convert_openai_to_claude_response(
    openai_response: dict, original_request: ClaudeMessagesRequest
) -> dict:
    """Convert OpenAI response to Claude format."""

    choices = openai_response.get("choices", [])
    if not choices:
        raise HTTPException(status_code=500, detail="No choices in OpenAI response")

    choice = choices[0]
    message = choice.get("message", {})

    content_blocks = []

    surface_thinking = _should_surface_thinking(original_request)

    # --- Provider reasoning channel (DeepSeek / MiniMax / Qwen / GLM-thinking) ---
    # Some OpenAI-compatible backends return chain-of-thought in a separate
    # `reasoning_content` (or `reasoning`) field rather than inline <think> tags.
    # Only surface it when the client's thinking `display` calls for it.
    reasoning_text = message.get("reasoning_content") or message.get("reasoning")
    if surface_thinking and isinstance(reasoning_text, str) and reasoning_text.strip():
        content_blocks.append({"type": "thinking", "thinking": reasoning_text})

    # --- Feature 1: handle <think> tags in text content ---
    text_content = message.get("content")
    if text_content is not None:
        has_think = "<think>" in text_content.lower()
        if surface_thinking and has_think:
            for kind, value in _split_thinking_and_text(text_content):
                if kind == "thinking":
                    content_blocks.append(
                        {
                            "type": "thinking",
                            "thinking": value,
                        }
                    )
                else:
                    content_blocks.append(
                        {
                            "type": Constants.CONTENT_TEXT,
                            "text": value,
                        }
                    )
        elif has_think:
            # Thinking not surfaced (disabled, or display=omitted on adaptive):
            # strip the <think> span so reasoning never leaks as visible text.
            content_blocks.append(
                {
                    "type": Constants.CONTENT_TEXT,
                    "text": _strip_think_tags(text_content),
                }
            )
        else:
            content_blocks.append(
                {
                    "type": Constants.CONTENT_TEXT,
                    "text": text_content,
                }
            )

    # Tool calls
    tool_calls = message.get("tool_calls", []) or []
    seen_signatures = set()  # (name, normalized_args) — used for dedup
    for tool_call in tool_calls:
        if tool_call.get("type") == Constants.TOOL_FUNCTION:
            function_data = tool_call.get(Constants.TOOL_FUNCTION, {})
            raw_name = function_data.get("name", "")
            arguments_str = function_data.get("arguments", "{}")

            # --- Sanitize + JSON-repair tool-call arguments ---
            actual_name, arguments_str, parsed = _finalize_tool_args(raw_name, arguments_str)

            if parsed is not None:
                arguments = parsed
            else:
                # repair mode, unparseable: keep historical fallback shape
                arguments = {"raw_arguments": arguments_str}

            # Dedup: same (name, args) emitted twice in the same turn is a
            # known open-model glitch (GLM-4.5 has been seen doing this).
            try:
                signature = (
                    actual_name,
                    json.dumps(arguments, sort_keys=True, ensure_ascii=False),
                )
            except (TypeError, ValueError):
                signature = (actual_name, str(arguments))
            if signature in seen_signatures:
                logger.info(
                    f"[DEDUP] Dropped duplicate tool_use {actual_name} in same turn"
                )
                continue
            seen_signatures.add(signature)

            content_blocks.append(
                {
                    "type": Constants.CONTENT_TOOL_USE,
                    "id": tool_call.get("id", f"tool_{uuid.uuid4()}"),
                    "name": actual_name,
                    "input": arguments,
                }
            )

    # Ensure at least one content block
    if not content_blocks:
        content_blocks.append({"type": Constants.CONTENT_TEXT, "text": ""})

    stop_reason = _map_finish_reason(choice.get("finish_reason"))

    # --- Feature 5: full usage with cache fields ---
    usage = _extract_usage(openai_response.get("usage"))

    return {
        "id": openai_response.get("id", f"msg_{uuid.uuid4()}"),
        "type": "message",
        "role": Constants.ROLE_ASSISTANT,
        "model": original_request.model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage,
    }


# ---------------------------------------------------------------------------
# Unified streaming converter  (Fix 3: single implementation)
# ---------------------------------------------------------------------------


async def convert_openai_streaming_to_claude_with_cancellation(
    openai_stream,
    original_request: ClaudeMessagesRequest,
    logger,
    http_request: Optional[Request] = None,
    openai_client=None,
    request_id: Optional[str] = None,
    observability_context: Optional[dict] = None,
):
    """Convert OpenAI streaming response to Claude streaming format.

    This is the single, unified streaming converter that handles both
    cancellation-aware and simple streaming (Fix 3).
    When http_request / openai_client / request_id are None the cancellation
    logic is simply skipped, so this replaces the old non-cancellation variant.
    """

    message_id = f"msg_{uuid.uuid4().hex[:24]}"

    # --- Feature 1: thinking state machine ---
    # Whether to surface the backend's thinking *text* (honors display/override).
    surface_thinking = _should_surface_thinking(original_request)
    # States: "idle", "in_thinking", "in_text"
    thinking_state = "idle"
    text_buffer = ""  # Buffer to detect <think> at chunk boundaries
    thinking_block_index = None  # index of the current thinking content block
    text_block_started = False
    text_emitted_any = False  # Track whether any real text was emitted (Fix 4)
    reasoning_block_open = False  # provider reasoning_content -> thinking block

    # We'll track the current block index dynamically
    current_block_index = -1  # will be incremented as blocks are started

    def _next_index():
        nonlocal current_block_index
        current_block_index += 1
        return current_block_index

    # --- Send message_start ---
    yield _sse(
        Constants.EVENT_MESSAGE_START,
        {
            "type": Constants.EVENT_MESSAGE_START,
            "message": {
                "id": message_id,
                "type": "message",
                "role": Constants.ROLE_ASSISTANT,
                "model": original_request.model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    )

    yield _sse(Constants.EVENT_PING, {"type": Constants.EVENT_PING})

    # --- Feature 3: heartbeat state ---
    HEARTBEAT_INTERVAL = 15  # seconds
    last_data_time = time.monotonic()

    # Streaming state
    tool_block_counter = 0
    current_tool_calls = {}
    final_stop_reason = Constants.STOP_END_TURN
    usage_data = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    estimated_output_tokens = 0
    observed_tool_calls = []
    started_blocks = []  # track indices of blocks we've started (for Fix 4)
    stopped_blocks = set()  # track indices already stopped (avoid double-stop)

    if observability_context is not None:
        observability_context.setdefault("usage", usage_data)
        observability_context.setdefault("tool_calls", observed_tool_calls)
        observability_context.setdefault("status", "success")

    def _start_text_block():
        """Lazily start the text content block when we first have text."""
        nonlocal text_block_started
        if not text_block_started:
            idx = _next_index()
            text_block_started = True
            started_blocks.append(("text", idx))
            return _sse(
                Constants.EVENT_CONTENT_BLOCK_START,
                {
                    "type": Constants.EVENT_CONTENT_BLOCK_START,
                    "index": idx,
                    "content_block": {"type": Constants.CONTENT_TEXT, "text": ""},
                },
            )
        return ""

    def _start_thinking_block():
        """Start a thinking content block."""
        nonlocal thinking_block_index
        idx = _next_index()
        thinking_block_index = idx
        started_blocks.append(("thinking", idx))
        return _sse(
            Constants.EVENT_CONTENT_BLOCK_START,
            {
                "type": Constants.EVENT_CONTENT_BLOCK_START,
                "index": idx,
                "content_block": {"type": "thinking", "thinking": ""},
            },
        )

    def _get_text_block_index():
        """Get the most recent text block index."""
        for kind, idx in reversed(started_blocks):
            if kind == "text":
                return idx
        return 0

    def _get_thinking_block_index():
        return thinking_block_index

    async def _emit_text_dropping_think(fragment: str):
        """Stream visible text while silently dropping <think>…</think> spans.

        Used when thinking must not be surfaced (thinking disabled, or
        display=omitted on adaptive mode). Buffers across chunks so a tag split
        between chunks is still caught; the think body is discarded, never sent
        as a thinking_delta.
        """
        nonlocal thinking_state, text_buffer, text_emitted_any
        text_buffer += fragment
        while text_buffer:
            if thinking_state == "in_thinking":
                m = _THINK_CLOSE.search(text_buffer)
                if m:
                    thinking_state = "in_text"
                    text_buffer = text_buffer[m.end():]
                    continue
                # No close yet: drop consumed think text, keep a short tail in
                # case "</think>" is split across the chunk boundary.
                safe = len(text_buffer) - 8  # len("</think>")
                if safe > 0:
                    text_buffer = text_buffer[safe:]
                break
            else:
                m = _THINK_OPEN.search(text_buffer)
                if m:
                    before = text_buffer[: m.start()]
                    if before:
                        events = _start_text_block()
                        if events:
                            yield events
                        text_emitted_any = True
                        yield _sse(
                            Constants.EVENT_CONTENT_BLOCK_DELTA,
                            {
                                "type": Constants.EVENT_CONTENT_BLOCK_DELTA,
                                "index": _get_text_block_index(),
                                "delta": {"type": Constants.DELTA_TEXT, "text": before},
                            },
                        )
                    thinking_state = "in_thinking"
                    text_buffer = text_buffer[m.end():]
                    continue
                safe_emit_len = len(text_buffer) - 6  # len("<think")
                if safe_emit_len > 0 and "<" in text_buffer[safe_emit_len:]:
                    to_emit = text_buffer[:safe_emit_len]
                    text_buffer = text_buffer[safe_emit_len:]
                else:
                    to_emit = text_buffer
                    text_buffer = ""
                if to_emit:
                    events = _start_text_block()
                    if events:
                        yield events
                    text_emitted_any = True
                    yield _sse(
                        Constants.EVENT_CONTENT_BLOCK_DELTA,
                        {
                            "type": Constants.EVENT_CONTENT_BLOCK_DELTA,
                            "index": _get_text_block_index(),
                            "delta": {"type": Constants.DELTA_TEXT, "text": to_emit},
                        },
                    )
                break

    async def _process_text_fragment(fragment: str):
        """Process a text fragment, handling <think> tag detection.

        Yields SSE strings.
        """
        nonlocal thinking_state, text_buffer, text_emitted_any, text_block_started

        if not surface_thinking:
            # Thinking not surfaced — emit only visible text, strip <think> spans.
            async for ev in _emit_text_dropping_think(fragment):
                yield ev
            return

        # Buffer text to handle <think> tags that may span chunks
        text_buffer += fragment

        while text_buffer:
            if thinking_state == "idle" or thinking_state == "in_text":
                # Look for <think> opening
                m = _THINK_OPEN.search(text_buffer)
                if m:
                    # Emit text before the tag
                    before = text_buffer[: m.start()]
                    if before:
                        events = _start_text_block()
                        if events:
                            yield events
                        text_emitted_any = True
                        yield _sse(
                            Constants.EVENT_CONTENT_BLOCK_DELTA,
                            {
                                "type": Constants.EVENT_CONTENT_BLOCK_DELTA,
                                "index": _get_text_block_index(),
                                "delta": {"type": Constants.DELTA_TEXT, "text": before},
                            },
                        )
                    # Close text block if open, start thinking block
                    if text_block_started:
                        text_idx = _get_text_block_index()
                        yield _sse(
                            Constants.EVENT_CONTENT_BLOCK_STOP,
                            {
                                "type": Constants.EVENT_CONTENT_BLOCK_STOP,
                                "index": text_idx,
                            },
                        )
                        stopped_blocks.add(text_idx)
                    yield _start_thinking_block()
                    thinking_state = "in_thinking"
                    text_buffer = text_buffer[m.end() :]
                else:
                    # No <think> found. But the tag might be split across
                    # chunks, so hold back the last few chars if they could
                    # be a partial "<think>" prefix.
                    safe_emit_len = len(text_buffer) - 6  # len("<think") = 6
                    if safe_emit_len > 0 and "<" in text_buffer[safe_emit_len:]:
                        to_emit = text_buffer[:safe_emit_len]
                        text_buffer = text_buffer[safe_emit_len:]
                    else:
                        to_emit = text_buffer
                        text_buffer = ""

                    if to_emit:
                        events = _start_text_block()
                        if events:
                            yield events
                        text_emitted_any = True
                        yield _sse(
                            Constants.EVENT_CONTENT_BLOCK_DELTA,
                            {
                                "type": Constants.EVENT_CONTENT_BLOCK_DELTA,
                                "index": _get_text_block_index(),
                                "delta": {"type": Constants.DELTA_TEXT, "text": to_emit},
                            },
                        )
                    break  # wait for more data

            elif thinking_state == "in_thinking":
                m = _THINK_CLOSE.search(text_buffer)
                if m:
                    # Emit thinking content before the close tag
                    thinking_text = text_buffer[: m.start()]
                    if thinking_text:
                        yield _sse(
                            Constants.EVENT_CONTENT_BLOCK_DELTA,
                            {
                                "type": Constants.EVENT_CONTENT_BLOCK_DELTA,
                                "index": _get_thinking_block_index(),
                                "delta": {"type": "thinking_delta", "thinking": thinking_text},
                            },
                        )
                    # Stop thinking block
                    thinking_idx = _get_thinking_block_index()
                    yield _sse(
                        Constants.EVENT_CONTENT_BLOCK_STOP,
                        {
                            "type": Constants.EVENT_CONTENT_BLOCK_STOP,
                            "index": thinking_idx,
                        },
                    )
                    stopped_blocks.add(thinking_idx)
                    thinking_state = "in_text"
                    text_buffer = text_buffer[m.end() :]
                    # Reset so next text creates a fresh content block
                    text_block_started = False
                else:
                    # Still inside thinking — check for partial </think>
                    safe_len = len(text_buffer) - 8  # len("</think>") = 8
                    if safe_len > 0:
                        to_emit = text_buffer[:safe_len]
                        text_buffer = text_buffer[safe_len:]
                    else:
                        to_emit = ""
                    if to_emit:
                        yield _sse(
                            Constants.EVENT_CONTENT_BLOCK_DELTA,
                            {
                                "type": Constants.EVENT_CONTENT_BLOCK_DELTA,
                                "index": _get_thinking_block_index(),
                                "delta": {"type": "thinking_delta", "thinking": to_emit},
                            },
                        )
                    break  # wait for more data

    async def _process_reasoning_fragment(fragment: str):
        """Stream provider reasoning_content as a Claude thinking block.

        Only active when the model emits a separate reasoning channel; the
        default <think>-tag path is unaffected.
        """
        nonlocal reasoning_block_open
        if not surface_thinking or not fragment:
            return
        if not reasoning_block_open and thinking_block_index is None:
            yield _start_thinking_block()
            reasoning_block_open = True
        yield _sse(
            Constants.EVENT_CONTENT_BLOCK_DELTA,
            {
                "type": Constants.EVENT_CONTENT_BLOCK_DELTA,
                "index": _get_thinking_block_index(),
                "delta": {"type": "thinking_delta", "thinking": fragment},
            },
        )

    try:
        async for line in openai_stream:
            now = time.monotonic()

            # --- Cancellation check ---
            if http_request is not None:
                if await http_request.is_disconnected():
                    logger.info(f"Client disconnected, cancelling request {request_id}")
                    if openai_client and request_id:
                        openai_client.cancel_request(request_id)
                    if observability_context is not None:
                        observability_context["status"] = "cancelled"
                        observability_context["error_type"] = "client_disconnected"
                        observability_context["error_message"] = "Client disconnected"
                    break

            # --- Feature 3: heartbeat ping if no data for a while ---
            if now - last_data_time > HEARTBEAT_INTERVAL:
                yield _sse(Constants.EVENT_PING, {"type": Constants.EVENT_PING})
                last_data_time = now

            if not line.strip():
                continue
            if not line.startswith("data: "):
                continue

            last_data_time = now
            chunk_data = line[6:]
            if chunk_data.strip() == "[DONE]":
                break

            try:
                chunk = json.loads(chunk_data)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse chunk: {chunk_data}, error: {e}")
                continue

            # --- Feature 5: extract usage from chunk ---
            raw_usage = chunk.get("usage")
            if raw_usage:
                usage_data = _extract_usage(raw_usage)
                if observability_context is not None:
                    observability_context["usage"] = usage_data

            choices = chunk.get("choices", [])
            if not choices:
                continue

            choice = choices[0]
            delta = choice.get("delta", {})
            finish_reason = choice.get("finish_reason")

            # Debug: log raw tool_calls from model
            if "tool_calls" in delta and delta["tool_calls"]:
                logger.info(
                    f"[PROXY DEBUG] Raw tool_calls from model: {json.dumps(delta['tool_calls'])}"
                )

            # --- Handle provider reasoning channel (separate from content) ---
            reasoning_fragment = delta.get("reasoning_content") or delta.get("reasoning")
            if reasoning_fragment:
                estimated_output_tokens += _count_tokens_text(reasoning_fragment)
                async for event in _process_reasoning_fragment(reasoning_fragment):
                    yield event

            # --- Handle text delta (with thinking support) ---
            if delta and "content" in delta and delta["content"] is not None:
                # Close an open reasoning thinking block before text begins.
                if reasoning_block_open:
                    _r_idx = _get_thinking_block_index()
                    yield _sse(
                        Constants.EVENT_CONTENT_BLOCK_STOP,
                        {"type": Constants.EVENT_CONTENT_BLOCK_STOP, "index": _r_idx},
                    )
                    stopped_blocks.add(_r_idx)
                    reasoning_block_open = False
                    thinking_block_index = None
                    text_block_started = False
                estimated_output_tokens += _count_tokens_text(delta["content"])
                async for event in _process_text_fragment(delta["content"]):
                    yield event

            # --- Handle tool call deltas (Fix 1: incremental partial_json) ---
            if "tool_calls" in delta and delta["tool_calls"]:
                for tc_delta in delta["tool_calls"]:
                    tc_index = tc_delta.get("index", 0)

                    if tc_index not in current_tool_calls:
                        current_tool_calls[tc_index] = {
                            "id": None,
                            "name": None,
                            "args_buffer": "",
                            "claude_index": None,
                            "started": False,
                            "args_pending": False,
                        }

                    tool_call = current_tool_calls[tc_index]

                    if tc_delta.get("id"):
                        tool_call["id"] = tc_delta["id"]

                    function_data = tc_delta.get(Constants.TOOL_FUNCTION, {})
                    raw_name = function_data.get("name", "")

                    # --- Sanitize malformed function name / embedded args ---
                    if raw_name:
                        clean_name, extracted_args = _sanitize_tool_arguments(
                            raw_name, tool_call["args_buffer"] or ""
                        )
                        tool_call["name"] = clean_name
                        # Only update args_buffer if sanitizer found real args
                        # (not just the default "{}" from empty input)
                        if (
                            extracted_args
                            and extracted_args.strip() not in ("", "{}")
                            and extracted_args != tool_call["args_buffer"]
                        ):
                            tool_call["args_buffer"] = extracted_args
                            logger.info(
                                f"[PROXY] Sanitized tool call: name={clean_name} "
                                f"args={extracted_args[:200]}"
                            )

                    # Buffer arguments that arrive BEFORE block starts (same
                    # delta as name/id). Once started, buffering happens in
                    # the elif branch below.
                    if not tool_call["started"]:
                        if "arguments" in function_data and function_data["arguments"] is not None:
                            arg_val = function_data["arguments"]
                            if arg_val and arg_val.strip() not in ("", "{}"):
                                tool_call["args_buffer"] += arg_val

                    logger.debug(
                        f"Tool call delta: index={tc_index} id={tool_call['id']} "
                        f"name={tool_call['name']} started={tool_call['started']} "
                        f"args_buffer_len={len(tool_call['args_buffer'])} "
                        f"raw_function_data={function_data}"
                    )

                    # Start tool content block when we have id + name
                    if tool_call["id"] and tool_call["name"] and not tool_call["started"]:
                        # Make sure text block is closed before tool blocks
                        if text_block_started:
                            # Flush any remaining text buffer
                            if text_buffer:
                                async for event in _process_text_fragment(""):
                                    yield event

                        tool_block_counter += 1
                        idx = _next_index()
                        tool_call["claude_index"] = idx
                        tool_call["started"] = True
                        started_blocks.append(("tool", idx))

                        yield _sse(
                            Constants.EVENT_CONTENT_BLOCK_START,
                            {
                                "type": Constants.EVENT_CONTENT_BLOCK_START,
                                "index": idx,
                                "content_block": {
                                    "type": Constants.CONTENT_TOOL_USE,
                                    "id": tool_call["id"],
                                    "name": tool_call["name"],
                                    "input": {},
                                },
                            },
                        )
                        # Don't send args yet — buffer ALL args and send
                        # a single sanitized JSON at finish_reason to avoid
                        # Claude Code receiving broken partial concatenations.

                    # --- Buffer argument fragments (sent at finish_reason) ---
                    elif (
                        "arguments" in function_data
                        and tool_call["started"]
                        and function_data["arguments"] is not None
                    ):
                        fragment = function_data["arguments"]
                        tool_call["args_buffer"] += fragment
                        tool_call["args_pending"] = True

            # Handle finish reason
            if finish_reason:
                # Flush ALL buffered tool arguments as sanitized JSON.
                # Apply final-args resolution (sanitize → JSON repair) and
                # dedup duplicate (name, args) tool calls produced in the
                # same turn.
                seen_signatures = set()
                for tc_idx, tc_data in current_tool_calls.items():
                    if not tc_data["started"]:
                        continue

                    has_args = bool(tc_data["args_buffer"])
                    if has_args:
                        final_name, sanitized, parsed = _finalize_tool_args(
                            tc_data["name"], tc_data["args_buffer"]
                        )
                    else:
                        # No args streamed — preserve prior behavior of not
                        # emitting a redundant input_json_delta. The
                        # content_block_start already carried `"input": {}`.
                        final_name, sanitized, parsed = tc_data["name"], None, None

                    # Build a signature for dedup. If parsing succeeded use
                    # canonical form; otherwise use the raw sanitized string.
                    try:
                        sig = (
                            final_name,
                            json.dumps(parsed, sort_keys=True, ensure_ascii=False)
                            if parsed is not None
                            else (sanitized or ""),
                        )
                    except (TypeError, ValueError):
                        sig = (final_name, sanitized or "")

                    if sig in seen_signatures:
                        logger.info(
                            f"[DEDUP] Dropped duplicate streamed tool_use "
                            f"{final_name} (idx={tc_data['claude_index']})"
                        )
                        continue
                    seen_signatures.add(sig)

                    if has_args and sanitized is not None:
                        yield _sse(
                            Constants.EVENT_CONTENT_BLOCK_DELTA,
                            {
                                "type": Constants.EVENT_CONTENT_BLOCK_DELTA,
                                "index": tc_data["claude_index"],
                                "delta": {
                                    "type": Constants.DELTA_INPUT_JSON,
                                    "partial_json": sanitized,
                                },
                            },
                        )
                        logger.info(
                            f"[PROXY] Flushed sanitized args for {final_name}: "
                            f"{sanitized[:200]}"
                        )

                    estimated_output_tokens += _count_tokens_text(
                        f"{final_name} {sanitized or tc_data['args_buffer'] or '{}'}"
                    )
                    observed_tool_calls.append(
                        {
                            "tool_id": tc_data["id"],
                            "tool_name": final_name,
                            "arguments": sanitized or tc_data["args_buffer"] or "{}",
                            "status": "emitted",
                            "sanitized": bool(
                                sanitized
                                and sanitized != (tc_data["args_buffer"] or "{}")
                            ),
                        }
                    )
                final_stop_reason = _map_finish_reason(finish_reason)
                if observability_context is not None:
                    observability_context["stop_reason"] = final_stop_reason
                    observability_context["tool_calls"] = observed_tool_calls
                    observability_context["estimated_output_tokens"] = estimated_output_tokens
                break

    except HTTPException as e:
        if observability_context is not None:
            observability_context["status"] = "cancelled" if e.status_code == 499 else "error"
            observability_context["error_type"] = "HTTPException"
            observability_context["error_message"] = str(e.detail)
        if e.status_code == 499:
            logger.info(f"Request {request_id} was cancelled")
            yield _sse(
                "error",
                {
                    "type": "error",
                    "error": {"type": "cancelled", "message": "Request was cancelled by client"},
                },
            )
            return
        raise
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        logger.error(traceback.format_exc())
        if observability_context is not None:
            observability_context["status"] = "error"
            observability_context["error_type"] = type(e).__name__
            observability_context["error_message"] = str(e)
        yield _sse(
            "error",
            {
                "type": "error",
                "error": {"type": "api_error", "message": f"Streaming error: {str(e)}"},
            },
        )
        return

    # --- Flush remaining text buffer (thinking support) ---
    if text_buffer:
        if thinking_state == "in_thinking":
            if surface_thinking:
                yield _sse(
                    Constants.EVENT_CONTENT_BLOCK_DELTA,
                    {
                        "type": Constants.EVENT_CONTENT_BLOCK_DELTA,
                        "index": _get_thinking_block_index(),
                        "delta": {"type": "thinking_delta", "thinking": text_buffer},
                    },
                )
            # else: residual thinking text is dropped (not surfaced)
        else:
            events = _start_text_block()
            if events:
                yield events
            text_emitted_any = True
            yield _sse(
                Constants.EVENT_CONTENT_BLOCK_DELTA,
                {
                    "type": Constants.EVENT_CONTENT_BLOCK_DELTA,
                    "index": _get_text_block_index(),
                    "delta": {"type": Constants.DELTA_TEXT, "text": text_buffer},
                },
            )

    # --- Fix 4: Only emit content_block_stop for blocks we actually started ---
    # If no text was emitted and no blocks were started, emit a minimal text block
    # so Claude Code always gets at least one content block.
    if not started_blocks:
        idx = _next_index()
        started_blocks.append(("text", idx))
        yield _sse(
            Constants.EVENT_CONTENT_BLOCK_START,
            {
                "type": Constants.EVENT_CONTENT_BLOCK_START,
                "index": idx,
                "content_block": {"type": Constants.CONTENT_TEXT, "text": ""},
            },
        )

    for kind, idx in started_blocks:
        if idx not in stopped_blocks:
            yield _sse(
                Constants.EVENT_CONTENT_BLOCK_STOP,
                {
                    "type": Constants.EVENT_CONTENT_BLOCK_STOP,
                    "index": idx,
                },
            )

    # --- message_delta with final stop reason + usage ---
    yield _sse(
        Constants.EVENT_MESSAGE_DELTA,
        {
            "type": Constants.EVENT_MESSAGE_DELTA,
            "delta": {"stop_reason": final_stop_reason, "stop_sequence": None},
            "usage": usage_data,
        },
    )
    yield _sse(Constants.EVENT_MESSAGE_STOP, {"type": Constants.EVENT_MESSAGE_STOP})
    if observability_context is not None:
        observability_context["usage"] = usage_data
        observability_context["stop_reason"] = final_stop_reason
        observability_context["tool_calls"] = observed_tool_calls
        observability_context["estimated_output_tokens"] = estimated_output_tokens


# ---------------------------------------------------------------------------
# Backward-compatible alias (Fix 3)
# ---------------------------------------------------------------------------


async def convert_openai_streaming_to_claude(
    openai_stream, original_request: ClaudeMessagesRequest, logger
):
    """Legacy wrapper — delegates to the unified converter."""
    async for event in convert_openai_streaming_to_claude_with_cancellation(
        openai_stream, original_request, logger
    ):
        yield event


def claude_response_to_sse(response: dict):
    """Serialize a complete Claude response dict into Anthropic SSE.

    Unlike ``optimized_response_to_sse`` (text-only), this emits text, thinking,
    AND tool_use blocks — so a non-streamed result (e.g. the server-side search
    loop's final response, which may contain client tool calls) streams back to
    Claude Code without dropping tool calls.
    """
    content = response.get("content") or []
    message = dict(response)
    message["content"] = []
    message["stop_reason"] = None

    yield _sse(Constants.EVENT_MESSAGE_START, {"type": Constants.EVENT_MESSAGE_START, "message": message})

    emitted = 0
    for idx, block in enumerate(content):
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == Constants.CONTENT_TOOL_USE:
            yield _sse(Constants.EVENT_CONTENT_BLOCK_START, {
                "type": Constants.EVENT_CONTENT_BLOCK_START, "index": idx,
                "content_block": {
                    "type": Constants.CONTENT_TOOL_USE,
                    "id": block.get("id"),
                    "name": block.get("name"),
                    "input": {},
                },
            })
            args_json = json.dumps(block.get("input") or {}, ensure_ascii=False)
            yield _sse(Constants.EVENT_CONTENT_BLOCK_DELTA, {
                "type": Constants.EVENT_CONTENT_BLOCK_DELTA, "index": idx,
                "delta": {"type": Constants.DELTA_INPUT_JSON, "partial_json": args_json},
            })
            yield _sse(Constants.EVENT_CONTENT_BLOCK_STOP, {"type": Constants.EVENT_CONTENT_BLOCK_STOP, "index": idx})
            emitted += 1
        elif btype == "thinking":
            yield _sse(Constants.EVENT_CONTENT_BLOCK_START, {
                "type": Constants.EVENT_CONTENT_BLOCK_START, "index": idx,
                "content_block": {"type": "thinking", "thinking": ""},
            })
            yield _sse(Constants.EVENT_CONTENT_BLOCK_DELTA, {
                "type": Constants.EVENT_CONTENT_BLOCK_DELTA, "index": idx,
                "delta": {"type": "thinking_delta", "thinking": block.get("thinking") or ""},
            })
            yield _sse(Constants.EVENT_CONTENT_BLOCK_STOP, {"type": Constants.EVENT_CONTENT_BLOCK_STOP, "index": idx})
            emitted += 1
        else:  # text
            yield _sse(Constants.EVENT_CONTENT_BLOCK_START, {
                "type": Constants.EVENT_CONTENT_BLOCK_START, "index": idx,
                "content_block": {"type": Constants.CONTENT_TEXT, "text": ""},
            })
            text = block.get("text") or ""
            if text:
                yield _sse(Constants.EVENT_CONTENT_BLOCK_DELTA, {
                    "type": Constants.EVENT_CONTENT_BLOCK_DELTA, "index": idx,
                    "delta": {"type": Constants.DELTA_TEXT, "text": text},
                })
            yield _sse(Constants.EVENT_CONTENT_BLOCK_STOP, {"type": Constants.EVENT_CONTENT_BLOCK_STOP, "index": idx})
            emitted += 1

    if emitted == 0:
        # Always give Claude Code at least one block.
        yield _sse(Constants.EVENT_CONTENT_BLOCK_START, {
            "type": Constants.EVENT_CONTENT_BLOCK_START, "index": 0,
            "content_block": {"type": Constants.CONTENT_TEXT, "text": ""},
        })
        yield _sse(Constants.EVENT_CONTENT_BLOCK_STOP, {"type": Constants.EVENT_CONTENT_BLOCK_STOP, "index": 0})

    yield _sse(Constants.EVENT_MESSAGE_DELTA, {
        "type": Constants.EVENT_MESSAGE_DELTA,
        "delta": {
            "stop_reason": response.get("stop_reason") or Constants.STOP_END_TURN,
            "stop_sequence": response.get("stop_sequence"),
        },
        "usage": response.get("usage") or {},
    })
    yield _sse(Constants.EVENT_MESSAGE_STOP, {"type": Constants.EVENT_MESSAGE_STOP})
