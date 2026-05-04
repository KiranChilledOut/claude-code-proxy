"""Local fast-path responses for Claude Code housekeeping requests."""

import json
import re
import shlex
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.conversion.request_converter import _count_tokens_text, count_claude_request_tokens
from src.core.config import config
from src.core.logging import logger
from src.models.claude import ClaudeMessagesRequest

_ENV_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")


@dataclass(frozen=True)
class LocalOptimizationResult:
    kind: str
    response: Dict[str, Any]


def _get_field(value: Any, field: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(field, default)
    return getattr(value, field, default)


def extract_text_from_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    parts: List[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue

        text = _get_field(block, "text")
        if isinstance(text, str):
            parts.append(text)
            continue

        if _get_field(block, "type") == "tool_result":
            tool_content = _get_field(block, "content", "")
            if isinstance(tool_content, str):
                parts.append(tool_content)
            elif tool_content:
                try:
                    parts.append(json.dumps(tool_content, ensure_ascii=False))
                except (TypeError, ValueError):
                    parts.append(str(tool_content))

    return "\n".join(part for part in parts if part)


def _is_env_assignment(part: str) -> bool:
    return bool(_ENV_ASSIGNMENT_RE.match(part))


def _strip_env_assignments(parts: List[str]) -> List[str]:
    cmd_start = 0
    for index, part in enumerate(parts):
        if _is_env_assignment(part):
            cmd_start = index + 1
        else:
            break
    return parts[cmd_start:]


def extract_command_prefix(command: str) -> str:
    """Extract a stable command prefix for Claude Code permission checks."""
    if "`" in command or "$(" in command:
        return "command_injection_detected"

    try:
        parts = shlex.split(command, posix=False)
        if not parts:
            return "none"

        env_prefix = []
        cmd_start = 0
        for index, part in enumerate(parts):
            if _is_env_assignment(part):
                env_prefix.append(part)
                cmd_start = index + 1
            else:
                break

        cmd_parts = parts[cmd_start:]
        if not cmd_parts:
            return "none"

        first_word = cmd_parts[0]
        two_word_commands = {"git", "npm", "docker", "kubectl", "cargo", "go", "pip", "yarn"}
        if first_word in two_word_commands and len(cmd_parts) > 1:
            second_word = cmd_parts[1]
            if not second_word.startswith("-"):
                return f"{first_word} {second_word}"
            return first_word
        if env_prefix:
            return " ".join(env_prefix) + " " + first_word
        return first_word
    except ValueError:
        parts = command.split()
        cmd_parts = _strip_env_assignments(parts)
        return cmd_parts[0] if cmd_parts else "none"


def extract_filepaths_from_command(command: str, _output: str = "") -> str:
    """Return file paths read by common shell commands in Claude's <filepaths> format."""
    listing_commands = {"ls", "dir", "find", "tree", "pwd", "cd", "mkdir", "rmdir", "rm"}
    reading_commands = {"cat", "head", "tail", "less", "more", "bat", "type"}

    try:
        parts = shlex.split(command, posix=False)
    except ValueError:
        return "<filepaths>\n</filepaths>"

    if not parts:
        return "<filepaths>\n</filepaths>"

    cmd_parts = _strip_env_assignments(parts)
    if not cmd_parts:
        return "<filepaths>\n</filepaths>"

    base_cmd = cmd_parts[0].split("/")[-1].split("\\")[-1].lower()
    if base_cmd in listing_commands:
        return "<filepaths>\n</filepaths>"

    filepaths: List[str] = []
    if base_cmd in reading_commands:
        filepaths = [part for part in cmd_parts[1:] if not part.startswith("-")]
    elif base_cmd == "grep":
        filepaths = _extract_grep_filepaths(cmd_parts)

    if not filepaths:
        return "<filepaths>\n</filepaths>"
    return "<filepaths>\n" + "\n".join(filepaths) + "\n</filepaths>"


def _extract_grep_filepaths(cmd_parts: List[str]) -> List[str]:
    flags_with_args = {"-e", "-f", "-m", "-A", "-B", "-C"}
    pattern_provided_via_flag = False
    positional: List[str] = []
    skip_next = False

    for part in cmd_parts[1:]:
        if skip_next:
            skip_next = False
            continue
        if part.startswith("-"):
            if part in flags_with_args:
                if part in {"-e", "-f"}:
                    pattern_provided_via_flag = True
                skip_next = True
            continue
        positional.append(part)

    return positional if pattern_provided_via_flag else positional[1:]


def is_quota_check_request(request: ClaudeMessagesRequest) -> bool:
    if request.max_tokens != 1 or len(request.messages) != 1:
        return False
    message = request.messages[0]
    if message.role != "user":
        return False
    return "quota" in extract_text_from_content(message.content).lower()


def is_prefix_detection_request(request: ClaudeMessagesRequest) -> Tuple[bool, str]:
    if len(request.messages) != 1 or request.messages[0].role != "user":
        return False, ""

    content = extract_text_from_content(request.messages[0].content)
    if "<policy_spec>" not in content or "Command:" not in content:
        return False, ""

    command_start = content.rfind("Command:") + len("Command:")
    return True, content[command_start:].strip()


def is_title_generation_request(request: ClaudeMessagesRequest) -> bool:
    if not request.system or request.tools:
        return False

    system_text = extract_text_from_content(request.system).lower()
    if "title" not in system_text:
        return False

    return "sentence-case title" in system_text or (
        "return json" in system_text
        and "field" in system_text
        and ("coding session" in system_text or "this session" in system_text)
    )


def is_suggestion_mode_request(request: ClaudeMessagesRequest) -> bool:
    for message in request.messages:
        if message.role != "user":
            continue
        if "[SUGGESTION MODE:" in extract_text_from_content(message.content):
            return True
    return False


def is_filepath_extraction_request(request: ClaudeMessagesRequest) -> Tuple[bool, str, str]:
    if len(request.messages) != 1 or request.messages[0].role != "user" or request.tools:
        return False, "", ""

    content = extract_text_from_content(request.messages[0].content)
    if "Command:" not in content or "Output:" not in content:
        return False, "", ""

    content_lower = content.lower()
    system_text = extract_text_from_content(request.system).lower() if request.system else ""
    asks_for_filepaths = "filepaths" in content_lower or "<filepaths>" in content_lower
    system_asks = (
        "extract any file paths" in system_text or "file paths that this command" in system_text
    )
    if not asks_for_filepaths and not system_asks:
        return False, "", ""

    command_start = content.find("Command:") + len("Command:")
    output_marker = content.find("Output:", command_start)
    if output_marker == -1:
        return False, "", ""

    command = content[command_start:output_marker].strip()
    output = content[output_marker + len("Output:") :].strip()
    for marker in ("<", "\n\n"):
        if marker in output:
            output = output.split(marker, 1)[0].strip()

    return True, command, output


def try_local_optimization(request: ClaudeMessagesRequest) -> Optional[LocalOptimizationResult]:
    if not config.enable_request_optimizations:
        return None

    handlers = (
        _try_quota_mock,
        _try_prefix_detection,
        _try_title_skip,
        _try_suggestion_skip,
        _try_filepath_mock,
    )
    for handler in handlers:
        result = handler(request)
        if result is not None:
            return result
    return None


def optimized_response_to_sse(response: Dict[str, Any]):
    """Yield a minimal Anthropic SSE lifecycle for an optimized text response."""
    content = response.get("content") or [{"type": "text", "text": ""}]
    text = ""
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = str(block.get("text") or "")
            break

    usage = response.get("usage") or {}
    message = dict(response)
    message["content"] = []
    message["stop_reason"] = None

    yield _sse("message_start", {"type": "message_start", "message": message})
    yield _sse(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
    )
    if text:
        yield _sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            },
        )
    yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
    yield _sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": response.get("stop_reason") or "end_turn",
                "stop_sequence": response.get("stop_sequence"),
            },
            "usage": usage,
        },
    )
    yield _sse("message_stop", {"type": "message_stop"})


def _try_quota_mock(request: ClaudeMessagesRequest) -> Optional[LocalOptimizationResult]:
    if not config.enable_network_probe_mock or not is_quota_check_request(request):
        return None
    logger.info("Optimization: intercepted Claude Code quota probe")
    return _text_response(request, "Quota check passed.", "quota_probe")


def _try_prefix_detection(request: ClaudeMessagesRequest) -> Optional[LocalOptimizationResult]:
    if not config.fast_prefix_detection:
        return None
    matched, command = is_prefix_detection_request(request)
    if not matched:
        return None
    logger.info("Optimization: answered Claude Code command-prefix detection locally")
    return _text_response(request, extract_command_prefix(command), "prefix_detection")


def _try_title_skip(request: ClaudeMessagesRequest) -> Optional[LocalOptimizationResult]:
    if not config.enable_title_generation_skip or not is_title_generation_request(request):
        return None
    logger.info("Optimization: skipped Claude Code title generation")
    return _text_response(request, "Conversation", "title_generation")


def _try_suggestion_skip(request: ClaudeMessagesRequest) -> Optional[LocalOptimizationResult]:
    if not config.enable_suggestion_mode_skip or not is_suggestion_mode_request(request):
        return None
    logger.info("Optimization: skipped Claude Code suggestion-mode request")
    return _text_response(request, "", "suggestion_mode")


def _try_filepath_mock(request: ClaudeMessagesRequest) -> Optional[LocalOptimizationResult]:
    if not config.enable_filepath_extraction_mock:
        return None
    matched, command, output = is_filepath_extraction_request(request)
    if not matched:
        return None
    logger.info("Optimization: answered Claude Code filepath extraction locally")
    return _text_response(
        request,
        extract_filepaths_from_command(command, output),
        "filepath_extraction",
    )


def _text_response(request: ClaudeMessagesRequest, text: str, kind: str) -> LocalOptimizationResult:
    usage = {
        "input_tokens": _safe_input_tokens(request),
        "output_tokens": max(_count_tokens_text(text), 1),
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    return LocalOptimizationResult(
        kind=kind,
        response={
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "role": "assistant",
            "model": request.model,
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": usage,
        },
    )


def _safe_input_tokens(request: ClaudeMessagesRequest) -> int:
    try:
        return count_claude_request_tokens(request)
    except Exception:
        return 1


def _sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
