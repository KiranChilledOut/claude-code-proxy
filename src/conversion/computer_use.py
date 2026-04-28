"""Computer Use tool conversion: Claude schema-less tools <-> OpenAI function tools.

Claude's computer use, bash, and text_editor tools are "schema-less" — the schema
is baked into the model.  OpenAI-compatible backends only understand standard
function tools, so we:

  1. On the REQUEST side: convert schema-less tools into explicit function tool
     definitions with full JSON Schema, so any OpenAI-compatible model can call them.
  2. On the RESPONSE side: the standard function-call conversion already works;
     no special handling is needed — the model returns function calls with the
     same name/arguments and the existing converter turns them into Claude
     tool_use blocks.

The schemas below are extracted from the Anthropic documentation and cover:
  - computer_20251124 / computer_20250124  (screenshot, click, type, key, scroll …)
  - bash_20250124  (command, restart)
  - text_editor_20250728 / text_editor_20250124  (view, create, str_replace, insert, undo_edit)
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Explicit JSON Schemas for schema-less Anthropic tools
# ---------------------------------------------------------------------------

COMPUTER_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": [
                "screenshot",
                "left_click",
                "right_click",
                "middle_click",
                "double_click",
                "triple_click",
                "left_click_drag",
                "mouse_move",
                "type",
                "key",
                "scroll",
                "wait",
                "hold_key",
                "left_mouse_down",
                "left_mouse_up",
                "zoom",
                "open_app",
                "open_url",
                "cursor_position",
            ],
            "description": "The computer action to perform. Use 'open_app' with text='AppName' to launch applications. Use 'open_url' with text='https://...' to open URLs.",
        },
        "coordinate": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 2,
            "maxItems": 2,
            "description": "[x, y] pixel coordinates for click, drag, scroll, and mouse_move actions.",
        },
        "text": {
            "type": "string",
            "description": "Text to type (for 'type' action), key combo (for 'key' action, e.g. 'ctrl+s'), or modifier key for click/scroll actions (e.g. 'shift').",
        },
        "scroll_direction": {
            "type": "string",
            "enum": ["up", "down", "left", "right"],
            "description": "Direction to scroll.",
        },
        "scroll_amount": {
            "type": "integer",
            "description": "Number of scroll clicks.",
        },
        "start_coordinate": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 2,
            "maxItems": 2,
            "description": "[x, y] start coordinates for left_click_drag.",
        },
        "duration": {
            "type": "number",
            "description": "Duration in seconds for 'wait' and 'hold_key' actions.",
        },
        "region": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 4,
            "maxItems": 4,
            "description": "[x1, y1, x2, y2] rectangle for 'zoom' action.",
        },
    },
    "required": ["action"],
}


BASH_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "command": {
            "type": "string",
            "description": "The bash command to execute.",
        },
        "restart": {
            "type": "boolean",
            "description": "If true, restart the bash session.",
        },
    },
}


TEXT_EDITOR_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "command": {
            "type": "string",
            "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
            "description": "The text editor command.",
        },
        "path": {
            "type": "string",
            "description": "Absolute file path to operate on.",
        },
        "file_text": {
            "type": "string",
            "description": "Full file content for 'create' command.",
        },
        "old_str": {
            "type": "string",
            "description": "String to replace (for 'str_replace').",
        },
        "new_str": {
            "type": "string",
            "description": "Replacement string (for 'str_replace' and 'insert').",
        },
        "insert_line": {
            "type": "integer",
            "description": "Line number to insert at (for 'insert').",
        },
        "view_range": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 2,
            "maxItems": 2,
            "description": "[start_line, end_line] for 'view' command.",
        },
    },
    "required": ["command", "path"],
}


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def is_computer_use_tool(tool) -> bool:
    """Check if a tool is a schema-less Anthropic computer-use family tool."""
    tool_type = getattr(tool, "type", None) or ""
    return any(tool_type.startswith(prefix) for prefix in ("computer_", "bash_", "text_editor_"))


def get_schema_for_tool(tool) -> Optional[Dict[str, Any]]:
    """Return the explicit function schema for a schema-less Anthropic tool."""
    tool_type = getattr(tool, "type", None) or ""
    if tool_type.startswith("computer_"):
        return COMPUTER_TOOL_SCHEMA
    if tool_type.startswith("bash_"):
        return BASH_TOOL_SCHEMA
    if tool_type.startswith("text_editor_"):
        return TEXT_EDITOR_TOOL_SCHEMA
    return None


def build_computer_use_system_prompt(tools) -> str:
    """Build an optional system-prompt supplement that tells the model about
    the computer environment (screen dimensions, available actions).

    This helps non-Claude models understand what the tools do, since they
    don't have the built-in computer-use system prompt that Claude has.
    """
    parts = []
    for tool in tools:
        tool_type = getattr(tool, "type", None) or ""

        if tool_type.startswith("computer_"):
            w = getattr(tool, "display_width_px", 1024) or 1024
            h = getattr(tool, "display_height_px", 768) or 768
            parts.append(
                f"You have access to a computer tool named '{tool.name}' that controls "
                f"a {w}x{h} pixel display. You can take screenshots, click at coordinates, "
                f"type text, press keys, scroll, drag, and more. When you need to see "
                f"what's on screen, use the screenshot action. Coordinates are [x, y] "
                f"pixels from the top-left corner. Always take a screenshot first before "
                f"interacting with the screen.\n\n"
                f"CRITICAL RULES:\n"
                f"1. To OPEN AN APP: use action='open_app' with text='AppName' "
                f"(e.g. action='open_app', text='Google Chrome'). "
                f"NEVER try to click on dock icons — use open_app instead.\n"
                f"2. To OPEN A URL: use action='open_url' with text='https://example.com'.\n"
                f"3. Use 'type' action ONLY for typing into GUI text fields already focused.\n"
                f"4. All tool arguments MUST be valid JSON. Use the 'action' field for the "
                f"action name, 'coordinate' for [x,y], 'text' for strings."
            )

        elif tool_type.startswith("bash_"):
            parts.append(
                f"You have access to a bash tool named '{tool.name}' that executes "
                f"shell commands. Use the 'command' field to run commands. Set 'restart' "
                f"to true to restart the bash session if needed."
            )

        elif tool_type.startswith("text_editor_"):
            parts.append(
                f"You have access to a text editor tool named '{tool.name}' for file "
                f"operations. Commands: 'view' (view file or line range), 'create' "
                f"(create new file), 'str_replace' (find and replace text), 'insert' "
                f"(insert text at line), 'undo_edit' (undo last edit). Always provide "
                f"an absolute file path."
            )

    if not parts:
        return ""
    return (
        "\n\n[Computer Use Environment]\n"
        + "\n".join(parts)
        + "\n\nAfter each action, evaluate whether you achieved the right outcome. "
        "If not correct, try again. Only when you confirm a step was executed "
        "correctly should you move on to the next one."
    )


# ---------------------------------------------------------------------------
# Request-side conversion
# ---------------------------------------------------------------------------


def convert_schema_less_tools(tools) -> tuple:
    """Convert schema-less Anthropic tools to standard OpenAI function tools.

    Returns:
        (converted_tools, computer_use_system_supplement, has_computer_use)

    - converted_tools: list of OpenAI-format tool dicts (schema-less tools
      replaced with function tools; standard tools returned unchanged as None
      placeholders).
    - computer_use_system_supplement: str to prepend/append to the system prompt
    - has_computer_use: True if any computer-use tools were detected
    """
    converted = []
    has_computer_use = False

    for tool in tools:
        if is_computer_use_tool(tool):
            has_computer_use = True
            schema = get_schema_for_tool(tool)
            if schema is None:
                continue  # unknown schema-less tool type, skip

            tool_type = getattr(tool, "type", None) or ""
            desc = ""
            if tool_type.startswith("computer_"):
                w = getattr(tool, "display_width_px", 1024) or 1024
                h = getattr(tool, "display_height_px", 768) or 768
                desc = (
                    f"Control a {w}x{h} pixel computer display. Take screenshots, "
                    f"click at coordinates, type text, press keys, scroll, and more."
                )
            elif tool_type.startswith("bash_"):
                desc = "Execute bash shell commands and scripts."
            elif tool_type.startswith("text_editor_"):
                desc = "View, create, and edit text files using commands like view, create, str_replace, insert, undo_edit."

            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": desc,
                        "parameters": schema,
                    },
                }
            )
            logger.info(
                "Converted schema-less tool '%s' (type=%s) to function tool",
                tool.name,
                tool_type,
            )
        else:
            # Standard tool — will be converted by the normal path, mark as None
            converted.append(None)

    supplement = build_computer_use_system_prompt(tools) if has_computer_use else ""
    return converted, supplement, has_computer_use
