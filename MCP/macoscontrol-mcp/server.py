"""
macOS Control MCP Server
========================
A local MCP server that gives any LLM (via Claude Code + proxy) the ability
to see and control your screen — screenshots, mouse, keyboard, scrolling,
and drag operations.

Also includes bash and text-editor tools matching the Anthropic schema-less
tool surface so the full computer-use agent loop works end to end.

Usage:
    # stdio mode (for Claude Code MCP config)
    python server.py

    # or via the installed entry-point
    macoscontrol-mcp
"""

import base64
import io
import json
import logging
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent, ImageContent

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("macoscontrol-mcp")

# ---------------------------------------------------------------------------
# Screen / input backend
# ---------------------------------------------------------------------------
try:
    import pyautogui

    pyautogui.FAILSAFE = True  # move mouse to corner to abort
    pyautogui.PAUSE = 0.05  # small pause between actions
    HAS_PYAUTOGUI = True
except ImportError:
    HAS_PYAUTOGUI = False
    logger.warning("pyautogui not installed — mouse/keyboard actions disabled")

try:
    import mss
    import mss.tools

    HAS_MSS = True
except ImportError:
    HAS_MSS = False
    logger.warning("mss not installed — screenshot fallback to pyautogui")

try:
    from PIL import Image as PILImage

    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    logger.warning("Pillow not installed — screenshot resizing disabled")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DISPLAY_WIDTH = int(os.environ.get("DISPLAY_WIDTH", "0"))
DISPLAY_HEIGHT = int(os.environ.get("DISPLAY_HEIGHT", "0"))
SCREENSHOT_MAX_LONG_EDGE = int(os.environ.get("SCREENSHOT_MAX_EDGE", "1280"))  # Balance between detail and token cost
SCREENSHOT_QUALITY = int(os.environ.get("SCREENSHOT_QUALITY", "55"))  # Lower quality = fewer tokens

# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "macOS Control",
    instructions=(
        "This server provides tools to control the local computer: "
        "take screenshots, click, type, scroll, drag, press keys, "
        "run bash commands, and edit text files.\n\n"
        "CRITICAL RULES — follow these strictly:\n\n"
        "1. To OPEN AN APP: use action='open_app' with text='AppName' "
        "(e.g. action='open_app', text='Google Chrome'). "
        "NEVER try to click on dock icons — use open_app instead.\n\n"
        "2. To OPEN A URL: use action='open_url' with text='https://example.com'. "
        "This opens the URL in the default browser reliably.\n\n"
        "3. To RUN SHELL COMMANDS: use the 'bash' tool with the command parameter. "
        "Do NOT type shell commands — use bash tool.\n\n"
        "4. Use 'type' action ONLY for typing into GUI text fields that are already focused.\n\n"
        "5. To save a screenshot: pass save_path='~/Desktop/screenshot.png'.\n\n"
        "6. Take MINIMAL screenshots — each one uses ~8K tokens of context."
    ),
)


# ===================================================================
# HELPERS (defined first so tools can use them)
# ===================================================================

def _parse_coord(val) -> Optional[list]:
    """Parse a coordinate value into a list of numbers.

    Handles: JSON string '[100, 200]', comma string '100,200',
    actual list [100, 200], or None.
    """
    if val is None:
        return None
    if isinstance(val, list):
        return val
    if isinstance(val, (int, float)):
        return [val]
    val = str(val).strip()
    if not val:
        return None
    # Try JSON parse first
    try:
        parsed = json.loads(val)
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    # Try comma-separated fallback
    try:
        parts = [float(x.strip()) for x in val.split(",")]
        return parts
    except (ValueError, AttributeError):
        pass
    return None


def _get_screen_size() -> tuple:
    """Get actual screen resolution."""
    if DISPLAY_WIDTH and DISPLAY_HEIGHT:
        return DISPLAY_WIDTH, DISPLAY_HEIGHT
    if HAS_PYAUTOGUI:
        size = pyautogui.size()
        return size.width, size.height
    return 1920, 1080  # fallback


def _ensure_pyautogui():
    if not HAS_PYAUTOGUI:
        raise RuntimeError(
            "pyautogui is not installed. Run: pip install pyautogui"
        )


def _type_via_clipboard_mac(text: str):
    """Type text on macOS via clipboard for reliable Unicode support."""
    import subprocess as sp

    # Save current clipboard
    try:
        old_clip = sp.run(["pbpaste"], capture_output=True, text=True, timeout=2).stdout
    except Exception:
        old_clip = None

    # Set clipboard and paste
    sp.run(["pbcopy"], input=text, text=True, timeout=2)
    pyautogui.hotkey("command", "v")
    time.sleep(0.1)

    # Restore old clipboard
    if old_clip is not None:
        try:
            sp.run(["pbcopy"], input=old_clip, text=True, timeout=2)
        except Exception:
            pass


def _capture_screenshot(region: Optional[list] = None, save_path: Optional[str] = None) -> tuple:
    """Capture screenshot and return as (base64_str, format_str) tuple.

    If save_path is provided, also saves the full-resolution image as PNG to that path.
    format_str is 'png' or 'jpeg' (for the API-optimized version).
    """
    # Capture
    img = None
    if HAS_MSS:
        with mss.mss() as sct:
            if region and len(region) == 4:
                x1, y1, x2, y2 = [int(v) for v in region]
                monitor = {"left": x1, "top": y1, "width": x2 - x1, "height": y2 - y1}
            else:
                monitor = sct.monitors[0]  # full screen (all monitors)
            raw = sct.grab(monitor)
            if HAS_PIL:
                img = PILImage.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
            else:
                png_bytes = mss.tools.to_png(raw.rgb, raw.size)
                if save_path:
                    _save_to_file(png_bytes, save_path)
                b64 = base64.b64encode(png_bytes).decode("utf-8")
                return b64, "png"
    elif HAS_PYAUTOGUI:
        img = pyautogui.screenshot(
            region=tuple(int(v) for v in region) if region and len(region) == 4 else None
        )
    else:
        raise RuntimeError("No screenshot backend available (install mss or pyautogui)")

    if img is None:
        raise RuntimeError("Screenshot capture returned None")

    # Save full-resolution PNG before any resizing
    if save_path and HAS_PIL:
        save_file = Path(save_path).expanduser()
        save_file.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(save_file), format="PNG")
        logger.info(f"Screenshot saved to {save_file}")
    elif save_path:
        buf_save = io.BytesIO()
        img.save(buf_save, format="PNG")
        _save_to_file(buf_save.getvalue(), save_path)

    if not HAS_PIL:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return b64, "png"

    # Resize to fit within Anthropic's constraints (for API return only)
    w, h = img.size
    long_edge = max(w, h)
    if long_edge > SCREENSHOT_MAX_LONG_EDGE:
        scale = SCREENSHOT_MAX_LONG_EDGE / long_edge
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), PILImage.LANCZOS)

    # Encode
    buf = io.BytesIO()
    if SCREENSHOT_QUALITY < 100:
        img = img.convert("RGB")
        img.save(buf, format="JPEG", quality=SCREENSHOT_QUALITY, optimize=True)
        fmt = "jpeg"
    else:
        img.save(buf, format="PNG", optimize=True)
        fmt = "png"

    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return b64, fmt


def _save_to_file(data: bytes, save_path: str):
    """Save raw bytes to a file."""
    filepath = Path(save_path).expanduser()
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_bytes(data)
    logger.info(f"Screenshot saved to {filepath}")


CoordinateValue = Optional[str | list[float]]


# ===================================================================
# COMPUTER TOOL
# ===================================================================

@mcp.tool()
def computer(
    action: str = "screenshot",
    coordinate: CoordinateValue = None,
    text: Optional[str] = None,
    scroll_direction: Optional[str] = None,
    scroll_amount: Optional[int] = None,
    start_coordinate: CoordinateValue = None,
    duration: Optional[float] = None,
    region: CoordinateValue = None,
    save_path: Optional[str] = None,
    key: Optional[str] = None,
    command: Optional[str] = None,
):
    """Control the computer screen — take screenshots, click, type, scroll, drag.

    Args:
        action: One of: screenshot, open_app, open_url, click, left_click, right_click,
                middle_click, double_click, triple_click, left_click_drag, mouse_move,
                type, key, press, scroll, wait, hold_key, cursor_position, zoom.
                Use open_app to launch apps, open_url to open websites. Defaults to screenshot.
        coordinate: Target position [x, y] — as a list or string, e.g. [100, 200] or "[100, 200]"
        text: Text to type, key combo to press, or modifier key name
        scroll_direction: up, down, left, or right
        scroll_amount: Number of scroll clicks (default 3)
        start_coordinate: Drag start position [x, y] — as a list or string
        duration: Seconds for wait, hold_key, or drag duration
        region: Zoom/crop region [x1, y1, x2, y2] — as a list or string
        save_path: File path to save screenshot/zoom as PNG (e.g. "~/Desktop/screenshot.png")
        key: Alias for text when action is key/press (e.g. "Return", "ctrl+s")
        command: Shell command — if provided, automatically uses bash tool instead
    """
    # --- Action aliases (models often use these instead of official names) ---
    ACTION_ALIASES = {
        "click": "left_click",
        "press": "key",
        "keypress": "key",
        "enter": "key",
        "hotkey": "key",
        "move": "mouse_move",
        "drag": "left_click_drag",
        "capture": "screenshot",
    }
    action = ACTION_ALIASES.get(action, action)

    # If 'key' param provided but text is empty, use key as text (for press/key actions)
    if key and not text:
        text = key

    # If 'command' param provided, delegate to bash tool directly
    if command:
        return bash(command=command)

    # --- Smart helper actions ---
    # open_app: open an application by name (macOS)
    if action == "open_app":
        app_name = text or "Google Chrome"
        return bash(command=f'open -a "{app_name}"')

    # open_url: open a URL in the default browser
    if action == "open_url":
        url = text or "https://google.com"
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        return bash(command=f'open "{url}"')

    # cursor_position: report current mouse position
    if action == "cursor_position":
        _ensure_pyautogui()
        pos = pyautogui.position()
        return f"Cursor is at ({pos.x}, {pos.y})"

    # Parse coordinate values into lists (accepts both lists and strings)
    coord = _parse_coord(coordinate)
    start_coord = _parse_coord(start_coordinate)
    reg = _parse_coord(region)

    logger.info(f"computer action={action} coordinate={coord} text={text}")

    try:
        # --- Screenshot actions (return image via MCP ImageContent) ---
        if action == "screenshot":
            b64, fmt = _capture_screenshot(save_path=save_path)
            w, h = _get_screen_size()
            mime = f"image/{fmt}"
            msg = f"Screenshot captured ({w}x{h} display)"
            if save_path:
                msg += f" — saved to {Path(save_path).expanduser()}"
            return CallToolResult(
                content=[
                    ImageContent(type="image", data=b64, mimeType=mime),
                    TextContent(type="text", text=msg),
                ],
            )

        elif action == "zoom":
            if not reg or len(reg) != 4:
                return "Error: zoom requires region='[x1, y1, x2, y2]'"
            b64, fmt = _capture_screenshot(region=reg, save_path=save_path)
            mime = f"image/{fmt}"
            msg = f"Zoomed region [{reg[0]},{reg[1]}]-[{reg[2]},{reg[3]}]"
            if save_path:
                msg += f" — saved to {Path(save_path).expanduser()}"
            return CallToolResult(
                content=[
                    ImageContent(type="image", data=b64, mimeType=mime),
                    TextContent(type="text", text=msg),
                ],
            )

        # --- Click actions ---
        elif action in ("left_click", "right_click", "middle_click", "double_click", "triple_click"):
            if not coord or len(coord) < 2:
                return f"Error: {action} requires coordinate='[x, y]'"
            _ensure_pyautogui()
            screen_w, screen_h = pyautogui.size()
            raw_x, raw_y = int(coord[0]), int(coord[1])
            # Clamp coordinates to screen bounds
            x = max(0, min(raw_x, screen_w - 1))
            y = max(0, min(raw_y, screen_h - 1))
            # Handle modifier keys via text parameter
            modifier = text.lower() if text else None
            key_map = {"shift": "shift", "ctrl": "ctrl", "alt": "alt", "super": "command", "cmd": "command"}
            held_key = key_map.get(modifier) if modifier else None
            modifier = text.lower() if text else None
            key_map = {"shift": "shift", "ctrl": "ctrl", "alt": "alt", "super": "command", "cmd": "command"}
            held_key = key_map.get(modifier) if modifier else None

            if held_key:
                pyautogui.keyDown(held_key)

            if action == "left_click":
                pyautogui.click(x, y)
            elif action == "right_click":
                pyautogui.rightClick(x, y)
            elif action == "middle_click":
                pyautogui.middleClick(x, y)
            elif action == "double_click":
                pyautogui.doubleClick(x, y)
            elif action == "triple_click":
                pyautogui.tripleClick(x, y)

            if held_key:
                pyautogui.keyUp(held_key)

            return f"{action} at ({x}, {y})" + (f" with {modifier}" if modifier else "")

        elif action == "left_click_drag":
            if not start_coord or len(start_coord) < 2:
                return "Error: left_click_drag requires start_coordinate='[x, y]'"
            if not coord or len(coord) < 2:
                return "Error: left_click_drag requires coordinate='[x, y]' (end)"
            _ensure_pyautogui()
            sx, sy = int(start_coord[0]), int(start_coord[1])
            ex, ey = int(coord[0]), int(coord[1])
            drag_duration = duration or 0.5
            pyautogui.moveTo(sx, sy)
            pyautogui.mouseDown()
            pyautogui.moveTo(ex, ey, duration=drag_duration)
            pyautogui.mouseUp()
            return f"Dragged from ({sx},{sy}) to ({ex},{ey})"

        elif action == "mouse_move":
            if not coord or len(coord) < 2:
                return "Error: mouse_move requires coordinate='[x, y]'"
            _ensure_pyautogui()
            x, y = int(coord[0]), int(coord[1])
            pyautogui.moveTo(x, y)
            return f"Mouse moved to ({x}, {y})"

        # --- Keyboard actions ---
        elif action == "type":
            if text is None:
                return "Error: type requires text parameter"
            _ensure_pyautogui()
            if platform.system() == "Darwin":
                _type_via_clipboard_mac(text)
            else:
                pyautogui.typewrite(text, interval=0.02) if text.isascii() else pyautogui.write(text)
            return f"Typed {len(text)} characters"

        elif action == "key":
            if text is None:
                return "Error: key requires text parameter (e.g. 'ctrl+s', 'enter')"
            _ensure_pyautogui()
            keys = [k.strip() for k in text.split("+")]
            key_map = {
                "cmd": "command", "super": "command", "win": "command",
                "ctrl": "ctrl", "alt": "alt", "shift": "shift",
                "enter": "enter", "return": "enter",
                "esc": "escape", "escape": "escape",
                "tab": "tab", "space": "space",
                "backspace": "backspace", "delete": "delete",
                "up": "up", "down": "down", "left": "left", "right": "right",
            }
            normalized = [key_map.get(k.lower(), k.lower()) for k in keys]
            if len(normalized) == 1:
                pyautogui.press(normalized[0])
            else:
                pyautogui.hotkey(*normalized)
            return f"Pressed {text}"

        # --- Scroll ---
        elif action == "scroll":
            _ensure_pyautogui()
            amount = scroll_amount or 3
            direction = (scroll_direction or "down").lower()
            x, y = None, None
            if coord and len(coord) >= 2:
                x, y = int(coord[0]), int(coord[1])
                pyautogui.moveTo(x, y)

            modifier = text.lower() if text else None
            key_map = {"shift": "shift", "ctrl": "ctrl", "alt": "alt", "super": "command"}
            held_key = key_map.get(modifier) if modifier else None

            if held_key:
                pyautogui.keyDown(held_key)

            if direction in ("up", "down"):
                clicks = amount if direction == "up" else -amount
                pyautogui.scroll(clicks)
            elif direction in ("left", "right"):
                clicks = -amount if direction == "left" else amount
                pyautogui.hscroll(clicks)

            if held_key:
                pyautogui.keyUp(held_key)

            pos = f" at ({x},{y})" if x is not None else ""
            return f"Scrolled {direction} {amount}{pos}"

        # --- Wait / hold ---
        elif action == "wait":
            wait_time = duration or 1.0
            time.sleep(min(wait_time, 10))
            return f"Waited {wait_time}s"

        elif action == "hold_key":
            if text is None:
                return "Error: hold_key requires text parameter (the key to hold)"
            _ensure_pyautogui()
            hold_duration = duration or 1.0
            pyautogui.keyDown(text)
            time.sleep(min(hold_duration, 10))
            pyautogui.keyUp(text)
            return f"Held '{text}' for {hold_duration}s"

        # --- Mouse down/up ---
        elif action in ("left_mouse_down", "left_mouse_up"):
            _ensure_pyautogui()
            if coord and len(coord) >= 2:
                pyautogui.moveTo(int(coord[0]), int(coord[1]))
            if action == "left_mouse_down":
                pyautogui.mouseDown()
            else:
                pyautogui.mouseUp()
            return action

        else:
            return f"Error: Unknown action: {action}"

    except Exception as e:
        logger.error(f"computer action failed: {e}", exc_info=True)
        return f"Error: {e}"


# ===================================================================
# BASH TOOL
# ===================================================================

_bash_process: Optional[subprocess.Popen] = None


@mcp.tool()
def bash(
    command: Optional[str] = None,
    restart: bool = False,
):
    """Execute a bash command and return stdout+stderr.

    Args:
        command: The shell command to run
        restart: If true, restart the bash session
    """
    global _bash_process

    if restart:
        if _bash_process and _bash_process.poll() is None:
            _bash_process.terminate()
            _bash_process = None
        return "Bash session restarted."

    if not command:
        return "No command provided."

    logger.info(f"bash: {command[:200]}...")

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=os.path.expanduser("~"),
            env={**os.environ, "TERM": "dumb"},
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            if output:
                output += "\n"
            output += result.stderr
        if result.returncode != 0:
            output += f"\n[Exit code: {result.returncode}]"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 120 seconds"
    except Exception as e:
        return f"Error: {e}"


# ===================================================================
# TEXT EDITOR TOOL
# ===================================================================

_undo_history: dict = {}


@mcp.tool()
def str_replace_based_edit_tool(
    command: str,
    path: str,
    file_text: Optional[str] = None,
    old_str: Optional[str] = None,
    new_str: Optional[str] = None,
    insert_line: Optional[int] = None,
    view_range: CoordinateValue = None,
):
    """View, create, and edit text files.

    Args:
        command: One of: view, create, str_replace, insert, undo_edit
        path: Absolute file path
        file_text: File content for the create command
        old_str: Text to find for str_replace
        new_str: Replacement text for str_replace or text to add for insert
        insert_line: Line number for insert command
        view_range: Line range as JSON array string, e.g. "[1, 50]"
    """
    logger.info(f"text_editor: {command} {path}")
    filepath = Path(path).expanduser()
    vr = _parse_coord(view_range)

    try:
        if command == "view":
            if not filepath.exists():
                return f"Error: File {path} does not exist"
            content = filepath.read_text()
            lines = content.splitlines(keepends=True)
            if vr and len(vr) == 2:
                start, end = max(1, int(vr[0])), min(len(lines), int(vr[1]))
                selected = lines[start - 1:end]
                numbered = "".join(
                    f"{i}: {line}" for i, line in enumerate(selected, start=start)
                )
                return numbered or "(empty range)"
            else:
                numbered = "".join(
                    f"{i}: {line}" for i, line in enumerate(lines, start=1)
                )
                return numbered or "(empty file)"

        elif command == "create":
            if file_text is None:
                return "Error: create requires file_text parameter"
            filepath.parent.mkdir(parents=True, exist_ok=True)
            if filepath.exists():
                _save_undo(filepath)
            filepath.write_text(file_text)
            return f"File created: {path} ({len(file_text)} chars)"

        elif command == "str_replace":
            if old_str is None:
                return "Error: str_replace requires old_str parameter"
            if not filepath.exists():
                return f"Error: File {path} does not exist"
            content = filepath.read_text()
            if old_str not in content:
                return f"Error: old_str not found in {path}"
            count = content.count(old_str)
            if count > 1:
                return f"Error: old_str appears {count} times — must be unique. Add more context."
            _save_undo(filepath)
            new_content = content.replace(old_str, new_str or "", 1)
            filepath.write_text(new_content)
            return f"Replaced in {path}" + (f" ({len(new_str or '')} chars)" if new_str else " (deleted)")

        elif command == "insert":
            if new_str is None:
                return "Error: insert requires new_str parameter"
            if insert_line is None:
                return "Error: insert requires insert_line parameter"
            if not filepath.exists():
                return f"Error: File {path} does not exist"
            content = filepath.read_text()
            _save_undo(filepath)
            lines = content.splitlines(keepends=True)
            idx = max(0, min(insert_line, len(lines)))
            new_line = new_str if new_str.endswith("\n") else new_str + "\n"
            lines.insert(idx, new_line)
            filepath.write_text("".join(lines))
            return f"Inserted at line {insert_line} in {path}"

        elif command == "undo_edit":
            if path not in _undo_history or not _undo_history[path]:
                return f"Error: No undo history for {path}"
            prev_content = _undo_history[path].pop()
            filepath.write_text(prev_content)
            return f"Undid last edit to {path}"

        else:
            return f"Error: Unknown command '{command}'. Use: view, create, str_replace, insert, undo_edit"

    except Exception as e:
        return f"Error: {e}"


def _save_undo(filepath: Path):
    """Save current file content for undo."""
    path_str = str(filepath)
    if path_str not in _undo_history:
        _undo_history[path_str] = []
    try:
        content = filepath.read_text()
        _undo_history[path_str].append(content)
        if len(_undo_history[path_str]) > 10:
            _undo_history[path_str] = _undo_history[path_str][-10:]
    except Exception:
        pass


# ===================================================================
# ENTRY POINT
# ===================================================================

def main():
    """Run the MCP server via stdio transport."""
    logger.info("Starting macOS Control MCP server (stdio)")
    w, h = _get_screen_size()
    logger.info(f"Screen: {w}x{h}")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
