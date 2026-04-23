# macOS Control MCP Server

A local MCP server that gives **any LLM** (via Claude Code + your proxy) the ability to see and control your macOS screen — screenshots, mouse, keyboard, scrolling, and drag operations.

This is a **free alternative** to Anthropic's built-in computer use feature (which requires Pro/Max). It works with any model through your Claude Code proxy.

## What's included

| Tool | Description |
|------|-------------|
| `computer` | Screenshot, click, type, key, scroll, drag, zoom — all 16 Anthropic actions |
| `bash` | Execute shell commands with output capture |
| `str_replace_based_edit_tool` | View, create, and edit files (view, create, str_replace, insert, undo_edit) |

## Quick Setup

### 1. Install dependencies

```bash
cd MCP/macoscontrol-mcp
pip install -e .
```

Or install manually:

```bash
pip install "mcp[cli]>=1.2.0" pyautogui Pillow mss
```

If you keep the provided project-level [.mcp.json](../../.mcp.json), Claude Code can also bootstrap the server automatically via `./MCP/macoscontrol-mcp/run.sh` with no hardcoded absolute paths.

### 2. Grant macOS Accessibility permissions

pyautogui needs Accessibility access to control mouse/keyboard:

1. Open **System Settings → Privacy & Security → Accessibility**
2. Add your terminal app (Terminal, iTerm2, VS Code, etc.)
3. Toggle it **ON**

### 3. Configure Claude Code to use this MCP

Add to your Claude Code MCP config (`~/.claude/claude_desktop_config.json` or the project-level `.mcp.json`):

```json
{
  "mcpServers": {
    "macoscontrol-mcp": {
      "command": "bash",
      "args": ["./MCP/macoscontrol-mcp/run.sh"],
      "env": {}
    }
  }
}
```

Or if you installed it as a package:

```json
{
  "mcpServers": {
    "macoscontrol-mcp": {
      "command": "macoscontrol-mcp",
      "env": {}
    }
  }
}
```

### 4. Use it

In Claude Code, you can now ask:

- "Take a screenshot of my screen"
- "Click on the search bar and type hello"
- "Open Terminal and run ls -la"
- "Edit the file at ~/test.py and fix the bug"

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DISPLAY_WIDTH` | auto-detected | Override screen width |
| `DISPLAY_HEIGHT` | auto-detected | Override screen height |
| `SCREENSHOT_QUALITY` | `80` | JPEG quality (1-100, 100=PNG) |

## How it works

```
Claude Code  →  Your Proxy (Nebius/GLM)  →  Model returns tool_use
                                               ↓
                                          MCP Server executes action
                                               ↓
                                          Returns screenshot/result
                                               ↓
Claude Code  ←  Your Proxy  ←  Model sees result, decides next action
```

The MCP server runs locally on your machine and directly controls your screen using pyautogui (mouse/keyboard) and mss (screenshots). Screenshots are automatically scaled to fit within API constraints.

## Safety

- **FAILSAFE**: Move your mouse to any corner of the screen to abort all pyautogui actions
- Commands timeout after 120 seconds
- Wait/hold actions capped at 10 seconds
- The MCP server only runs when Claude Code is active
