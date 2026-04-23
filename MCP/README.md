# MCP Servers

This directory contains bundled Model Context Protocol servers intended to work with this repository out of the box.

## Included Servers

- `macoscontrol-mcp` — local macOS screen-control MCP for screenshots, mouse, keyboard, and related desktop automation

## Conventions

- Keep each MCP self-contained in its own subdirectory.
- Use repo-relative launchers for checked-in config.
- Avoid hardcoded machine-specific paths.
- Keep generated environments and caches out of version control.
