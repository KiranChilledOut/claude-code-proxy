# Claude Code Project Notes

This repository is organized around two related concerns:

- `src/`: the Nebius-focused Claude API proxy
- `MCP/`: bundled MCP servers used by Claude Code

## Read First

Before making non-trivial changes, prefer the tracked reference docs in `docs/`:

- `docs/ARCHITECTURE.md`
- `docs/TOOL_CALL_FORMAT.md`
- `docs/MCP_SERVER_GUIDE.md`
- `docs/BINARY_PACKAGING.md`

## Working Conventions

- Read target files before editing them.
- Prefer focused, surgical changes over speculative refactors.
- Keep public documentation in tracked files under `docs/`.
- Do not depend on ignored local-only state such as `.claude/`.
- Keep checked-in MCP config portable and repo-relative.

## Repository Layout

- `src/`: proxy implementation
- `tests/`: automated tests
- `docs/`: tracked project documentation
- `MCP/`: bundled MCP servers
- `start_proxy.py`: local convenience launcher

## MCP Notes

- The bundled macOS control MCP lives in `MCP/macoscontrol-mcp/`.
- The checked-in `.mcp.json` is intended to work from a fresh clone without absolute paths.

## Current Focus Areas

- Tool-call streaming and argument sanitization are core integration paths.
- MCP compatibility guidance is documented in `docs/MCP_SERVER_GUIDE.md`.
