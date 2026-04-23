# MCP Server Guide

This proxy can work with Claude Code MCP servers, including schema-less Claude Code tool families that get converted into explicit JSON-schema tools.

## Recommendations

### Use explicit type hints

FastMCP generates tool schemas from type hints. Prefer precise types over `Any`.

### Give safe defaults

Models can omit optional fields or produce empty objects. Parameters should generally have safe defaults.

### Parse defensively

Tool parameters may arrive as lists, JSON strings, or comma-separated strings. MCP implementations should normalize inputs instead of assuming one format.

### Return proper MCP result types

Image-producing tools should return `ImageContent` alongside useful text context where appropriate.

### Keep descriptions concrete

Good tool docstrings materially improve model behavior because they shape the emitted tool schema.

## Claude Code Compatibility

For schema-less tool families like computer-use, bash, and text-editor tools, the proxy converts them into explicit function tools. MCP servers should still be written as if a non-Anthropic model might call them with imperfect JSON.

## Bundled MCP Layout

Bundled MCP servers live under `MCP/`. Each server should be self-contained and avoid hardcoded machine-specific paths in its checked-in launcher or docs.
