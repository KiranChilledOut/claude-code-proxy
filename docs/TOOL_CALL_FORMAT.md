# Tool Call Format

Claude Code expects Anthropic SSE tool-call semantics even when the upstream model produces OpenAI-style streaming tool calls.

## Expected Claude SSE Pattern

```text
content_block_start  -> tool_use block with empty `input: {}`
content_block_delta  -> one or more `input_json_delta.partial_json` fragments
content_block_stop
```

## Critical Rule

Do not emit `partial_json: "{}"` at tool-block start.

Claude Code concatenates every `partial_json` fragment. If the stream starts with `{}` and later emits the real arguments, the result becomes invalid JSON such as:

```text
{}{"command":"ls -la"}
```

## Proxy Strategy

1. Start the tool block when the first tool-call delta arrives.
2. Buffer argument fragments from upstream deltas.
3. Sanitize the full argument string once the tool call completes.
4. Emit a single clean `partial_json` payload.

## Common Failure Modes

- XML-style arguments instead of JSON
- arguments embedded in the function name
- empty arguments on the first streamed tool delta
- duplicated buffering before and after block start
- required MCP parameters missing from malformed model output

## Debugging Checklist

- Inspect raw tool-call deltas from the upstream model.
- Confirm the proxy emits a single flushed sanitized JSON blob.
- Verify `finish_reason` handling for tool calls.
- Compare sanitized output with the MCP tool schema Claude Code receives.
