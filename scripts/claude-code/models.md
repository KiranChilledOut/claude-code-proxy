---
description: List all Token Factory models with their shortnames; pick one to switch
argument-hint: [optional: model id, shortname, or list number]
model: glm
allowed-tools: Bash
---

Switch the active Claude Code model to a Token Factory model by writing the choice to `~/.claude/settings.local.json`. All real work lives in `~/.claude/commands/_models_helper.py` — your job is to call it. Do not roll your own bash; the helper handles all parsing.

## With argument

If `$ARGUMENTS` is non-empty, run exactly:

```
python3 ~/.claude/commands/_models_helper.py set "$ARGUMENTS"
```

The helper accepts both ids (`glm`, `moonshotai/Kimi-K2.5`) and list numbers (`5`). Then tell the user in one short sentence: `Set model to <id>; next request in this session will route through it.`

If the helper exits non-zero, report what it printed and stop.

## Without argument

1. Run exactly:

   ```
   python3 ~/.claude/commands/_models_helper.py list
   ```

   Show the user the helper's stdout verbatim (it's already formatted as a numbered list with shortcuts on top, then the Token Factory catalog with `(aka ...)` annotations).

   If the helper printed `PROXY_UNREACHABLE` on stderr, stop and tell the user the proxy isn't running on `$ANTHROPIC_BASE_URL` (default `http://localhost:8083`); they need to start it from `~/Documents/claude-nebius/proxy`.

2. Ask the user, in one line: `Type a number or paste any model id.`

3. When the user replies, run exactly:

   ```
   python3 ~/.claude/commands/_models_helper.py set "<their-reply-verbatim>"
   ```

   The helper accepts both numbers and ids; no need to look anything up yourself. Then tell the user: `Set model to <id-from-helper-output>; next request in this session will route through it.`

## Rules

- Two bash invocations max (`list`, then `set`). Never construct an inline curl or python one-liner.
- Do not modify `~/.claude/settings.json`. The helper writes only `.local.json`.
- Do not echo `$ARGUMENTS` back to the user; they already typed it.
