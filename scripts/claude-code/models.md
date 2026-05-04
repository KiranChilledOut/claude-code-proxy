---
description: List all Token Factory models with their shortnames; pick one to switch
argument-hint: [optional: model id, shortname, or list number]
model: glm
allowed-tools: Bash
---

Switch the active Claude Code model to a Token Factory model. The helper at `~/.claude/commands/_models_helper.py` does the actual work.

## With argument

If `$ARGUMENTS` is non-empty, run exactly:

```
python3 ~/.claude/commands/_models_helper.py set "$ARGUMENTS"
```

Then tell the user in one short sentence: `Set model to <id>; next request in this session will route through it.`

If the helper exits non-zero, report what it printed and stop.

## Without argument

1. Run exactly:

   ```
   python3 ~/.claude/commands/_models_helper.py list
   ```

   Claude Code displays the helper's stdout as the bash tool output and collapses it past ~3 lines (`ctrl+o to expand`). That's expected — the bash output is the authoritative list. **Do not retype, paraphrase, or summarize the list in your text response.** Earlier attempts at retyping showed the model leaking memorized ids from training (`Kimi-K2` instead of `Kimi-K2.5`, `GLM-4.5` instead of `GLM-5`, etc.) — which makes "pick by id" actively unsafe.

   If the helper printed `PROXY_UNREACHABLE` on stderr, stop and tell the user the proxy isn't running on `$ANTHROPIC_BASE_URL` (default `http://localhost:8083`); they need to start it from `~/Documents/claude-nebius/proxy`.

2. Reply with exactly this single line and nothing else: `Pick a number or paste a model id (ctrl+o on the list above to expand).`

3. When the user replies, run exactly:

   ```
   python3 ~/.claude/commands/_models_helper.py set "<their-reply-verbatim>"
   ```

   The helper accepts numbers (1-30), shortnames (`glm`, `qwen-32`, etc.), and full ids — no need to look anything up yourself. Then confirm: `Set model to <id-from-helper-output>; next request in this session will route through it.`

## Rules

- Never type a model id from memory. The bash output above is the only authoritative source; if it didn't show what you expected, say so rather than guessing.
- Do not modify `~/.claude/settings.json`; the helper writes only `.local.json`.
- Do not echo `$ARGUMENTS` back to the user; they already typed it.
