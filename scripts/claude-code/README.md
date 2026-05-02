# `/models` slash command for Claude Code

A two-file Claude Code custom slash command that surfaces this proxy's full Token Factory shortcut catalog inside Claude Code (the built-in `/model` picker only shows the four hardcoded Anthropic entries plus the active custom model — see `docs/ARCHITECTURE.md` for why).

Typing `/models` renders a 30-entry combined list (curated shortcuts + any live catalog extras), takes a number or pasted id, and writes the choice to `~/.claude/settings.local.json` (the same file Claude Code's built-in `/model` writes to). The next request in your session uses the new model.

## Files

- **`models.md`** — the slash command body. Includes the 30-entry catalog as static text the model copies verbatim into its reply (so the picker doesn't get hidden behind Claude Code's bash-output collapse). Pinned to `model: glm` so the command itself runs on a model capable enough to follow tool-use instructions, regardless of what's currently selected.
- **`_models_helper.py`** — companion script. Subcommands:
  - `set <id-or-number>` — writes the choice to `~/.claude/settings.local.json`. Resolves numbers (1-30), short names, and full ids.
  - `extras` — fetches `/v1/models` and prints any upstream ids that aren't in the hardcoded list. Empty when the catalog matches.
  - `list` — combined hardcoded + live listing (kept for `_models_helper.py` standalone use; not used by the slash command itself, which uses static text from `models.md`).

The catalog appears in **both** files: `models.md` has it as static markdown for display (model can't hallucinate a verbatim copy of its own prompt), and `_models_helper.py` has it as `HARDCODED_SHORTCUTS` for `set` lookup. They must stay in sync — edit both at the same time when Nebius rotates a model id.

## Install

Copy both files into your user-level Claude Code commands directory:

```bash
mkdir -p ~/.claude/commands
cp scripts/claude-code/models.md          ~/.claude/commands/
cp scripts/claude-code/_models_helper.py  ~/.claude/commands/
chmod +x ~/.claude/commands/_models_helper.py
```

Then start a fresh Claude Code session (`claudius`) and try `/models`.

## Usage

```
/models                  # render the combined list, prompt for a pick
/models glm              # set directly to glm (proxy alias)
/models qwen-32          # set directly to a helper-only shortcut
/models 5                # set to whatever is at index 5 in the list
/models Qwen/Qwen3-32B   # paste any full id
```

## How shortcuts vs full ids get written

For the 13 aliases the proxy itself recognizes (`glm`, `kimi`, `gemma`, `qwen`, `nemotron`, `super`, `nano`, `minimax`, `hermes`, `gpt`, `llama`, `prime`, `deepseek`), `_models_helper.py` writes the **short form** to `settings.local.json` so the bottom statusline stays compact (e.g. `[nebius://kimi]`).

For helper-only shortcuts (`qwen-32`, `qwen-235`, `kimi-fast`, etc.) the proxy's alias map doesn't know the short name, so the helper writes the **full upstream id** instead and relies on the proxy's slash-passthrough rule (any `provider/model` id passes through verbatim).

## Updating the catalog

Nebius rotates model availability. When an id changes, edit **both** the static catalog in `models.md` and the `HARDCODED_SHORTCUTS` list in `_models_helper.py` (keep them in sync — the picker reads the static block, but `set <number>` resolves via the helper). No proxy restart needed; the next `/models` invocation picks up the change.
