# `/models` slash command for Claude Code

A two-file Claude Code custom slash command that surfaces this proxy's full Token Factory shortcut catalog inside Claude Code (the built-in `/model` picker only shows the four hardcoded Anthropic entries plus the active custom model — see `docs/ARCHITECTURE.md` for why).

Typing `/models` renders a 30-entry combined list (curated shortcuts + any live catalog extras), takes a number or pasted id, and writes the choice to `~/.claude/settings.local.json` (the same file Claude Code's built-in `/model` writes to). The next request in your session uses the new model.

## Files

- **`models.md`** — the slash command body. Pinned to `model: glm` so the command itself runs on a model capable enough to follow tool-use instructions, regardless of what's currently selected. Tells the model to call `_models_helper.py list` and NOT to retype the bash output (see "Why bash output, not model text" below).
- **`_models_helper.py`** — companion script. Subcommands:
  - `list` — print the full numbered catalog (hardcoded shortcuts + any live extras) to stdout. The slash command runs this; Claude Code displays it as bash output (collapsed past ~3 lines with a `ctrl+o to expand` hint).
  - `set <id-or-number>` — writes the choice to `~/.claude/settings.local.json`. Resolves numbers (1-30), short names, and full ids.
  - `extras` — fetches `/v1/models` and prints only the upstream ids that aren't in the hardcoded list, one per line. Empty when the hardcoded list covers everything.

## Why bash output, not model text

Earlier iterations had the slash command body include the 30-entry catalog as static markdown so the model could "copy it verbatim." In practice even capable models (GLM-5, Kimi-K2.5) leaked memorized ids from training — `Kimi-K2.5` got rendered as `Kimi-K2`, `GLM-5` as `GLM-4.5`, `Qwen3-235B-A22B-Instruct-2507` as `…-2505`, etc. Picking by id from the model's text would have set the wrong model. Bash output is the only display that's guaranteed accurate, so the slash command uses it and accepts the collapse.

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

Nebius rotates model availability. When an id changes, edit `_models_helper.py`'s `HARDCODED_SHORTCUTS` list — single source of truth, no proxy restart needed. The next `/models` invocation picks up the change.
