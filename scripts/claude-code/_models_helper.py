#!/usr/bin/env python3
"""Backend for the /models slash command.

Subcommands:
  list           Print a single combined numbered list — hardcoded shortcuts
                 first (in the curated order below), then any live catalog
                 entries that aren't already represented by a shortcut.
  set <id|int>   Write the chosen model id to ~/.claude/settings.local.json.
                 If <int>, look up the corresponding entry from the same
                 combined ordering as `list`. If the chosen entry has a
                 shortcut that the proxy itself knows about (PROXY_KNOWN_ALIASES),
                 we write the short name so the bottom statusline stays clean;
                 otherwise we write the full upstream id and rely on the
                 proxy's slash-passthrough rule to route it.

Why hardcode the catalog: the picker should work even if the proxy is
unreachable, and editing this file is faster than rebuilding the proxy
when Nebius rotates a model id. The live `/v1/models` fetch only adds
entries we don't already know about.
"""

import json
import os
import sys
import urllib.request

def settings_path() -> str:
    """Where to write the model selection.

    Priority order:
      1. CLAUDE_CODE_PROXY_DIR (explicit pin) -- write to that dir's
         .claude/settings.local.json. Useful when a single proxy dir
         should own all picker state regardless of where claudius was
         launched from.
      2. ANTHROPIC_BASE_URL set (we're in a proxy session, e.g. via
         the upstream `claude --proxy` / `claudius` wrapper from
         docs/SHELL_FUNCTION.md) -- scope to cwd/.claude/ so picks
         don't leak into bare `claude` invocations elsewhere on the
         machine. (Bare claude in the same cwd will still see them;
         that's a per-project leak rather than a global one.)
      3. Fallback: user-level ~/.claude/settings.local.json. Only
         reached when there's no signal we're proxying at all.
    """
    proxy_dir = os.environ.get("CLAUDE_CODE_PROXY_DIR")
    if proxy_dir:
        claude_dir = os.path.join(os.path.expanduser(proxy_dir), ".claude")
    elif os.environ.get("ANTHROPIC_BASE_URL"):
        claude_dir = os.path.join(os.getcwd(), ".claude")
    else:
        return os.path.expanduser("~/.claude/settings.local.json")
    os.makedirs(claude_dir, exist_ok=True)
    return os.path.join(claude_dir, "settings.local.json")

# Curated shortname -> full upstream id, in display order. Each entry must
# point at an id that's actually live on Nebius's Token Factory; if a name
# rotates, update this list (no proxy restart needed).
HARDCODED_SHORTCUTS = [
    # General-purpose flagships
    ("glm",            "zai-org/GLM-5"),
    ("kimi",           "moonshotai/Kimi-K2.5"),
    ("qwen",           "Qwen/Qwen3.5-397B-A17B"),
    ("nemotron",       "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1"),
    ("hermes",         "NousResearch/Hermes-4-405B"),
    ("deepseek",       "deepseek-ai/DeepSeek-V3.2"),
    ("minimax",        "MiniMaxAI/MiniMax-M2.5"),
    ("prime",          "PrimeIntellect/INTELLECT-3"),
    ("gpt",            "openai/gpt-oss-120b"),
    # Smaller / specialized
    ("gemma",          "google/gemma-3-27b-it"),
    ("gemma-tiny",     "google/gemma-2-2b-it"),
    ("llama",          "meta-llama/Meta-Llama-3.1-8B-Instruct"),
    ("llama-big",      "meta-llama/Llama-3.3-70B-Instruct"),
    ("super",          "nvidia/nemotron-3-super-120b-a12b"),
    ("nano",           "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"),
    ("omni",           "nvidia/Nemotron-3-Nano-Omni"),
    ("hermes-small",   "NousResearch/Hermes-4-70B"),
    # Qwen variants
    ("qwen-235",       "Qwen/Qwen3-235B-A22B-Instruct-2507"),
    ("qwen-235-think", "Qwen/Qwen3-235B-A22B-Thinking-2507-fast"),
    ("qwen-32",        "Qwen/Qwen3-32B"),
    ("qwen-30",        "Qwen/Qwen3-30B-A3B-Instruct-2507"),
    ("qwen-next",      "Qwen/Qwen3-Next-80B-A3B-Thinking"),
    ("qwen-vl",        "Qwen/Qwen2.5-VL-72B-Instruct"),
    ("qwen-embed",     "Qwen/Qwen3-Embedding-8B"),
    # Speed-tuned variants
    ("kimi-fast",      "moonshotai/Kimi-K2.5-fast"),
    ("qwen-fast",      "Qwen/Qwen3.5-397B-A17B-fast"),
    ("qwen-next-fast", "Qwen/Qwen3-Next-80B-A3B-Thinking-fast"),
    ("deepseek-fast",  "deepseek-ai/DeepSeek-V3.2-fast"),
    ("gpt-fast",       "openai/gpt-oss-120b-fast"),
    ("minimax-fast",   "MiniMaxAI/MiniMax-M2.5-fast"),
]

# Aliases the proxy recognizes natively (per src/core/model_manager.py in
# PR #24). For these we write the short name to settings.local.json so the
# bottom statusline shows e.g. `[nebius://glm]` instead of the full id.
# For helper-only shortcuts (everything else), we write the full id.
PROXY_KNOWN_ALIASES = {
    "glm", "kimi", "gemma", "qwen", "nemotron", "super", "nano",
    "minimax", "hermes", "gpt", "llama", "prime", "deepseek",
}


def fetch_catalog_quietly():
    """Return the live /v1/models data list, or [] on any failure."""
    base = os.environ.get("ANTHROPIC_BASE_URL", "http://localhost:8083")
    req = urllib.request.Request(
        f"{base}/v1/models",
        headers={"x-api-key": os.environ.get("ANTHROPIC_API_KEY", "claude-local")},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.load(r).get("data", [])
    except Exception:
        return []


def live_extras(catalog_data):
    """Return upstream ids from the live catalog that aren't already pointed
    at by a hardcoded shortcut. Skips alias-style entries (where the entry's
    `backend_model` differs from its `id`)."""
    hardcoded_full_ids = {full for _, full in HARDCODED_SHORTCUTS}
    extras = []
    seen = set()
    for m in catalog_data:
        mid = m.get("id")
        if not mid:
            continue
        backend = m.get("backend_model")
        if backend and backend != mid:
            # alias-style row from the proxy — represented in our shortcuts
            continue
        if mid in hardcoded_full_ids or mid in seen:
            continue
        seen.add(mid)
        extras.append(mid)
    return extras


def build_combined_ordering():
    """Same ordering used by both `list` and numeric `set` lookup."""
    extras = live_extras(fetch_catalog_quietly())
    combined = [(short, full) for short, full in HARDCODED_SHORTCUTS]
    combined.extend((None, full) for full in extras)
    return combined


def cmd_list():
    combined = build_combined_ordering()
    width = max(len(s) for s, _ in combined if s)

    i = 1
    for short, full in combined:
        if short:
            print(f"  [{i:2d}] {short:<{width}s}  -> {full}")
        else:
            # First "no shortcut" row gets a divider above it.
            if i > 1 and combined[i - 2][0] is not None:
                print()
                print("Live catalog (no shortcut):")
            print(f"  [{i:2d}] {' ' * width}     {full}")
        i += 1


def cmd_set(value):
    value = value.strip()
    short_to_full = {s: f for s, f in HARDCODED_SHORTCUTS}

    # Numeric: index into combined ordering
    if value.isdigit():
        combined = build_combined_ordering()
        idx = int(value) - 1
        if not (0 <= idx < len(combined)):
            print(f"INDEX_OUT_OF_RANGE: {value} (have 1..{len(combined)})", file=sys.stderr)
            sys.exit(4)
        short, full = combined[idx]
        write_value = short if (short and short in PROXY_KNOWN_ALIASES) else full

    # Shortname: resolve via hardcoded mapping
    elif value in short_to_full:
        write_value = value if value in PROXY_KNOWN_ALIASES else short_to_full[value]

    # Otherwise: verbatim (full id or arbitrary)
    else:
        write_value = value

    try:
        with open(settings_path()) as f:
            text = f.read()
            settings = json.loads(text) if text.strip() else {}
    except FileNotFoundError:
        settings = {}
    settings["model"] = write_value
    with open(settings_path(), "w") as f:
        json.dump(settings, f, indent=2)
        f.write("\n")
    print(f"settings.local.json updated: model = {write_value}")


def cmd_extras():
    """Print only the live catalog ids that aren't already represented by a
    hardcoded shortcut, one per line, no decoration. Empty output if the
    proxy is unreachable or there are no extras."""
    catalog = fetch_catalog_quietly()
    for full in live_extras(catalog):
        print(full)


def main():
    if len(sys.argv) < 2:
        print("usage: _models_helper.py {list | extras | set <id-or-number>}", file=sys.stderr)
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "list":
        cmd_list()
    elif cmd == "extras":
        cmd_extras()
    elif cmd == "set":
        if len(sys.argv) < 3:
            print("usage: _models_helper.py set <id-or-number>", file=sys.stderr)
            sys.exit(1)
        cmd_set(sys.argv[2])
    else:
        print(f"unknown subcommand: {cmd}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
