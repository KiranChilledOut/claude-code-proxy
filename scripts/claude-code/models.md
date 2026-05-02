---
description: List all Token Factory models with their shortnames; pick one to switch
argument-hint: [optional: model id, shortname, or list number]
model: glm
allowed-tools: Bash
---

Switch the active Claude Code model to a Token Factory model. The helper at `~/.claude/commands/_models_helper.py` does the actual write to `~/.claude/settings.local.json`.

## With argument

If `$ARGUMENTS` is non-empty, run exactly:

```
python3 ~/.claude/commands/_models_helper.py set "$ARGUMENTS"
```

Then tell the user in one short sentence: `Set model to <id>; next request in this session will route through it.`

If the helper exits non-zero, report what it printed and stop.

## Without argument

Step 1 — print the catalog below verbatim as your text response. Every line, exactly as shown. Do not abbreviate, do not re-order, do not invent ids. This is your authoritative reference; the user picks from this list.

```
Token Factory shortcuts:
  [ 1] glm             -> zai-org/GLM-5
  [ 2] kimi            -> moonshotai/Kimi-K2.5
  [ 3] qwen            -> Qwen/Qwen3.5-397B-A17B
  [ 4] nemotron        -> nvidia/Llama-3_1-Nemotron-Ultra-253B-v1
  [ 5] hermes          -> NousResearch/Hermes-4-405B
  [ 6] deepseek        -> deepseek-ai/DeepSeek-V3.2
  [ 7] minimax         -> MiniMaxAI/MiniMax-M2.5
  [ 8] prime           -> PrimeIntellect/INTELLECT-3
  [ 9] gpt             -> openai/gpt-oss-120b
  [10] gemma           -> google/gemma-3-27b-it
  [11] gemma-tiny      -> google/gemma-2-2b-it
  [12] llama           -> meta-llama/Meta-Llama-3.1-8B-Instruct
  [13] llama-big       -> meta-llama/Llama-3.3-70B-Instruct
  [14] super           -> nvidia/nemotron-3-super-120b-a12b
  [15] nano            -> nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B
  [16] omni            -> nvidia/Nemotron-3-Nano-Omni
  [17] hermes-small    -> NousResearch/Hermes-4-70B
  [18] qwen-235        -> Qwen/Qwen3-235B-A22B-Instruct-2507
  [19] qwen-235-think  -> Qwen/Qwen3-235B-A22B-Thinking-2507-fast
  [20] qwen-32         -> Qwen/Qwen3-32B
  [21] qwen-30         -> Qwen/Qwen3-30B-A3B-Instruct-2507
  [22] qwen-next       -> Qwen/Qwen3-Next-80B-A3B-Thinking
  [23] qwen-vl         -> Qwen/Qwen2.5-VL-72B-Instruct
  [24] qwen-embed      -> Qwen/Qwen3-Embedding-8B
  [25] kimi-fast       -> moonshotai/Kimi-K2.5-fast
  [26] qwen-fast       -> Qwen/Qwen3.5-397B-A17B-fast
  [27] qwen-next-fast  -> Qwen/Qwen3-Next-80B-A3B-Thinking-fast
  [28] deepseek-fast   -> deepseek-ai/DeepSeek-V3.2-fast
  [29] gpt-fast        -> openai/gpt-oss-120b-fast
  [30] minimax-fast    -> MiniMaxAI/MiniMax-M2.5-fast
```

Step 2 — run exactly:

```
python3 ~/.claude/commands/_models_helper.py extras
```

If the bash output has any non-empty lines, append them to your text response under a new heading `Live catalog (paste the id to pick):` with each id as a bullet (`  - <id>`). If the bash output is empty or fails, skip this step entirely (do not mention it).

Step 3 — on a new line below everything else, ask exactly: `Type a number from the shortcuts list, or paste any model id.`

Step 4 — wait for the user's reply, then run exactly:

```
python3 ~/.claude/commands/_models_helper.py set "<their-reply-verbatim>"
```

The helper accepts numbers (1-30), shortnames (`glm`, `qwen-32`, etc.), and full ids. Then confirm: `Set model to <id-from-helper-output>; next request in this session will route through it.`

## Rules

- Do not modify `~/.claude/settings.json`; the helper writes only `.local.json`.
- Never invent or paraphrase a model id. If you can't recall an id exactly, look at the catalog block above — it is your authoritative reference.
- Two bash invocations max in step-2 + step-4 path (`extras`, then `set`). Never construct an inline curl or python one-liner.
