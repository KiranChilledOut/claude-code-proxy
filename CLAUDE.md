# Claude Code Project Notes

This repository is organized around one concern:

- `src/`: the Nebius-focused Claude API proxy

## Read First

Before making non-trivial changes, prefer the tracked reference docs in `docs/`:

- `docs/ARCHITECTURE.md`
- `docs/TOOL_CALL_FORMAT.md`
- `docs/BINARY_PACKAGING.md`

## Working Conventions

- Read target files before editing them.
- Prefer focused, surgical changes over speculative refactors.
- Keep public documentation in tracked files under `docs/`.
- Do not depend on ignored local-only state such as `.claude/`.

## Repository Layout

- `src/`: proxy implementation
- `tests/`: automated tests
- `docs/`: tracked project documentation
- `start_proxy.py`: local convenience launcher

## Current Focus Areas

- Tool-call streaming and argument sanitization are core integration paths.

---

## Coding Guidelines

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" -> "Write tests for invalid inputs, then make them pass"
- "Fix the bug" -> "Write a test that reproduces it, then make it pass"
- "Refactor X" -> "Ensure tests pass before and after"

For multi-step tasks, state a brief plan.

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.
