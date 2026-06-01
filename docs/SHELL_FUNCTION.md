# Claude Shell Function Reference

The TUI installer (`./install.sh`) can automatically append the most recent version of these functions to your shell profile (`~/.zshrc`, `~/.bashrc`, or PowerShell `$PROFILE`).

What the installed function does:
- `claude` — direct subscription login (no proxy)
- `claude --proxy` — routes through the local Nebius proxy via a per-session forwarder
- `claudius` — alias for `claude --proxy`

## Usage

| Command | Description |
|---------|-------------|
| `claude` | Direct Claude Code (subscription login) |
| `claude --proxy` | Proxy mode via Nebius with session forwarder |
| `claude --proxy <dir>` | Proxy mode starting in a specific directory |
| `claudius` | Alias for `claude --proxy` |

## Session Forwarder

The installed bash/zsh function uses `scripts/session_forwarder.py` to spin up a temporary forwarder on a random free port for each proxy session. This gives the statusline independent per-session metrics. When Claude Code exits, the forwarder is cleaned up automatically.

The PowerShell function uses `Start-Job` for equivalent behaviour.

Forwarder output (including network errors) is written to `logs/session-forwarder.log` so it does not appear in the Claude Code TUI.

## Visual Feedback

- **Green** (`▐▛▜▌ Claude Direct`) = Subscription login mode
- **Purple** (`▐▛▜▌ Claude via Proxy`) = Proxy mode via Nebius

## Troubleshooting

### Proxy not running?

```bash
cd /path/to/claude-code-proxy
.venv/bin/python start_proxy.py
```

### Session-forwarder errors in the TUI?

If you see errors like `[forwarder] request forwarding failed` inside Claude Code, they are expected during normal operation (brief upstream disconnects). They are redirected to `logs/session-forwarder.log` by the installed shell function. If they appear in the TUI, re-run `./install.sh` to get the latest shell function.

### Port different from 8083?

Re-run `./install.sh` and enter your custom port on the API Key & Port step.
