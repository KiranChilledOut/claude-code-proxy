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

## Visual Feedback

- **Green** (`▐▛▜▌ Claude Direct`) = Subscription login mode
- **Purple** (`▐▛▜▌ Claude via Proxy`) = Proxy mode via Nebius

## Troubleshooting

### Proxy not running?

```bash
cd /path/to/claude-code-proxy
.venv/bin/python start_proxy.py
```

### Port different from 8083?

Re-run `./install.sh` and enter your custom port on the API Key & Port step.
