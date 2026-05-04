# Claude Shell Function Configuration

This document describes the shell functions that enable easy switching between **direct** (subscription auth) and **proxy** (local Nebius proxy) connections.

## Quick Start

For zsh or bash, add the following to your `~/.zshrc` or `~/.bashrc`:

```bash
# Claude Shell Function — enables claude, claude --proxy, and claudius
claude() {
    local proxy_url="http://localhost:8083"

    if [[ "$1" == "--proxy" ]] || [[ "$1" == "claudius" ]]; then
        printf "\033[38;5;129m▐▛▜▌ Claude via Proxy\033[0m  \033[38;5;244m→ bearer auth via local proxy\033[0m\n"
        (
            unset ANTHROPIC_API_KEY
            export ANTHROPIC_AUTH_TOKEN="claude-local"
            export ANTHROPIC_BASE_URL="$proxy_url"
            command claude "${@:2}"
        )
    else
        printf "\033[38;5;46m▐▛▜▌ Claude Direct\033[0m  \033[38;5;244m→ subscription login auth\033[0m\n"
        (
            unset ANTHROPIC_BASE_URL ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN
            command claude "$@"
        )
    fi
}

# Alias for users who prefer claudius --proxy style
alias claudius='claude --proxy'
```

Then restart your shell or run: `source ~/.zshrc` (or `~/.bashrc`)

For PowerShell (`pwsh`), run `./install.sh` from PowerShell and accept the
prompt. The installer writes a native PowerShell function to `$PROFILE`.
Restart PowerShell or run:

```powershell
. $PROFILE
```

## Usage

| Command | Description |
|---------|-------------|
| `claude` | Direct connection using your subscription login |
| `claude --proxy` | Connect via local proxy (Nebius API) |
| `claude --proxy <prompt>` | Proxy connection with a prompt |
| `claudius` | Alias for `claude --proxy` |
| `claudius <prompt>` | Alias for `claude --proxy <prompt>` |

## Requirements

- **For direct mode**: Valid Claude subscription with login credentials
- **For proxy mode**: The proxy must be running (`python start_proxy.py` in the project directory)

## Visual Feedback

When you run a command, you'll see a colored indicator:

- **Green** (`▐▛▜▌ Claude Direct`) = Direct subscription connection
- **Purple** (`▐▛▜▌ Claude via Proxy`) = Local proxy connection (Nebius)

## Troubleshooting

### Proxy not running?

```bash
# Start the proxy
cd /path/to/claude-code-proxy
.venv/bin/python start_proxy.py
```

### Port different from 8083?

Edit the `proxy_url` variable in the function:

```bash
local proxy_url="http://localhost:9090"  # Your custom port
```

## Auto-Installation

The `install.sh` script can automatically configure this for zsh, bash, or
PowerShell. Run:

```bash
./install.sh
```

and it will prompt you to add the shell function to your profile.
