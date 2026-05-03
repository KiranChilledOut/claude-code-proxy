# Claude Shell Function Configuration

This document describes the shell functions that enable easy switching between **direct** (subscription auth) and **proxy** (local Nebius proxy) connections.

## Quick Start

Add the following to your `~/.zshrc` or `~/.bashrc`:

```bash
# Claude Shell Function — enables claude, claude --proxy, and claudius
claude() {
    local proxy_url="http://localhost:8083"

    if [[ "$1" == "--proxy" ]] || [[ "$1" == "claudius" ]]; then
        printf "\033[38;5;129m▐▛▜▌ Claude via Proxy\033[0m  \033[38;5;244m→ API key auth via local proxy\033[0m\n"
        ANTHROPIC_AUTH_TOKEN="tokenfactory" \
        ANTHROPIC_API_KEY="dummy" \
        ANTHROPIC_BASE_URL="$proxy_url" \
        command claude "${@:2}"
    else
        printf "\033[38;5;46m▐▛▜▌ Claude Direct\033[0m  \033[38;5;244m→ subscription login auth\033[0m\n"
        env -u ANTHROPIC_BASE_URL -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN \
        command claude "$@"
    fi
}

# Alias for users who prefer claudius --proxy style
alias claudius='claude --proxy'
```

Then restart your shell or run: `source ~/.zshrc` (or `~/.bashrc`)

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

The `install.sh` script can automatically configure this for you. Run:

```bash
./install.sh
```

and it will prompt you to add the shell function to your profile.