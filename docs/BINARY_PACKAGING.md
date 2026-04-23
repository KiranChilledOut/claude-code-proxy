# Binary Packaging

This project can be packaged as a standalone executable with PyInstaller.

## Requirements

- Python 3.9+
- `uv` optional but recommended
- `pyinstaller`

Install the packager:

```bash
uv add --dev pyinstaller
```

## Basic Packaging

Directory-style build:

```bash
uv run pyinstaller claude-proxy.spec
```

Single-file build:

```bash
uv run pyinstaller --onefile --name claude-code-proxy-single src/main.py
```

## Cross-Platform Note

Build binaries on the target operating system:

- Linux builds on Linux
- macOS builds on macOS
- Windows builds on Windows

## Runtime Configuration

The packaged binary still expects environment variables such as:

```bash
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.tokenfactory.nebius.com/v1
HOST=0.0.0.0
PORT=8083
```

## Deployment Options

- direct binary execution
- background service / systemd
- Docker container wrapping the packaged binary
