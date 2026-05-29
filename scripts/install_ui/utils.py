from __future__ import annotations

import json
import os
import pathlib
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum


class ShellType(Enum):
    BASH = "bash"
    ZSH = "zsh"
    PWSH = "pwsh"
    UNKNOWN = "unknown"


@dataclass
class InstallState:
    python_version: str = ""
    has_pip: bool = False
    has_curl: bool = False
    api_key: str = ""
    port: int = 8083
    base_url: str = "https://api.tokenfactory.nebius.com/v1"
    big_model: str = ""
    middle_model: str = ""
    small_model: str = ""
    vision_model: str = ""
    shell_type: ShellType = ShellType.UNKNOWN
    shell_rc: str = ""
    configure_shell: bool = True
    configure_statusline: bool = True
    statusline_exists: bool = False
    venv_exists: bool = False
    deps_installed: bool = False
    models_fetched: bool = False
    available_models: list[str] = field(default_factory=list)
    smoke_test_passed: bool = False


def get_repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def detect_shell() -> tuple[ShellType, str]:
    parent = os.environ.get("PPID", "0")
    try:
        comm = subprocess.run(
            ["ps", "-p", parent, "-o", "comm="],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        parent_name = os.path.basename(comm)
    except Exception:
        parent_name = ""

    if parent_name in {"bash"}:
        return ShellType.BASH, os.path.expanduser("~/.bashrc")
    if parent_name in {"zsh"}:
        return ShellType.ZSH, os.path.expanduser("~/.zshrc")

    if shutil.which("pwsh"):
        try:
            result = subprocess.run(
                ["pwsh", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", "$PROFILE"],
                capture_output=True,
                text=True,
                check=False,
            )
            profile = result.stdout.strip().splitlines()[-1].strip()
            if profile:
                if parent_name in {"pwsh", "pwsh.exe", "powershell", "powershell.exe"}:
                    return ShellType.PWSH, profile
        except Exception:
            pass

    user_shell = os.environ.get("SHELL", "/bin/bash")
    if "zsh" in user_shell:
        return ShellType.ZSH, os.path.expanduser("~/.zshrc")
    if "bash" in user_shell:
        return ShellType.BASH, os.path.expanduser("~/.bashrc")

    if os.path.isfile(os.path.expanduser("~/.zshrc")):
        return ShellType.ZSH, os.path.expanduser("~/.zshrc")
    if os.path.isfile(os.path.expanduser("~/.bashrc")):
        return ShellType.BASH, os.path.expanduser("~/.bashrc")
    if shutil.which("pwsh"):
        try:
            result = subprocess.run(
                ["pwsh", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", "$PROFILE"],
                capture_output=True,
                text=True,
                check=False,
            )
            profile = result.stdout.strip().splitlines()[-1].strip()
            if profile:
                return ShellType.PWSH, profile
        except Exception:
            pass

    return ShellType.UNKNOWN, ""


def get_claude_settings_path() -> pathlib.Path:
    return pathlib.Path.home() / ".claude" / "settings.json"


def safe_merge_settings(statusline_command: str, repo_root: pathlib.Path) -> dict[str, str]:
    settings_path = get_claude_settings_path()
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    if settings_path.exists():
        existing = json.loads(settings_path.read_text(encoding="utf-8"))
        backup = settings_path.with_suffix(f".json.bak.{int(os.path.getmtime(settings_path))}")
        shutil.copy2(settings_path, backup)

        existing_statusline = existing.get("statusLine")
        if existing_statusline:
            existing_cmd = existing_statusline.get("command", "") if isinstance(existing_statusline, dict) else ""
            new_cmd = statusline_command
            if existing_cmd.strip() == new_cmd.strip():
                return {"action": "exists", "message": "statusLine already configured identically."}
            else:
                return {"action": "updated", "message": "statusLine exists with different value."}

        existing["statusLine"] = {"type": "command", "command": statusline_command}
        settings_path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
        return {"action": "added", "message": "Added statusLine configuration."}
    else:
        config = {"statusLine": {"type": "command", "command": statusline_command}}
        settings_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        return {"action": "created", "message": "Created settings.json with statusLine."}


def write_env(state: InstallState) -> None:
    """Merge .env.example as baseline, overlay existing .env, then apply state overrides."""
    repo = get_repo_root()
    env_path = repo / ".env"
    example_path = repo / ".env.example"

    # Start from .env.example as the baseline
    base_lines = example_path.read_text(encoding="utf-8").splitlines()

    # Parse existing .env into a dict of overrides (comment lines preserved)
    existing: dict[str, str] = {}
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                key, val = line.split("=", 1)
                existing[key] = val

    # What the TUI has collected
    state_overrides: dict[str, str] = {
        "OPENAI_API_KEY": state.api_key,
        "PORT": str(state.port),
        "BIG_MODEL": state.big_model,
        "MIDDLE_MODEL": state.middle_model,
        "SMALL_MODEL": state.small_model,
        "VISION_MODEL": state.vision_model,
    }

    # Build merged lines from the .env.example template
    merged: dict[str, str] = {}
    for line in base_lines:
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            key, val = line.split("=", 1)
            # Priority: TUI state > existing .env > .env.example default
            merged[key] = state_overrides.get(key, existing.get(key, val))

    # Ensure all state overrides are present
    for key, val in state_overrides.items():
        merged[key] = val

    # Re-assemble lines, preserving order from .env.example
    out_lines: list[str] = []
    seen = set()
    for line in base_lines:
        stripped = line.strip()
        if stripped.startswith("#") or stripped == "":
            out_lines.append(line)
            continue
        key, _ = stripped.split("=", 1)
        if key in merged and key not in seen:
            if " " in merged[key]:
                out_lines.append(f'{key}="{merged[key]}"')
            else:
                out_lines.append(f'{key}={merged[key]}')
            seen.add(key)

    # Append any remaining keys that weren't in the template
    for key, val in merged.items():
        if key not in seen:
            if " " in val:
                out_lines.append(f'{key}="{val}"')
            else:
                out_lines.append(f'{key}={val}')

    env_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    os.chmod(env_path, 0o600)


def sync_venv_create() -> tuple[bool, str]:
    """Create venv if missing. Returns (success, message)."""
    repo = get_repo_root()
    venv = repo / ".venv"
    if venv.exists():
        return True, ".venv already exists"
    result = subprocess.run(
        ["python3", "-m", "venv", str(venv)],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return True, ".venv created"
    return False, result.stderr.strip() or "Unknown error"


def sync_pip_install() -> tuple[bool, str]:
    """Install requirements into venv. Returns (success, message)."""
    repo = get_repo_root()
    pip = repo / ".venv" / "bin" / "pip"
    if platform.system() == "Windows":
        pip = repo / ".venv" / "Scripts" / "pip.exe"

    if not pip.exists():
        return False, f"pip not found at {pip}"

    # Upgrade pip first
    subprocess.run(
        [str(pip), "install", "--quiet", "--upgrade", "pip"],
        capture_output=True,
        check=False,
    )

    req = repo / "requirements.txt"
    result = subprocess.run(
        [str(pip), "install", "-q", "-r", str(req)],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return True, "Dependencies installed"
    return False, result.stderr.strip() or "pip install failed"


def fetch_nebius_models(api_key: str, base_url: str) -> dict:
    """Fetch available models from Nebius."""
    import ssl
    import urllib.request

    endpoint = base_url.rstrip("/") + "/models"
    req = urllib.request.Request(
        endpoint, headers={"Authorization": f"Bearer {api_key}"}
    )
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
            data = json.load(resp)
            models = [m["id"] for m in data.get("data", [])]
            return {"ok": True, "models": models}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def pick_default_models(available: list[str]) -> dict[str, str]:
    """Return {BIG_MODEL, MIDDLE_MODEL, SMALL_MODEL, VISION_MODEL} from available."""

    def pick(candidates: list[str]) -> str:
        for c in candidates:
            if c in available:
                return c
        return available[0] if available else ""

    return {
        "BIG_MODEL": pick([
            "deepseek-ai/DeepSeek-V4-Pro",
            "Qwen/Qwen3-235B-A22B-Instruct-2507",
            "meta-llama/Llama-3.3-70B-Instruct",
            "moonshotai/Kimi-K2.6",
        ]),
        "MIDDLE_MODEL": pick([
            "deepseek-ai/DeepSeek-V3.2",
            "Qwen/Qwen3-235B-A22B-Instruct-2507",
            "meta-llama/Llama-3.3-70B-Instruct",
            "moonshotai/Kimi-K2.6",
        ]),
        "SMALL_MODEL": pick([
            "deepseek-ai/DeepSeek-V3.2",
            "Qwen/Qwen3-32B",
            "meta-llama/Llama-3.3-70B-Instruct",
            "moonshotai/Kimi-K2.6",
        ]),
        "VISION_MODEL": pick([
            "Qwen/Qwen2.5-VL-72B-Instruct",
            "Qwen/Qwen3-235B-A22B-Instruct-2507",
            "moonshotai/Kimi-K2.6",
        ]),
    }


def shell_function_is_present(shell_type: ShellType, rc_path: str) -> bool:
    """Check if the claude shell function already exists in the profile."""
    if not rc_path or not os.path.isfile(rc_path):
        return False
    content = pathlib.Path(rc_path).read_text(encoding="utf-8")
    if shell_type == ShellType.PWSH:
        return "function claude" in content or "function global:claude" in content
    return "claude() {" in content


def append_shell_function(
    shell_type: ShellType,
    rc_path: str,
    port: int,
    repo_root: pathlib.Path,
) -> bool:
    """Append the convenience shell function to the user's profile."""
    if not rc_path:
        return False
    os.makedirs(os.path.dirname(rc_path), exist_ok=True)
    if not os.path.isfile(rc_path):
        pathlib.Path(rc_path).touch()

    # Backup
    backup = f"{rc_path}.bak.{int(os.path.getmtime(rc_path))}"
    shutil.copy2(rc_path, backup)

    if shell_type == ShellType.PWSH:
        _append_pwsh(rc_path, port, repo_root)
    else:
        _append_bash_zsh(rc_path, port, repo_root)

    return True


def _append_bash_zsh(rc_path: str, port: int, repo_root: pathlib.Path) -> None:
    func = f"""
# Claude Shell Function — enables claude, claude --proxy, and claudius
claude() {{
    local main_proxy="http://localhost:{port}"
    local repo_root="{repo_root}"
    if [[ "$1" == "--proxy" ]]; then
        printf "\\033[38;5;129m▐▛▜▌ Claude via Proxy\\033[0m  \\033[38;5;244m→ bearer auth via local proxy\\033[0m\\n"
        local default_name="session-$(date +%Y%m%d-%H%M%S)"
        printf "\\033[38;5;244mSession name\\033[0m [\\033[38;5;75m%s\\033[0m]: " "$default_name"
        read -r session_name
        session_name="${{session_name:-$default_name}}"
        local local_port
        local_port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
        mkdir -p "$repo_root/logs"
        python3 "$repo_root/scripts/session_forwarder.py" "$local_port" "localhost:{port}" "$session_name" >> "$repo_root/logs/session-forwarder.log" 2>&1 &
        local forwarder_pid=$!
        sleep 0.5
        local forwarder_url="http://localhost:$local_port"
        (
            unset ANTHROPIC_API_KEY
            export ANTHROPIC_AUTH_TOKEN="claude-local"
            export ANTHROPIC_BASE_URL="$forwarder_url"
            command claude "${{@:2}}"
        )
        local claude_exit=$?
        kill "$forwarder_pid" 2>/dev/null || true
        wait "$forwarder_pid" 2>/dev/null || true
        return $claude_exit
    else
        printf "\\033[38;5;46m▐▛▜▌ Claude Direct\\033[0m  \\033[38;5;244m→ subscription login auth\\033[0m\\n"
        (
            unset ANTHROPIC_BASE_URL ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN
            command claude "$@"
        )
    fi
}}
alias claudius='claude --proxy'
"""
    with open(rc_path, "a", encoding="utf-8") as f:
        f.write(func)


def _append_pwsh(rc_path: str, port: int, repo_root: pathlib.Path) -> None:
    func = f"""
# Claude Shell Function - enables claude, claude --proxy, and claudius
function claude {{
    param([Parameter(ValueFromRemainingArguments = $true)] [string[]] $ClaudeArgs)
    $mainProxy = "http://localhost:{port}"
    $repoRoot = "{repo_root}"
    $claudeCommand = (Get-Command claude -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
    $oldAuthToken = $env:ANTHROPIC_AUTH_TOKEN
    $oldApiKey = $env:ANTHROPIC_API_KEY
    $oldBaseUrl = $env:ANTHROPIC_BASE_URL
    if ($ClaudeArgs.Count -gt 0 -and $ClaudeArgs[0] -eq "--proxy") {{
        Write-Host "`e[38;5;129m▐▛▜▌ Claude via Proxy`e[0m  `e[38;5;244m-> bearer auth via local proxy`e[0m"
        $defaultName = "session-" + (Get-Date -Format "yyyyMMdd-HHmmss")
        Write-Host "Session name [`e[38;5;75m$defaultName`e[0m]: " -NoNewline
        [string] $sessionName = Read-Host
        if ([string]::IsNullOrWhiteSpace($sessionName)) {{ $sessionName = $defaultName }}
        [int] $localPort = python3 -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1]); s.close()"
        $forwarderJob = Start-Job -ScriptBlock {{
            param($port, $target, $name, $repo)
            python3 "$repo/scripts/session_forwarder.py" $port $target $name
        }} -ArgumentList $localPort, "localhost:{port}", $sessionName, $repoRoot
        Start-Sleep -Milliseconds 800
        [string[]] $remainingArgs = @()
        if ($ClaudeArgs.Count -gt 1) {{
            $remainingArgs = [string[]] $ClaudeArgs[1..($ClaudeArgs.Count - 1)]
        }}
        $forwarderUrl = "http://localhost:$localPort"
        try {{
            $env:ANTHROPIC_AUTH_TOKEN = "claude-local"
            Remove-Item Env:ANTHROPIC_API_KEY -ErrorAction SilentlyContinue
            $env:ANTHROPIC_BASE_URL = $forwarderUrl
            & $claudeCommand @remainingArgs
        }} finally {{
            if ($forwarderJob) {{ Stop-Job $forwarderJob -ErrorAction SilentlyContinue; Remove-Job $forwarderJob -ErrorAction SilentlyContinue }}
            if ($null -eq $oldAuthToken) {{ Remove-Item Env:ANTHROPIC_AUTH_TOKEN -ErrorAction SilentlyContinue }} else {{ $env:ANTHROPIC_AUTH_TOKEN = $oldAuthToken }}
            if ($null -eq $oldApiKey) {{ Remove-Item Env:ANTHROPIC_API_KEY -ErrorAction SilentlyContinue }} else {{ $env:ANTHROPIC_API_KEY = $oldApiKey }}
            if ($null -eq $oldBaseUrl) {{ Remove-Item Env:ANTHROPIC_BASE_URL -ErrorAction SilentlyContinue }} else {{ $env:ANTHROPIC_BASE_URL = $oldBaseUrl }}
        }}
    }} else {{
        Write-Host "`e[38;5;46m▐▛▜▌ Claude Direct`e[0m  `e[38;5;244m-> subscription login auth`e[0m"
        try {{
            Remove-Item Env:ANTHROPIC_AUTH_TOKEN -ErrorAction SilentlyContinue
            Remove-Item Env:ANTHROPIC_API_KEY -ErrorAction SilentlyContinue
            Remove-Item Env:ANTHROPIC_BASE_URL -ErrorAction SilentlyContinue
            & $claudeCommand @ClaudeArgs
        }} finally {{
            if ($null -ne $oldAuthToken) {{ $env:ANTHROPIC_AUTH_TOKEN = $oldAuthToken }}
            if ($null -ne $oldApiKey) {{ $env:ANTHROPIC_API_KEY = $oldApiKey }}
            if ($null -ne $oldBaseUrl) {{ $env:ANTHROPIC_BASE_URL = $oldBaseUrl }}
        }}
    }}
}}
function claudius {{
    param([Parameter(ValueFromRemainingArguments = $true)] [string[]] $ClaudeArgs)
    claude --proxy @ClaudeArgs
}}
"""
    with open(rc_path, "a", encoding="utf-8") as f:
        f.write(func)
