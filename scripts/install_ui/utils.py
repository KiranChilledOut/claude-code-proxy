from __future__ import annotations

import json
import os
import re
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


class ClientChoice(Enum):
    CLAUDE = "claude"
    CODEX = "codex"


@dataclass
class InstallState:
    client: ClientChoice = ClientChoice.CLAUDE
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


def _get_pwsh_profile() -> str:
    """Return pwsh profile path if pwsh is available, else empty string."""
    if not shutil.which("pwsh"):
        return ""
    try:
        result = subprocess.run(
            ["pwsh", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", "$PROFILE"],
            capture_output=True,
            text=True,
            check=False,
        )
        lines = [ln.strip() for ln in result.stdout.strip().splitlines() if ln.strip()]
        if lines:
            return lines[-1]
    except Exception:
        pass
    return ""


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

    # 1. If parent IS bash/zsh → use that (TUI was launched from that shell)
    if parent_name in {"bash"}:
        return ShellType.BASH, os.path.expanduser("~/.bashrc")
    if parent_name in {"zsh"}:
        return ShellType.ZSH, os.path.expanduser("~/.zshrc")
    if parent_name in {"pwsh", "pwsh.exe", "powershell", "powershell.exe"}:
        pwsh_profile = _get_pwsh_profile()
        if pwsh_profile:
            return ShellType.PWSH, pwsh_profile

    # 2. Fall back: if pwsh is installed AND has a real profile, prefer it
    #    (many macOS users run TUI from zsh but primarily use pwsh)
    pwsh_profile = _get_pwsh_profile()
    if pwsh_profile:
        return ShellType.PWSH, pwsh_profile

    # 3. Fall back to user's login shell
    user_shell = os.environ.get("SHELL", "/bin/bash")
    if "zsh" in user_shell:
        return ShellType.ZSH, os.path.expanduser("~/.zshrc")
    if "bash" in user_shell:
        return ShellType.BASH, os.path.expanduser("~/.bashrc")

    # 4. Last resort — any existing RC file
    if os.path.isfile(os.path.expanduser("~/.zshrc")):
        return ShellType.ZSH, os.path.expanduser("~/.zshrc")
    if os.path.isfile(os.path.expanduser("~/.bashrc")):
        return ShellType.BASH, os.path.expanduser("~/.bashrc")
    if pwsh_profile:
        return ShellType.PWSH, pwsh_profile

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
    req = repo / "requirements.txt"
    venv = repo / ".venv"
    python = venv / "bin" / "python"
    if platform.system() == "Windows":
        python = venv / "Scripts" / "python.exe"

    # ─── uv-managed venvs have no pip binary — use uv directly ───────
    uv_path = shutil.which("uv")
    pip3 = venv / "bin" / "pip3"
    pip_bin = venv / "bin" / "pip"
    if platform.system() == "Windows":
        pip3 = venv / "Scripts" / "pip3.exe"
        pip_bin = venv / "Scripts" / "pip.exe"

    has_pip = pip_bin.exists() or pip3.exists()
    if uv_path and not has_pip:
        # Try uv pip install --python .venv/bin/python
        result = subprocess.run(
            [uv_path, "pip", "install", "--python", str(python), "-r", str(req)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return True, "Dependencies installed (via uv)"
        return False, result.stderr.strip() or "uv pip install failed"

    if not has_pip:
        # Bootstrap pip into the venv
        bootstrap = subprocess.run(
            [str(python), "-m", "ensurepip", "--upgrade"],
            capture_output=True,
            text=True,
        )
        if bootstrap.returncode != 0:
            return False, f"pip not found and bootstrap failed: {bootstrap.stderr.strip() or 'no stderr'}"
        # Re-check after bootstrap
        pip3 = venv / "bin" / "pip3"
        pip_bin = venv / "bin" / "pip"
        has_pip = pip_bin.exists() or pip3.exists()
        if not has_pip:
            return False, "pip still missing after ensurepip"

    # Use python -m pip (works regardless of whether it's pip or pip3)
    result = subprocess.run(
        [str(python), "-m", "pip", "install", "--quiet", "-r", str(req)],
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
    """Check if BOTH claude and codex shell functions already exist in the profile.

    Returns True only if all four shortcuts are found (claude + claudius + codex + codexius),
    so that adding codex to an old claude-only install still proceeds.
    """
    if not rc_path or not os.path.isfile(rc_path):
        return False
    content = pathlib.Path(rc_path).read_text(encoding="utf-8")
    if shell_type == ShellType.PWSH:
        has_claude = "function claude" in content or "function global:claude" in content
        has_claudius = "function claudius" in content or "function global:claudius" in content
        has_codex = "function codex" in content or "function global:codex" in content
        has_codexius = "function codexius" in content or "function global:codexius" in content
        return has_claude and has_claudius and has_codex and has_codexius
    has_claude = "claude() {" in content
    has_claudius = "alias claudius=" in content
    has_codex = "codex() {" in content
    has_codexius = "alias codexius=" in content
    return has_claude and has_claudius and has_codex and has_codexius


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

# Codex Shell Function — enables codex, codex --proxy, and codexius
codex() {{
    local repo_root="{repo_root}"
    if [[ "$1" == "--proxy" ]]; then
        local api_key="${{OPENAI_API_KEY:-}}"
        if [[ -z "$api_key" && -f "$repo_root/.env" ]]; then
            api_key=$(grep -E '^OPENAI_API_KEY=' "$repo_root/.env" | head -1 | cut -d= -f2- | tr -d '" \t' 2>/dev/null || true)
        fi
        if [[ -z "$api_key" ]]; then
            printf "\\033[38;5;197m▐▛▜▌ Error\\033[0m  \\033[38;5;244m→ OPENAI_API_KEY not set (export or add to $repo_root/.env)\\033[0m\\n" >&2
            return 1
        fi
        export OPENAI_API_KEY="$api_key"
        printf "\\033[38;5;129m▐▛▜▌ Codex via Proxy\\033[0m  \\033[38;5;244m→ Nebius backend\\033[0m\\n"
        command codex "${{@:2}}"
    else
        printf "\\033[38;5;46m▐▛▜▌ Codex Direct\\033[0m  \\033[38;5;244m→ standard auth\\033[0m\\n"
        command codex "$@"
    fi
}}
alias codexius='codex --proxy'
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

# Codex Shell Function - enables codex, codex --proxy, and codexius
function codex {{
    param([Parameter(ValueFromRemainingArguments = $true)] [string[]] $CodexArgs)
    $repoRoot = "{repo_root}"
    $codexCommand = (Get-Command codex -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
    if ($CodexArgs.Count -gt 0 -and $CodexArgs[0] -eq "--proxy") {{
        $apiKey = $env:OPENAI_API_KEY
        if (-not $apiKey -and (Test-Path "$repoRoot/.env")) {{
            $line = Select-String -Path "$repoRoot/.env" -Pattern "^OPENAI_API_KEY=" | Select-Object -First 1
            if ($line) {{ $apiKey = ($line.Line -replace '^OPENAI_API_KEY=', '').Trim('"', ' ', "`t") }}
        }}
        if (-not $apiKey) {{
            Write-Host "`e[38;5;197m▐▛▜▌ Error`e[0m  `e[38;5;244m-> OPENAI_API_KEY not set (set env var or add to $repoRoot/.env)`e[0m" -ForegroundColor Red
            return
        }}
        $env:OPENAI_API_KEY = $apiKey
        Write-Host "`e[38;5;129m▐▛▜▌ Codex via Proxy`e[0m  `e[38;5;244m-> Nebius backend`e[0m"
        $remainingArgs = [string[]] @()
        if ($CodexArgs.Count -gt 1) {{
            $remainingArgs = [string[]] $CodexArgs[1..($CodexArgs.Count - 1)]
        }}
        & $codexCommand @remainingArgs
    }} else {{
        Write-Host "`e[38;5;46m▐▛▜▌ Codex Direct`e[0m  `e[38;5;244m-> standard auth`e[0m"
        & $codexCommand @CodexArgs
    }}
}}
function codexius {{
    param([Parameter(ValueFromRemainingArguments = $true)] [string[]] $CodexArgs)
    codex --proxy @CodexArgs
}}
"""
    with open(rc_path, "a", encoding="utf-8") as f:
        f.write(func)


# ── Codex Config ──

def get_codex_config_path() -> pathlib.Path | None:
    """Return the existing Codex config file (.toml preferred, falls back to .json)."""
    codex_dir = pathlib.Path.home() / ".codex"
    toml_path = codex_dir / "config.toml"
    json_path = codex_dir / "config.json"
    if toml_path.exists():
        return toml_path
    if json_path.exists():
        return json_path
    return toml_path  # Prefer .toml for new files


def _normalize_model_for_codex(model: str) -> str:
    """Codex expects nebius/<model> when using a custom provider."""
    model = model.strip()
    if model.startswith("nebius/"):
        return model
    return f"nebius/{model}"


def write_codex_config(
    big_model: str,
    proxy_port: int,
    repo_root: pathlib.Path,
) -> dict:
    """Write/update ~/.codex/config.toml (or .json) with proxy routing.

    Returns a dict with keys: action, message, path.
    """
    config_path = get_codex_config_path()
    if config_path is None:
        return {"action": "error", "message": "Could not locate Codex config directory", "path": ""}

    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve actual env value from .env
    env_key = "OPENAI_API_KEY"
    env_value = ""
    env_file = repo_root / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(f"{env_key}="):
                env_value = line.split("=", 1)[1].strip('"\'')
                break

    model = _normalize_model_for_codex(big_model)
    base_url = f"http://127.0.0.1:{proxy_port}/v1"

    if config_path.suffix == ".json":
        data: dict = {}
        if config_path.exists():
            try:
                data = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        data["model"] = model
        data["model_provider"] = "nebius"
        providers = data.setdefault("model_providers", {})
        providers["nebius"] = {
            "name": "Nebius Proxy",
            "base_url": base_url,
            "env_key": env_key,
            "wire_api": "responses",
        }
        config_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return {"action": "written", "message": f"Updated {config_path}", "path": str(config_path)}

    # TOML path — string-manipulation (no extra deps needed)
    raw = ""
    if config_path.exists():
        raw = config_path.read_text(encoding="utf-8")

    # Helper to replace or append a top-level key
    def _set_kv(key: str, value: str) -> None:
        nonlocal raw
        pattern = rf"^{re.escape(key)}\s*=.*$"
        replacement = f'{key} = "{value}"'
        if re.search(pattern, raw, re.MULTILINE):
            raw = re.sub(pattern, replacement, raw, flags=re.MULTILINE)
        else:
            raw = raw.rstrip("\n") + f"\n{replacement}\n"

    _set_kv("model", model)
    _set_kv("model_provider", "nebius")

    # Replace or insert [model_providers.nebius] block
    section = (
        "[model_providers.nebius]\n"
        f'    name = "Nebius Proxy"\n'
        f'    base_url = "{base_url}"\n'
        f'    env_key = "{env_key}"\n'
        f'    wire_api = "responses"\n'
    )
    block_re = r"\[model_providers\.nebius\].*?(?=\n\[|$)"
    block_match = re.search(block_re, raw, re.DOTALL)
    if block_match:
        raw = raw[:block_match.start()] + section + raw[block_match.end():]
    else:
        raw = raw.rstrip("\n") + "\n" + section

    config_path.write_text(raw.rstrip("\n") + "\n", encoding="utf-8")
    return {"action": "written", "message": f"Updated {config_path}", "path": str(config_path)}
