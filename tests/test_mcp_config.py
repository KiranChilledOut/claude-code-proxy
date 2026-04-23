import json
from pathlib import Path


def test_project_mcp_config_uses_portable_repo_relative_launcher():
    repo_root = Path(__file__).resolve().parent.parent
    mcp_config_path = repo_root / ".mcp.json"

    config = json.loads(mcp_config_path.read_text())
    server = config["mcpServers"]["macoscontrol-mcp"]

    assert server["command"] == "bash"
    assert server["args"] == ["./MCP/macoscontrol-mcp/run.sh"]
    assert "/Users/" not in json.dumps(server)
    assert (repo_root / "MCP/macoscontrol-mcp/run.sh").exists()
