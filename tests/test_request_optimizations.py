from unittest.mock import AsyncMock, Mock

from fastapi.testclient import TestClient

from src.api import endpoints
from src.api.optimization_handlers import (
    extract_command_prefix,
    extract_filepaths_from_command,
    is_filepath_extraction_request,
    is_prefix_detection_request,
    is_quota_check_request,
    is_suggestion_mode_request,
    is_title_generation_request,
    try_local_optimization,
)
from src.core.config import config
from src.main import app
from src.models.claude import ClaudeMessage, ClaudeMessagesRequest


def _request(content, *, max_tokens=100, system=None, tools=None, stream=False):
    return ClaudeMessagesRequest(
        model="claude-3-5-sonnet-20241022",
        max_tokens=max_tokens,
        messages=[ClaudeMessage(role="user", content=content)],
        system=system,
        tools=tools,
        stream=stream,
    )


def test_quota_probe_detection():
    assert is_quota_check_request(_request("Check my QUOTA", max_tokens=1)) is True
    assert is_quota_check_request(_request("Check my quota", max_tokens=100)) is False


def test_prefix_detection_and_command_prefix_extraction():
    matched, command = is_prefix_detection_request(
        _request("<policy_spec>policy</policy_spec>\nCommand: git commit -m test")
    )

    assert matched is True
    assert command == "git commit -m test"
    assert extract_command_prefix(command) == "git commit"
    assert extract_command_prefix("echo $(cat /etc/passwd)") == "command_injection_detected"


def test_title_and_suggestion_detection():
    title_req = _request(
        "make title",
        system=[
            {
                "type": "text",
                "text": (
                    "Generate a concise, sentence-case title for this coding session. "
                    "Return JSON with a single title field."
                ),
            }
        ],
    )
    suggestion_req = _request("hello\n[SUGGESTION MODE: on]\nworld")

    assert is_title_generation_request(title_req) is True
    assert is_suggestion_mode_request(suggestion_req) is True


def test_filepath_detection_and_extraction():
    req = _request(
        "Command: cat -n foo.txt bar.md\n" "Output: line1\nline2\n\n" "Please extract <filepaths>."
    )
    matched, command, output = is_filepath_extraction_request(req)

    assert matched is True
    assert command == "cat -n foo.txt bar.md"
    assert output == "line1\nline2"
    assert extract_filepaths_from_command(command, output) == (
        "<filepaths>\nfoo.txt\nbar.md\n</filepaths>"
    )
    assert extract_filepaths_from_command("ls -la") == "<filepaths>\n</filepaths>"


def test_try_local_optimization_can_be_disabled(monkeypatch):
    monkeypatch.setattr(config, "enable_request_optimizations", False)

    assert try_local_optimization(_request("quota", max_tokens=1)) is None


def test_non_streaming_fast_path_does_not_call_provider(monkeypatch):
    monkeypatch.setattr(config, "enable_request_optimizations", True)
    provider_call = AsyncMock(side_effect=AssertionError("provider should not be called"))
    monkeypatch.setattr(endpoints.openai_client, "create_chat_completion", provider_call)

    client = TestClient(app)
    response = client.post(
        "/v1/messages",
        json={
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "quota"}],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["content"] == [{"type": "text", "text": "Quota check passed."}]
    assert body["usage"]["input_tokens"] > 0
    assert body["usage"]["output_tokens"] > 0
    provider_call.assert_not_called()


def test_streaming_fast_path_returns_anthropic_sse_without_provider(monkeypatch):
    monkeypatch.setattr(config, "enable_request_optimizations", True)
    provider_call = Mock(side_effect=AssertionError("provider stream should not be called"))
    monkeypatch.setattr(endpoints.openai_client, "create_chat_completion_stream", provider_call)

    client = TestClient(app)
    response = client.post(
        "/v1/messages",
        json={
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 100,
            "stream": True,
            "messages": [
                {
                    "role": "user",
                    "content": "<policy_spec>policy</policy_spec>\nCommand: npm install pkg",
                }
            ],
        },
    )

    assert response.status_code == 200
    assert "event: message_start" in response.text
    assert "event: content_block_delta" in response.text
    assert '"text": "npm install"' in response.text
    assert "event: message_stop" in response.text
    provider_call.assert_not_called()
