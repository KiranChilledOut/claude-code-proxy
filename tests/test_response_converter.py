import json

import pytest

from src.conversion.response_converter import (
    _sanitize_tool_arguments,
    convert_openai_streaming_to_claude_with_cancellation,
    convert_openai_to_claude_response,
)
from src.models.claude import ClaudeMessage, ClaudeMessagesRequest


class _DummyRequest:
    async def is_disconnected(self):
        return False


class _DummyClient:
    def cancel_request(self, _request_id):
        return True


class _DummyLogger:
    def debug(self, *_args, **_kwargs):
        pass

    def info(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass

    def error(self, *_args, **_kwargs):
        pass


def test_sanitize_tool_arguments_extracts_xml_payload():
    name, arguments = _sanitize_tool_arguments(
        "Bash",
        "<arg_key>command</arg_key><arg_value>ls -la</arg_value>",
    )

    assert name == "Bash"
    assert json.loads(arguments) == {"command": "ls -la"}


def test_sanitize_tool_arguments_extracts_args_embedded_in_name():
    name, arguments = _sanitize_tool_arguments('bash(command="ls -la")', "")

    assert name == "bash"
    assert json.loads(arguments) == {"command": "ls -la"}


def test_non_streaming_response_sanitizes_tool_calls():
    request = ClaudeMessagesRequest(
        model="claude-3-5-sonnet-20241022",
        max_tokens=64,
        messages=[ClaudeMessage(role="user", content="hello")],
    )
    openai_response = {
        "id": "resp_1",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": 'bash(command="ls -la")',
                                "arguments": "",
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }

    response = convert_openai_to_claude_response(openai_response, request)

    assert response["stop_reason"] == "tool_use"
    assert response["content"] == [
        {
            "type": "tool_use",
            "id": "call_1",
            "name": "bash",
            "input": {"command": "ls -la"},
        }
    ]


async def _fake_stream():
    # Regular text delta
    yield "data: " + json.dumps({"choices": [{"delta": {"content": "A"}, "finish_reason": None}]})
    # Completion marker chunk
    yield "data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]})
    # Unexpected chunk after finish_reason that should be ignored
    yield "data: " + json.dumps({"choices": [{"delta": {"content": "B"}, "finish_reason": None}]})
    yield "data: [DONE]"


async def _fake_tool_stream():
    yield "data: " + json.dumps(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": 'bash(command="ls -la")',
                                },
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        }
    )
    yield "data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
    yield "data: [DONE]"


@pytest.mark.asyncio
async def test_streaming_stops_after_finish_reason():
    request = ClaudeMessagesRequest(
        model="claude-3-5-sonnet-20241022",
        max_tokens=64,
        messages=[ClaudeMessage(role="user", content="hello")],
        stream=True,
    )

    events = []
    async for event in convert_openai_streaming_to_claude_with_cancellation(
        _fake_stream(),
        request,
        _DummyLogger(),
        _DummyRequest(),
        _DummyClient(),
        "req_1",
    ):
        events.append(event)

    serialized = "".join(events)
    assert '"text": "A"' in serialized
    assert '"text": "B"' not in serialized
    assert "event: message_stop" in serialized


@pytest.mark.asyncio
async def test_streaming_flushes_sanitized_tool_arguments_on_finish():
    request = ClaudeMessagesRequest(
        model="claude-3-5-sonnet-20241022",
        max_tokens=64,
        messages=[ClaudeMessage(role="user", content="run ls")],
        stream=True,
    )

    events = []
    async for event in convert_openai_streaming_to_claude_with_cancellation(
        _fake_tool_stream(),
        request,
        _DummyLogger(),
        _DummyRequest(),
        _DummyClient(),
        "req_tool_1",
    ):
        events.append(event)

    serialized = "".join(events)
    assert '"type": "tool_use"' in serialized
    assert '"name": "bash"' in serialized
    assert '"partial_json": "{\\"command\\": \\"ls -la\\"}"' in serialized
    assert '"stop_reason": "tool_use"' in serialized
