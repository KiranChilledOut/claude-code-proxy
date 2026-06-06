"""Tests for server-side web search execution (Tavily-backed)."""

import pytest

from src.conversion import server_tools
from src.core.config import config
from src.models.claude import ClaudeMessagesRequest


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------
def test_is_search_tool_name():
    assert server_tools.is_search_tool_name("web_search")
    assert server_tools.is_search_tool_name("WebSearch")
    assert not server_tools.is_search_tool_name("WebFetch")
    assert not server_tools.is_search_tool_name("Bash")


def test_request_has_search_tool(monkeypatch):
    monkeypatch.setattr(config, "tavily_api_key", "tvly-x")
    monkeypatch.setattr(config, "server_search_enabled", True)
    req = ClaudeMessagesRequest(
        model="claude-opus-4-8", max_tokens=10,
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"name": "WebSearch", "input_schema": {"type": "object", "properties": {}}}],
    )
    assert server_tools.request_has_search_tool(req) is True
    # No key -> inert
    monkeypatch.setattr(config, "tavily_api_key", "")
    assert server_tools.request_has_search_tool(req) is False


def test_request_has_search_tool_false_without_search(monkeypatch):
    monkeypatch.setattr(config, "tavily_api_key", "tvly-x")
    req = ClaudeMessagesRequest(
        model="claude-opus-4-8", max_tokens=10,
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"name": "Bash", "input_schema": {"type": "object", "properties": {}}}],
    )
    assert server_tools.request_has_search_tool(req) is False


# --------------------------------------------------------------------------
# tavily_search formatting (httpx mocked)
# --------------------------------------------------------------------------
class _FakeResp:
    def __init__(self, data):
        self._d = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._d


class _FakeAsyncClient:
    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None, headers=None):
        return _FakeResp({"answer": "42", "results": [{"title": "T", "url": "u", "content": "c"}]})


@pytest.mark.asyncio
async def test_tavily_search_formats(monkeypatch):
    monkeypatch.setattr(config, "tavily_api_key", "tvly-x")
    monkeypatch.setattr(config, "tavily_max_results", 5)
    monkeypatch.setattr(server_tools.httpx, "AsyncClient", _FakeAsyncClient)
    out = await server_tools.tavily_search("q")
    assert "Answer: 42" in out
    assert "T (u)" in out


@pytest.mark.asyncio
async def test_tavily_search_empty_query():
    assert "No search query" in await server_tools.tavily_search("")


# --------------------------------------------------------------------------
# run_search_loop
# --------------------------------------------------------------------------
class _FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def create_chat_completion(self, req, request_id=None):
        self.calls.append(req)
        return self._responses.pop(0)


@pytest.mark.asyncio
async def test_loop_executes_search_then_answers(monkeypatch):
    monkeypatch.setattr(config, "server_search_max_iters", 4)

    async def fake_search(q):
        return f"RESULTS for {q}"

    monkeypatch.setattr(server_tools, "tavily_search", fake_search)

    first = {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
        {"id": "t1", "type": "function",
         "function": {"name": "web_search", "arguments": '{"query":"spacex"}'}}]}}]}
    final = {"choices": [{"finish_reason": "stop",
             "message": {"role": "assistant", "content": "SpaceX launches tomorrow."}}]}
    client = _FakeClient([first, final])

    resp = await server_tools.run_search_loop(
        {"model": "m", "messages": [{"role": "user", "content": "when"}]}, client, "rid"
    )
    assert resp is final
    # second backend call must carry the tool result
    second_msgs = client.calls[1]["messages"]
    assert any(m.get("role") == "tool" and "RESULTS for spacex" in m.get("content", "")
               for m in second_msgs)


@pytest.mark.asyncio
async def test_loop_passes_through_client_tools():
    first = {"choices": [{"message": {"tool_calls": [
        {"id": "b", "type": "function", "function": {"name": "Bash", "arguments": "{}"}}]}}]}
    client = _FakeClient([first])
    resp = await server_tools.run_search_loop({"messages": []}, client, "r")
    assert resp is first
    assert len(client.calls) == 1  # no extra backend round-trips


@pytest.mark.asyncio
async def test_loop_handles_kimi_token_args(monkeypatch):
    """Backend leaks Kimi control tokens in the search args; loop still extracts query."""
    seen = {}

    async def fake_search(q):
        seen["q"] = q
        return "ok"

    monkeypatch.setattr(server_tools, "tavily_search", fake_search)
    blob = ('<|tool_call_argument_begin|> {"query": "kimi test"} <|tool_call_end|>')
    first = {"choices": [{"message": {"tool_calls": [
        {"id": "t1", "type": "function", "function": {"name": "web_search", "arguments": blob}}]}}]}
    final = {"choices": [{"finish_reason": "stop", "message": {"content": "done"}}]}
    client = _FakeClient([first, final])
    await server_tools.run_search_loop({"messages": []}, client, "r")
    assert seen["q"] == "kimi test"
