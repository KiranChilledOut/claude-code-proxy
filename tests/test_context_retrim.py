"""Tests for upstream context-length self-heal (drop oldest + retry) and the
clearer context-window error message."""

from src.core import client as client_mod
from src.core.client import _is_context_length_error, _maybe_retrim_context


def test_is_context_length_error():
    assert _is_context_length_error(Exception("This model's maximum context length is 128000"))
    assert _is_context_length_error(Exception("Please reduce the length of the messages"))
    assert _is_context_length_error(Exception("input is too long for the context window"))
    assert not _is_context_length_error(Exception("invalid api key"))
    assert not _is_context_length_error(Exception("rate limit exceeded"))


def test_maybe_retrim_drops_and_signals_retry():
    msgs = [{"role": "system", "content": "system prompt"}]
    for _ in range(12):
        msgs.append({"role": "user", "content": "word " * 500})
        msgs.append({"role": "assistant", "content": "reply " * 500})
    req = {"model": "m", "messages": msgs}
    before = len(req["messages"])
    ok = _maybe_retrim_context(req, Exception("maximum context length exceeded"))
    assert ok is True
    assert len(req["messages"]) < before
    # system message preserved
    assert req["messages"][0]["role"] == "system"


def test_maybe_retrim_noop_on_other_error():
    req = {"messages": [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "hi " * 200},
        {"role": "assistant", "content": "yo " * 200},
    ]}
    assert _maybe_retrim_context(req, Exception("rate limit")) is False


def test_maybe_retrim_noop_when_too_few_messages():
    req = {"messages": [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "hi"},
    ]}
    assert _maybe_retrim_context(req, Exception("context length")) is False


def test_classify_context_length_message():
    c = client_mod.OpenAIClient("k", "http://localhost:9", 5)
    msg = c.classify_openai_error("This model's maximum context length is 128000 tokens")
    assert "context window" in msg.lower()
