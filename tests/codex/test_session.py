"""Tests for Codex SessionStore."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from src.codex.models import ResponsesItem
from src.codex.session import SessionData, SessionStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _async_sync(coro):
    """Run an async coroutine synchronously for test convenience."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Basic get/put
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_put_and_get():
    store = SessionStore(ttl_seconds=3600)
    items = [ResponsesItem(type="message", role="user", content="Hello")]
    await store.put("resp_1", items, [ResponsesItem(type="message", role="assistant", content="Hi")])
    session = await store.get("resp_1")
    assert session is not None
    assert session.response_id == "resp_1"
    assert len(session.input_items) == 1
    assert session.input_items[0].content == "Hello"
    assert session.output_items[0].content == "Hi"
    assert session.previous_id is None


@pytest.mark.asyncio
async def test_get_missing_returns_none():
    store = SessionStore()
    result = await store.get("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_put_with_previous_id():
    store = SessionStore()
    await store.put(
        "resp_2",
        [ResponsesItem(type="message", role="user", content="Turn 2")],
        [ResponsesItem(type="message", role="assistant", content="Answer 2")],
        previous_id="resp_1",
    )
    session = await store.get("resp_2")
    assert session is not None
    assert session.previous_id == "resp_1"


# ---------------------------------------------------------------------------
# TTL eviction
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ttl_eviction_expired():
    store = SessionStore(ttl_seconds=1)
    await store.put("expired", [ResponsesItem(type="text", text="x")], [])
    mock_time = 100.0
    loop = asyncio.get_event_loop()
    with patch.object(loop, "time", return_value=mock_time):
        # Within TTL
        store._store["expired"].created_at = mock_time - 0.5
        session = await store.get("expired")
        assert session is not None


@pytest.mark.asyncio
async def test_ttl_eviction_expired_get_returns_none():
    store = SessionStore(ttl_seconds=1)
    await store.put("gone", [ResponsesItem(type="text", text="x")], [])
    mock_time = 100.0
    loop = asyncio.get_event_loop()
    with patch.object(loop, "time", return_value=mock_time):
        # Expired (created_at from now, mock time is 100s in the future)
        store._store["gone"].created_at = mock_time - 2.0  # older than ttl=1
        session = await store.get("gone")
        assert session is None
        assert "gone" not in store._store


@pytest.mark.asyncio
async def test_evict_explicit():
    store = SessionStore(ttl_seconds=10)
    await store.put("stale", [], [])
    await store.put("fresh", [], [])

    mock_time = 100.0
    loop = asyncio.get_event_loop()
    with patch.object(loop, "time", return_value=mock_time):
        store._store["stale"].created_at = mock_time - 15.0  # expired
        store._store["fresh"].created_at = mock_time - 5.0   # not expired
        count = await store.evict()
        assert count == 1
        assert "stale" not in store._store
        assert "fresh" in store._store


# ---------------------------------------------------------------------------
# Session data representation
# ---------------------------------------------------------------------------
def test_session_data_fields():
    data = SessionData(
        response_id="r1",
        previous_id="r0",
        input_items=[ResponsesItem(type="text", text="in")],
        output_items=[ResponsesItem(type="text", text="out")],
        created_at=0.0,
    )
    assert data.response_id == "r1"
    assert data.previous_id == "r0"
    assert len(data.input_items) == 1
    assert len(data.output_items) == 1


# ---------------------------------------------------------------------------
# Store length
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_len():
    store = SessionStore()
    assert len(store) == 0
    await store.put("a", [], [])
    assert len(store) == 1
    await store.put("b", [], [])
    assert len(store) == 2
