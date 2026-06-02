import sqlite3
import time
from unittest.mock import MagicMock

import pytest

from src.observability.pricing import PricingCatalog
from src.observability.store import ObservabilityRecorder, utc_now_iso


def test_stream_usage_falls_back_to_estimate_when_provider_usage_is_missing():
    from src.api.endpoints import _stream_usage_with_fallback

    usage = _stream_usage_with_fallback({"usage": {}, "estimated_output_tokens": 12}, 345)

    assert usage == {
        "input_tokens": 345,
        "output_tokens": 12,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "source": "estimated",
    }


def test_pricing_catalog_computes_model_cost():
    catalog = PricingCatalog(
        '{"zai-org/GLM-4.7-FP8":{"input_per_1m":0.30,"output_per_1m":1.20,"advertised_tok_s":36.8}}'
    )

    quote = catalog.quote("zai-org/GLM-4.7-FP8", 1_000_000, 500_000)

    assert quote["input_cost"] == pytest.approx(0.30)
    assert quote["output_cost"] == pytest.approx(0.60)
    assert quote["estimated_cost"] == pytest.approx(0.90)
    assert quote["advertised_tok_s"] == pytest.approx(36.8)


def test_pricing_catalog_treats_local_optimizations_as_free():
    catalog = PricingCatalog("{}")

    quote = catalog.quote("local/quota_probe", 1_000_000, 500_000)

    assert quote["input_cost"] == 0
    assert quote["output_cost"] == 0
    assert quote["estimated_cost"] == 0
    assert quote["currency"] == "USD"


@pytest.mark.asyncio
async def test_observability_recorder_persists_request_and_tool_call(tmp_path):
    db_path = tmp_path / "observability.sqlite3"
    recorder = ObservabilityRecorder(
        enabled=True,
        db_path=str(db_path),
        queue_size=10,
        pricing_catalog=PricingCatalog(
            '{"model-a":{"input_per_1m":0.50,"output_per_1m":2.00,"advertised_tok_s":40}}'
        ),
        store_tool_args=True,
    )

    await recorder.start()
    recorder.record_request(
        request_id="req_1",
        started_at=utc_now_iso(),
        started_at_unix=time.time(),
        completed_at=utc_now_iso(),
        base_url="https://api.tokenfactory.nebius.com/v1",
        claude_model="claude-sonnet",
        backend_model="model-a",
        stream=True,
        status="success",
        http_status=200,
        latency_ms=1000,
        usage={"input_tokens": 1000, "output_tokens": 500, "source": "estimated"},
        stop_reason="tool_use",
        tool_calls=[
            {
                "tool_id": "call_1",
                "tool_name": "bash",
                "arguments": {"command": "echo ok", "api_key": "secret"},
                "status": "emitted",
                "sanitized": True,
            }
        ],
    )
    await recorder.stop()

    requests = recorder.fetch_requests(limit=10)
    tool_calls = recorder.fetch_tool_calls(limit=10)

    assert len(requests) == 1
    assert requests[0]["backend_model"] == "model-a"
    assert requests[0]["estimated_cost"] == pytest.approx(0.0015)
    assert requests[0]["observed_tok_s"] == pytest.approx(500)
    assert requests[0]["tool_call_count"] == 1
    assert requests[0]["usage_source"] == "estimated"

    assert len(tool_calls) == 1
    assert tool_calls[0]["tool_name"] == "bash"
    assert "echo ok" in tool_calls[0]["arguments_preview"]
    assert "secret" not in tool_calls[0]["arguments_preview"]
    assert "[redacted]" in tool_calls[0]["arguments_preview"]


def test_connect_closes_connection_even_on_exception(monkeypatch):
    """_connect context manager calls conn.close() in finally when an exception is raised."""
    recorder = ObservabilityRecorder(
        enabled=True,
        db_path=":memory:",
        queue_size=10,
        pricing_catalog=PricingCatalog("{}"),
    )
    mock_conn = MagicMock()
    monkeypatch.setattr(sqlite3, "connect", lambda _path: mock_conn)

    with pytest.raises(RuntimeError):
        with recorder._connect() as _conn:
            raise RuntimeError("boom")

    mock_conn.close.assert_called_once()


def test_connect_closes_after_successful_yield(monkeypatch):
    """_connect context manager calls conn.close() after normal completion."""
    recorder = ObservabilityRecorder(
        enabled=True,
        db_path=":memory:",
        queue_size=10,
        pricing_catalog=PricingCatalog("{}"),
    )
    mock_conn = MagicMock()
    monkeypatch.setattr(sqlite3, "connect", lambda _path: mock_conn)

    with recorder._connect() as _conn:
        pass

    mock_conn.close.assert_called_once()


def test_context_usage_for_returns_latest_nonzero_tokens(tmp_path):
    """_context_usage_for returns latest request with tokens > 0."""
    db_path = tmp_path / "observability.sqlite3"
    recorder = ObservabilityRecorder(
        enabled=True,
        db_path=str(db_path),
        queue_size=10,
        pricing_catalog=PricingCatalog("{}"),
    )
    recorder._init_db()
    with recorder._connect() as conn:
        conn.execute(
            """
            INSERT INTO requests (
                request_id, started_at, started_at_unix, status,
                total_tokens, input_tokens, output_tokens,
                session_id, claude_model, backend_model,
                cache_read_input_tokens, cache_creation_input_tokens,
                stream, latency_ms, usage_source
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,'provider')
            """,
            (
                "r1", "2024-01-01T00:00:00", 1, "success",
                0, 0, 0, "s1", "", "",
                0, 0, 0, 0,
            ),
        )
        conn.execute(
            """
            INSERT INTO requests (
                request_id, started_at, started_at_unix, status,
                total_tokens, input_tokens, output_tokens,
                session_id, claude_model, backend_model,
                cache_read_input_tokens, cache_creation_input_tokens,
                stream, latency_ms, usage_source
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,'provider')
            """,
            (
                "r2", "2024-01-01T00:00:01", 2, "success",
                100, 50, 50, "s1", "claude-sonnet", "model-a",
                5, 0, 0, 1000,
            ),
        )
        conn.execute(
            """
            INSERT INTO requests (
                request_id, started_at, started_at_unix, status,
                total_tokens, input_tokens, output_tokens,
                session_id, claude_model, backend_model,
                cache_read_input_tokens, cache_creation_input_tokens,
                stream, latency_ms, usage_source
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,'provider')
            """,
            (
                "r3", "2024-01-01T00:00:02", 3, "success",
                200, 80, 120, "s1", "claude-sonnet", "model-b",
                10, 0, 0, 1000,
            ),
        )
        result = recorder._context_usage_for(conn, "session_id", "s1")

    assert result is not None
    assert result["total_tokens"] == 200
    assert result["input_tokens"] == 80
    assert result["output_tokens"] == 120
    assert result["cache_read_input_tokens"] == 15
    assert result["request_count"] == 3


def test_context_usage_for_returns_none_when_no_rows(tmp_path):
    """_context_usage_for returns None when no matching session."""
    db_path = tmp_path / "observability.sqlite3"
    recorder = ObservabilityRecorder(
        enabled=True,
        db_path=str(db_path),
        queue_size=10,
        pricing_catalog=PricingCatalog("{}"),
    )
    recorder._init_db()
    with recorder._connect() as conn:
        result = recorder._context_usage_for(conn, "session_id", "nonexistent")
    assert result is None
