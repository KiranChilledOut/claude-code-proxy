import time

import pytest

from src.observability.pricing import PricingCatalog
from src.observability.store import ObservabilityRecorder, utc_now_iso


def test_pricing_catalog_computes_model_cost():
    catalog = PricingCatalog(
        '{"zai-org/GLM-4.7-FP8":{"input_per_1m":0.30,"output_per_1m":1.20,"advertised_tok_s":36.8}}'
    )

    quote = catalog.quote("zai-org/GLM-4.7-FP8", 1_000_000, 500_000)

    assert quote["input_cost"] == pytest.approx(0.30)
    assert quote["output_cost"] == pytest.approx(0.60)
    assert quote["estimated_cost"] == pytest.approx(0.90)
    assert quote["advertised_tok_s"] == pytest.approx(36.8)


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
        usage={"input_tokens": 1000, "output_tokens": 500},
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

    assert len(tool_calls) == 1
    assert tool_calls[0]["tool_name"] == "bash"
    assert "echo ok" in tool_calls[0]["arguments_preview"]
    assert "secret" not in tool_calls[0]["arguments_preview"]
    assert "[redacted]" in tool_calls[0]["arguments_preview"]
