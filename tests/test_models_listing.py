import pytest
from fastapi.testclient import TestClient

from src.api import endpoints
from src.core import tokenfactory_models


class _StubModel:
    def __init__(self, id, owned_by="nebius"):
        self.id = id
        self.owned_by = owned_by


class _StubModelsList:
    async def list(self):
        return type("R", (), {"data": [
            _StubModel("moonshotai/Kimi-K2.5"),
            _StubModel("zai-org/GLM-4.5"),
            _StubModel("Qwen/Qwen2.5-VL-72B-Instruct"),
            _StubModel("google/gemma-3-27b-it"),
            _StubModel("deepseek-ai/DeepSeek-V3"),
        ]})()


class _StubFailingModelsList:
    async def list(self):
        raise RuntimeError("upstream unreachable")


def _client():
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(endpoints.router)
    return TestClient(app)


def test_listing_includes_aliases_tf_catalog_and_no_claude_fakes(monkeypatch):
    tokenfactory_models.reset_cache()
    monkeypatch.setattr(endpoints.openai_client.client, "models", _StubModelsList())

    resp = _client().get("/v1/models")
    assert resp.status_code == 200
    ids = {m["id"] for m in resp.json()["data"]}

    # PR #24 short aliases must appear.
    assert {"glm", "kimi", "gemma"} <= ids

    # Stubbed Token Factory catalog must appear.
    assert {
        "moonshotai/Kimi-K2.5",
        "zai-org/GLM-4.5",
        "Qwen/Qwen2.5-VL-72B-Instruct",
        "deepseek-ai/DeepSeek-V3",
    } <= ids

    # The dropped fake claude-* entries must NOT appear.
    fake_claude = {i for i in ids if i.startswith(("claude-haiku", "claude-sonnet", "claude-opus", "claude-3-"))}
    assert fake_claude == set(), f"unexpected fake claude entries: {fake_claude}"


def test_listing_degrades_gracefully_when_upstream_fails(monkeypatch):
    tokenfactory_models.reset_cache()
    monkeypatch.setattr(endpoints.openai_client.client, "models", _StubFailingModelsList())

    resp = _client().get("/v1/models")
    assert resp.status_code == 200
    ids = {m["id"] for m in resp.json()["data"]}

    # Aliases still show up; env-configured ids still show up; no crash.
    assert {"glm", "kimi", "gemma"} <= ids


def test_listing_alias_ids_come_before_tf_catalog(monkeypatch):
    tokenfactory_models.reset_cache()
    monkeypatch.setattr(endpoints.openai_client.client, "models", _StubModelsList())

    data = _client().get("/v1/models").json()["data"]
    ids = [m["id"] for m in data]

    # The three aliases should be the first three entries (top of the picker).
    assert ids[0] == "glm"
    assert ids[1] == "kimi"
    assert ids[2] == "gemma"
