from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse

from src.core.config import config
from src.core.model_manager import model_manager
from src.observability.store import observability_recorder

router = APIRouter()
STATIC_DIR = Path(__file__).resolve().parent / "static"


async def validate_dashboard_api_key(
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """Mirror the proxy API-key policy without importing endpoint routes."""
    if config.ignore_client_api_key:
        return

    client_api_key = None
    if x_api_key:
        client_api_key = x_api_key
    elif authorization and authorization.startswith("Bearer "):
        client_api_key = authorization.replace("Bearer ", "")

    if config.anthropic_api_key and not config.validate_client_api_key(client_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key.")


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(_: None = Depends(validate_dashboard_api_key)):
    return HTMLResponse((STATIC_DIR / "dashboard.html").read_text(encoding="utf-8"))


@router.get("/dashboard/assets/{asset_name}")
async def dashboard_asset(asset_name: str, _: None = Depends(validate_dashboard_api_key)):
    allowed = {
        "dashboard.css": "text/css",
        "dashboard.js": "application/javascript",
    }
    if asset_name not in allowed:
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(STATIC_DIR / asset_name, media_type=allowed[asset_name])


@router.get("/api/observability/summary")
async def observability_summary(
    hours: int = Query(24, ge=1, le=168),
    _: None = Depends(validate_dashboard_api_key),
):
    summary = observability_recorder.fetch_summary(hours=hours)
    summary["provider"] = {
        "base_url": config.openai_base_url,
        "observability_db_path": config.observability_db_path,
        "observability_enabled": config.observability_enabled,
        "store_tool_args": config.observability_store_tool_args,
    }
    summary["configured_models"] = {
        "big": config.big_model,
        "middle": config.middle_model,
        "small": config.small_model,
        "vision": config.vision_model,
    }
    summary["pricing"] = observability_recorder.pricing_catalog.as_list()
    return summary


@router.get("/api/observability/requests")
async def observability_requests(
    limit: int = Query(100, ge=1, le=500),
    _: None = Depends(validate_dashboard_api_key),
):
    return {"data": observability_recorder.fetch_requests(limit=limit)}


@router.get("/api/observability/failures")
async def observability_failures(
    limit: int = Query(100, ge=1, le=500),
    _: None = Depends(validate_dashboard_api_key),
):
    return {"data": observability_recorder.fetch_failures(limit=limit)}


@router.get("/api/observability/tool-calls")
async def observability_tool_calls(
    limit: int = Query(100, ge=1, le=500),
    _: None = Depends(validate_dashboard_api_key),
):
    return {"data": observability_recorder.fetch_tool_calls(limit=limit)}


@router.get("/api/observability/config")
async def observability_config(_: None = Depends(validate_dashboard_api_key)):
    return {
        "base_url": config.openai_base_url,
        "configured_models": {
            "big": config.big_model,
            "middle": config.middle_model,
            "small": config.small_model,
            "vision": config.vision_model,
        },
        "pricing": observability_recorder.pricing_catalog.as_list(),
        "observability_enabled": config.observability_enabled,
        "observability_db_path": config.observability_db_path,
        "store_tool_args": config.observability_store_tool_args,
        "routing": {
            "haiku": model_manager.config.small_model,
            "sonnet": model_manager.config.middle_model,
            "opus": model_manager.config.big_model,
            "image": model_manager.config.vision_model,
        },
    }
