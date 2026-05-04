import logging
import time
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

_CACHE: Optional[Tuple[float, List[dict]]] = None


async def get_tokenfactory_models(openai_client: Any, ttl_seconds: int = 600) -> List[dict]:
    """Fetch the upstream model catalog from the configured OpenAI-compatible
    endpoint (Token Factory by default). Cached for ttl_seconds at module level.
    On error, returns the last cached result if any, otherwise an empty list."""
    global _CACHE
    now = time.time()
    if _CACHE and (now - _CACHE[0]) < ttl_seconds:
        return _CACHE[1]
    try:
        resp = await openai_client.client.models.list()
        models = [
            {"id": m.id, "owned_by": getattr(m, "owned_by", "")}
            for m in getattr(resp, "data", [])
        ]
        _CACHE = (now, models)
        return models
    except Exception as e:
        logger.warning("upstream /v1/models fetch failed: %s", e)
        return _CACHE[1] if _CACHE else []


def reset_cache() -> None:
    """Test helper: clear the module-level cache."""
    global _CACHE
    _CACHE = None
