"""In-memory session store with TTL eviction for previous_response_id continuity."""

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.codex.models import ResponsesItem


@dataclass
class SessionData:
    response_id: str
    previous_id: Optional[str]
    input_items: List[ResponsesItem]
    output_items: List[ResponsesItem]
    created_at: float  # monotonic timestamp


class SessionStore:
    """In-memory session store with TTL eviction for previous_response_id continuity."""

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self._store: Dict[str, SessionData] = {}
        self._ttl = ttl_seconds
        self._lock = asyncio.Lock()

    async def get(self, response_id: str) -> Optional[SessionData]:
        """Retrieve session by response_id, or None if expired / missing."""
        async with self._lock:
            session = self._store.get(response_id)
            if session is None:
                return None
            now = asyncio.get_event_loop().time()
            if now - session.created_at > self._ttl:
                self._store.pop(response_id, None)
                return None
            return session

    async def put(
        self,
        response_id: str,
        input_items: List[ResponsesItem],
        output_items: List[ResponsesItem],
        previous_id: Optional[str] = None,
    ) -> None:
        """Store a new session entry."""
        async with self._lock:
            self._store[response_id] = SessionData(
                response_id=response_id,
                previous_id=previous_id,
                input_items=list(input_items),
                output_items=list(output_items),
                created_at=asyncio.get_event_loop().time(),
            )

    async def evict(self) -> int:
        """Remove entries older than TTL. Returns number evicted."""
        now = asyncio.get_event_loop().time()
        evicted = 0
        async with self._lock:
            stale = [
                sid
                for sid, data in self._store.items()
                if now - data.created_at > self._ttl
            ]
            for sid in stale:
                self._store.pop(sid, None)
                evicted += 1
        return evicted

    def __len__(self) -> int:
        return len(self._store)
