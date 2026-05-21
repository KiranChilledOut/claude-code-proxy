import asyncio
import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.config import config
from src.observability.pricing import PricingCatalog

logger = logging.getLogger(__name__)

SENSITIVE_KEYS = ("api_key", "authorization", "bearer", "key", "password", "secret", "token")


class ObservabilityRecorder:
    """Best-effort request recorder backed by SQLite.

    Writes are queued and flushed by a background task so model calls do not
    wait on SQLite. When the bounded queue is full, records are dropped.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        db_path: str,
        queue_size: int,
        pricing_catalog: PricingCatalog,
        store_tool_args: bool = False,
    ):
        self.enabled = enabled
        self.db_path = db_path
        self.queue_size = max(1, queue_size)
        self.pricing_catalog = pricing_catalog
        self.store_tool_args = store_tool_args
        self.dropped_records = 0
        self._queue: Optional[asyncio.Queue] = None
        self._writer_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        if not self.enabled:
            return
        await asyncio.to_thread(self._init_db)
        self._queue = asyncio.Queue(maxsize=self.queue_size)
        self._writer_task = asyncio.create_task(self._writer_loop())
        logger.info("Observability enabled: db=%s", self.db_path)

    async def stop(self) -> None:
        if not self.enabled or self._queue is None:
            return
        await self._queue.put(None)
        if self._writer_task is not None:
            await self._writer_task
        self._queue = None
        self._writer_task = None

    def record_request(
        self,
        *,
        request_id: str,
        session_id: Optional[str] = None,
        session_name: Optional[str] = None,
        started_at: str,
        started_at_unix: float,
        completed_at: Optional[str],
        base_url: str,
        claude_model: str,
        backend_model: Optional[str],
        stream: bool,
        status: str,
        http_status: Optional[int],
        latency_ms: float,
        usage: Optional[Dict[str, Any]] = None,
        stop_reason: Optional[str] = None,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if not self.enabled or self._queue is None:
            return

        usage = usage or {}
        tool_calls = tool_calls or []
        input_tokens = _as_int(usage.get("input_tokens"))
        output_tokens = _as_int(usage.get("output_tokens"))
        cache_creation = _as_int(usage.get("cache_creation_input_tokens"))
        cache_read = _as_int(usage.get("cache_read_input_tokens"))
        usage_source = str(
            usage.get("source")
            or (
                "provider"
                if input_tokens or output_tokens or cache_creation or cache_read
                else "missing"
            )
        )
        pricing = self.pricing_catalog.quote(backend_model, input_tokens, output_tokens)
        observed_tok_s = None
        if output_tokens > 0 and latency_ms > 0:
            observed_tok_s = output_tokens / (latency_ms / 1000)

        item = {
            "kind": "request",
            "request": {
                "request_id": request_id,
                "session_id": session_id,
                "session_name": session_name,
                "started_at": started_at,
                "started_at_unix": started_at_unix,
                "completed_at": completed_at,
                "base_url": base_url,
                "claude_model": claude_model,
                "backend_model": backend_model,
                "stream": 1 if stream else 0,
                "status": status,
                "http_status": http_status,
                "stop_reason": stop_reason,
                "latency_ms": latency_ms,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_creation_input_tokens": cache_creation,
                "cache_read_input_tokens": cache_read,
                "usage_source": usage_source,
                "total_tokens": input_tokens + output_tokens,
                "input_cost": pricing["input_cost"],
                "output_cost": pricing["output_cost"],
                "estimated_cost": pricing["estimated_cost"],
                "currency": pricing["currency"],
                "advertised_tok_s": pricing["advertised_tok_s"],
                "observed_tok_s": observed_tok_s,
                "error_type": _truncate(error_type, 120),
                "error_message": _truncate(error_message, 1000),
                "tool_call_count": len(tool_calls),
            },
            "tool_calls": [
                {
                    "request_id": request_id,
                    "timestamp": completed_at or utc_now_iso(),
                    "tool_id": _truncate(str(tool.get("tool_id") or ""), 200),
                    "tool_name": _truncate(str(tool.get("tool_name") or ""), 200),
                    "arguments_preview": self._arguments_preview(tool.get("arguments")),
                    "status": _truncate(str(tool.get("status") or "emitted"), 80),
                    "sanitized": 1 if tool.get("sanitized") else 0,
                }
                for tool in tool_calls
            ],
        }

        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            self.dropped_records += 1

    def fetch_summary(self, *, hours: int = 24) -> Dict[str, Any]:
        if not self.enabled or not Path(self.db_path).exists():
            return self._empty_summary(hours)

        cutoff = time.time() - (hours * 3600)
        with self._connect() as conn:
            totals = conn.execute(
                """
                SELECT
                    COUNT(*) AS request_count,
                    COALESCE(SUM(CASE WHEN status != 'success' THEN 1 ELSE 0 END), 0) AS failure_count,
                    COALESCE(SUM(input_tokens), 0) AS input_tokens,
                    COALESCE(SUM(output_tokens), 0) AS output_tokens,
                    COALESCE(SUM(estimated_cost), 0) AS estimated_cost,
                    AVG(latency_ms) AS avg_latency_ms,
                    COALESCE(SUM(tool_call_count), 0) AS tool_call_count
                FROM requests
                WHERE started_at_unix >= ?
                """,
                (cutoff,),
            ).fetchone()
            all_time = conn.execute(
                """
                SELECT
                    COUNT(*) AS request_count,
                    COALESCE(SUM(input_tokens), 0) AS input_tokens,
                    COALESCE(SUM(output_tokens), 0) AS output_tokens,
                    COALESCE(SUM(estimated_cost), 0) AS estimated_cost
                FROM requests
                """
            ).fetchone()
            series = self._fetch_series(conn, cutoff, hours)
            model_stats = self._fetch_model_stats(conn, cutoff)

        return {
            "enabled": self.enabled,
            "hours": hours,
            "dropped_records": self.dropped_records,
            "window": dict(totals),
            "all_time": dict(all_time),
            "series": series,
            "model_stats": model_stats,
        }

    def fetch_requests(self, *, limit: int = 100) -> List[Dict[str, Any]]:
        return self._fetch_rows(
            """
            SELECT *
            FROM requests
            ORDER BY started_at_unix DESC
            LIMIT ?
            """,
            (max(1, min(limit, 500)),),
        )

    def fetch_failures(self, *, limit: int = 100) -> List[Dict[str, Any]]:
        return self._fetch_rows(
            """
            SELECT *
            FROM requests
            WHERE status != 'success'
            ORDER BY started_at_unix DESC
            LIMIT ?
            """,
            (max(1, min(limit, 500)),),
        )

    def fetch_tool_calls(self, *, limit: int = 100) -> List[Dict[str, Any]]:
        return self._fetch_rows(
            """
            SELECT tool_calls.*, requests.backend_model, requests.status AS request_status
            FROM tool_calls
            LEFT JOIN requests ON requests.request_id = tool_calls.request_id
            ORDER BY tool_calls.id DESC
            LIMIT ?
            """,
            (max(1, min(limit, 500)),),
        )

    def fetch_latest_session_id(self) -> Optional[str]:
        if not self.enabled or not Path(self.db_path).exists():
            return None
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT session_id
                FROM requests
                WHERE session_id IS NOT NULL
                ORDER BY started_at_unix DESC
                LIMIT 1
                """
            ).fetchone()
        return row["session_id"] if row else None

    def fetch_context_usage(self, session_id: str) -> Optional[Dict[str, Any]]:
        if not self.enabled or not Path(self.db_path).exists():
            return None

        with self._connect() as conn:
            latest = conn.execute(
                """
                SELECT claude_model, backend_model, completed_at
                FROM requests
                WHERE session_id = ? AND status = 'success'
                ORDER BY started_at_unix DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
            if latest is None:
                return None

            # Use MAX for tokens so local-optimisation (0-token housekeeping)
            # requests do not drop the reported context size.
            totals = conn.execute(
                """
                SELECT
                    COALESCE(MAX(total_tokens), 0) AS total_tokens,
                    COALESCE(MAX(input_tokens), 0) AS input_tokens,
                    COALESCE(MAX(output_tokens), 0) AS output_tokens,
                    COALESCE(SUM(cache_read_input_tokens), 0) AS cache_read_input_tokens,
                    COALESCE(SUM(cache_creation_input_tokens), 0) AS cache_creation_input_tokens,
                    COUNT(*) AS request_count
                FROM requests
                WHERE session_id = ? AND status = 'success'
                """,
                (session_id,),
            ).fetchone()

        return {
            "claude_model": latest["claude_model"],
            "backend_model": latest["backend_model"],
            "total_tokens": totals["total_tokens"] or 0,
            "input_tokens": totals["input_tokens"] or 0,
            "output_tokens": totals["output_tokens"] or 0,
            "cache_read_input_tokens": totals["cache_read_input_tokens"] or 0,
            "cache_creation_input_tokens": totals["cache_creation_input_tokens"] or 0,
            "request_count": totals["request_count"],
        }

    def fetch_context_usage_by_name(self, session_name: str) -> Optional[Dict[str, Any]]:
        if not self.enabled or not Path(self.db_path).exists():
            return None

        with self._connect() as conn:
            latest = conn.execute(
                """
                SELECT claude_model, backend_model, completed_at
                FROM requests
                WHERE session_name = ? AND status = 'success'
                ORDER BY started_at_unix DESC
                LIMIT 1
                """,
                (session_name,),
            ).fetchone()
            if latest is None:
                return None

            totals = conn.execute(
                """
                SELECT
                    COALESCE(MAX(total_tokens), 0) AS total_tokens,
                    COALESCE(MAX(input_tokens), 0) AS input_tokens,
                    COALESCE(MAX(output_tokens), 0) AS output_tokens,
                    COALESCE(SUM(cache_read_input_tokens), 0) AS cache_read_input_tokens,
                    COALESCE(SUM(cache_creation_input_tokens), 0) AS cache_creation_input_tokens,
                    COUNT(*) AS request_count
                FROM requests
                WHERE session_name = ? AND status = 'success'
                """,
                (session_name,),
            ).fetchone()

        return {
            "claude_model": latest["claude_model"],
            "backend_model": latest["backend_model"],
            "total_tokens": totals["total_tokens"] or 0,
            "input_tokens": totals["input_tokens"] or 0,
            "output_tokens": totals["output_tokens"] or 0,
            "cache_read_input_tokens": totals["cache_read_input_tokens"] or 0,
            "cache_creation_input_tokens": totals["cache_creation_input_tokens"] or 0,
            "request_count": totals["request_count"],
        }

    def fetch_sessions(self) -> List[Dict[str, Any]]:
        """Return distinct sessions from the requests table."""
        if not self.enabled or not Path(self.db_path).exists():
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    session_name,
                    session_id,
                    MIN(started_at) AS first_seen,
                    MAX(started_at) AS last_seen,
                    COUNT(*) AS total_requests,
                    COALESCE(SUM(total_tokens), 0) AS total_tokens,
                    COALESCE(MAX(total_tokens), 0) AS peak_context_tokens,
                    MAX(backend_model) AS backend_model,
                    COALESCE(SUM(estimated_cost), 0) AS total_cost
                FROM requests
                WHERE session_name IS NOT NULL
                GROUP BY session_name
                ORDER BY MAX(started_at_unix) DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def fetch_session_series(self, session_name: str, bucket_seconds: int = 300) -> List[Dict[str, Any]]:
        """Return bucketed time-series data for a single session."""
        if not self.enabled or not Path(self.db_path).exists():
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    CAST(started_at_unix / ? AS INTEGER) * ? AS bucket,
                    COUNT(*) AS request_count,
                    COALESCE(SUM(input_tokens), 0) AS input_tokens,
                    COALESCE(SUM(output_tokens), 0) AS output_tokens,
                    COALESCE(SUM(estimated_cost), 0) AS estimated_cost,
                    COALESCE(SUM(CASE WHEN status != 'success' THEN 1 ELSE 0 END), 0) AS failure_count
                FROM requests
                WHERE session_name = ?
                GROUP BY bucket
                ORDER BY bucket ASC
                """,
                (bucket_seconds, bucket_seconds, session_name),
            ).fetchall()
        return [
            {
                **dict(row),
                "timestamp": datetime.fromtimestamp(row["bucket"], tz=timezone.utc).isoformat(),
            }
            for row in rows
        ]

    def fetch_session_model_stats(self, session_name: str) -> List[Dict[str, Any]]:
        """Return per-model statistics for a single session."""
        if not self.enabled or not Path(self.db_path).exists():
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    backend_model,
                    COUNT(*) AS request_count,
                    COALESCE(SUM(input_tokens), 0) AS input_tokens,
                    COALESCE(SUM(output_tokens), 0) AS output_tokens,
                    COALESCE(SUM(estimated_cost), 0) AS estimated_cost,
                    AVG(latency_ms) AS avg_latency_ms,
                    AVG(observed_tok_s) AS avg_observed_tok_s,
                    MAX(advertised_tok_s) AS advertised_tok_s,
                    COALESCE(SUM(CASE WHEN status != 'success' THEN 1 ELSE 0 END), 0) AS failure_count
                FROM requests
                WHERE session_name = ?
                GROUP BY backend_model
                ORDER BY request_count DESC
                """,
                (session_name,),
            ).fetchall()
        return [dict(row) for row in rows]

    def fetch_session_summary(self, session_name: str) -> Dict[str, Any]:
        """Return a summary dict with window totals, model_stats, and series for a session."""
        if not self.enabled or not Path(self.db_path).exists():
            return {}
        with self._connect() as conn:
            totals = conn.execute(
                """
                SELECT
                    COUNT(*) AS request_count,
                    COALESCE(SUM(CASE WHEN status != 'success' THEN 1 ELSE 0 END), 0) AS failure_count,
                    COALESCE(SUM(input_tokens), 0) AS input_tokens,
                    COALESCE(SUM(output_tokens), 0) AS output_tokens,
                    COALESCE(SUM(estimated_cost), 0) AS estimated_cost,
                    AVG(latency_ms) AS avg_latency_ms,
                    COALESCE(SUM(tool_call_count), 0) AS tool_call_count
                FROM requests
                WHERE session_name = ?
                """,
                (session_name,),
            ).fetchone()
            if totals["request_count"] == 0:
                return {}
        series = self.fetch_session_series(session_name, bucket_seconds=300)
        model_stats = self.fetch_session_model_stats(session_name)
        return {
            "enabled": self.enabled,
            "session_name": session_name,
            "window": dict(totals),
            "series": series,
            "model_stats": model_stats,
        }

    def fetch_requests_by_session(self, session_name: str, limit: int = 500) -> List[Dict[str, Any]]:
        return self._fetch_rows(
            """
            SELECT *
            FROM requests
            WHERE session_name = ?
            ORDER BY started_at_unix DESC
            LIMIT ?
            """,
            (session_name, max(1, min(limit, 500))),
        )

    def fetch_failures_by_session(self, session_name: str, limit: int = 500) -> List[Dict[str, Any]]:
        return self._fetch_rows(
            """
            SELECT *
            FROM requests
            WHERE session_name = ? AND status != 'success'
            ORDER BY started_at_unix DESC
            LIMIT ?
            """,
            (session_name, max(1, min(limit, 500))),
        )

    def fetch_tool_calls_by_session(self, session_name: str, limit: int = 500) -> List[Dict[str, Any]]:
        return self._fetch_rows(
            """
            SELECT tool_calls.*, requests.backend_model, requests.status AS request_status
            FROM tool_calls
            LEFT JOIN requests ON requests.request_id = tool_calls.request_id
            WHERE requests.session_name = ?
            ORDER BY tool_calls.id DESC
            LIMIT ?
            """,
            (session_name, max(1, min(limit, 500))),
        )

    def _empty_summary(self, hours: int) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "hours": hours,
            "dropped_records": self.dropped_records,
            "window": {
                "request_count": 0,
                "failure_count": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "estimated_cost": 0,
                "avg_latency_ms": None,
                "tool_call_count": 0,
            },
            "all_time": {
                "request_count": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "estimated_cost": 0,
            },
            "series": [],
            "model_stats": [],
        }

    async def _writer_loop(self) -> None:
        assert self._queue is not None
        while True:
            item = await self._queue.get()
            if item is None:
                self._queue.task_done()
                break

            batch = [item]
            shutdown = False
            while len(batch) < 100:
                try:
                    next_item = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if next_item is None:
                    shutdown = True
                    self._queue.task_done()
                    break
                batch.append(next_item)

            try:
                await asyncio.to_thread(self._write_batch, batch)
            except Exception as exc:
                logger.warning("Failed to write observability batch: %s", exc)
            finally:
                for _ in batch:
                    self._queue.task_done()

            if shutdown:
                break

    def _init_db(self) -> None:
        db_file = Path(self.db_path)
        if db_file.parent and str(db_file.parent) not in ("", "."):
            db_file.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS requests (
                    request_id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    started_at_unix REAL NOT NULL,
                    completed_at TEXT,
                    base_url TEXT,
                    claude_model TEXT,
                    backend_model TEXT,
                    stream INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    http_status INTEGER,
                    stop_reason TEXT,
                    latency_ms REAL,
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    cache_creation_input_tokens INTEGER NOT NULL DEFAULT 0,
                    cache_read_input_tokens INTEGER NOT NULL DEFAULT 0,
                    usage_source TEXT NOT NULL DEFAULT 'provider',
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    input_cost REAL,
                    output_cost REAL,
                    estimated_cost REAL,
                    currency TEXT,
                    advertised_tok_s REAL,
                    observed_tok_s REAL,
                    error_type TEXT,
                    error_message TEXT,
                    tool_call_count INTEGER NOT NULL DEFAULT 0,
                    session_id TEXT
                )
                """
            )
            self._ensure_column(
                conn,
                "requests",
                "usage_source",
                "TEXT NOT NULL DEFAULT 'provider'",
            )
            self._ensure_column(
                conn,
                "requests",
                "session_id",
                "TEXT",
            )
            self._ensure_column(
                conn,
                "requests",
                "session_name",
                "TEXT",
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_requests_session ON requests(session_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_requests_session_name ON requests(session_name)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tool_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    tool_id TEXT,
                    tool_name TEXT,
                    arguments_preview TEXT,
                    status TEXT,
                    sanitized INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_requests_started ON requests(started_at_unix)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_requests_status ON requests(status)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tool_calls_request ON tool_calls(request_id)"
            )

    def _write_batch(self, batch: List[Dict[str, Any]]) -> None:
        with self._connect() as conn:
            for item in batch:
                request = item["request"]
                conn.execute(
                    """
                    INSERT OR REPLACE INTO requests (
                        request_id, session_id, session_name, started_at, started_at_unix, completed_at, base_url,
                        claude_model, backend_model, stream, status, http_status,
                        stop_reason, latency_ms, input_tokens, output_tokens,
                        cache_creation_input_tokens, cache_read_input_tokens, usage_source, total_tokens,
                        input_cost, output_cost, estimated_cost, currency,
                        advertised_tok_s, observed_tok_s, error_type, error_message,
                        tool_call_count
                    ) VALUES (
                        :request_id, :session_id, :session_name, :started_at, :started_at_unix, :completed_at, :base_url,
                        :claude_model, :backend_model, :stream, :status, :http_status,
                        :stop_reason, :latency_ms, :input_tokens, :output_tokens,
                        :cache_creation_input_tokens, :cache_read_input_tokens, :usage_source, :total_tokens,
                        :input_cost, :output_cost, :estimated_cost, :currency,
                        :advertised_tok_s, :observed_tok_s, :error_type, :error_message,
                        :tool_call_count
                    )
                    """,
                    request,
                )
                for tool_call in item["tool_calls"]:
                    conn.execute(
                        """
                        INSERT INTO tool_calls (
                            request_id, timestamp, tool_id, tool_name,
                            arguments_preview, status, sanitized
                        ) VALUES (
                            :request_id, :timestamp, :tool_id, :tool_name,
                            :arguments_preview, :status, :sanitized
                        )
                        """,
                        tool_call,
                    )

    def _ensure_column(
        self, conn: sqlite3.Connection, table: str, column: str, definition: str
    ) -> None:
        columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _fetch_rows(self, query: str, params: tuple) -> List[Dict[str, Any]]:
        if not self.enabled or not Path(self.db_path).exists():
            return []
        with self._connect() as conn:
            return [dict(row) for row in conn.execute(query, params).fetchall()]

    def _fetch_series(
        self, conn: sqlite3.Connection, cutoff: float, hours: int
    ) -> List[Dict[str, Any]]:
        if hours > 24 * 90:
            bucket_seconds = 24 * 60 * 60
        elif hours > 24 * 7:
            bucket_seconds = 60 * 60
        else:
            bucket_seconds = 5 * 60

        rows = conn.execute(
            """
            SELECT
                CAST(started_at_unix / ? AS INTEGER) * ? AS bucket,
                COUNT(*) AS request_count,
                COALESCE(SUM(input_tokens), 0) AS input_tokens,
                COALESCE(SUM(output_tokens), 0) AS output_tokens,
                COALESCE(SUM(estimated_cost), 0) AS estimated_cost,
                COALESCE(SUM(CASE WHEN status != 'success' THEN 1 ELSE 0 END), 0) AS failure_count
            FROM requests
            WHERE started_at_unix >= ?
            GROUP BY bucket
            ORDER BY bucket ASC
            """,
            (bucket_seconds, bucket_seconds, cutoff),
        ).fetchall()
        return [
            {
                **dict(row),
                "timestamp": datetime.fromtimestamp(row["bucket"], tz=timezone.utc).isoformat(),
            }
            for row in rows
        ]

    def _fetch_model_stats(self, conn: sqlite3.Connection, cutoff: float) -> List[Dict[str, Any]]:
        rows = conn.execute(
            """
            SELECT
                backend_model,
                COUNT(*) AS request_count,
                COALESCE(SUM(input_tokens), 0) AS input_tokens,
                COALESCE(SUM(output_tokens), 0) AS output_tokens,
                COALESCE(SUM(estimated_cost), 0) AS estimated_cost,
                AVG(latency_ms) AS avg_latency_ms,
                AVG(observed_tok_s) AS avg_observed_tok_s,
                MAX(advertised_tok_s) AS advertised_tok_s,
                COALESCE(SUM(CASE WHEN status != 'success' THEN 1 ELSE 0 END), 0) AS failure_count
            FROM requests
            WHERE started_at_unix >= ?
            GROUP BY backend_model
            ORDER BY request_count DESC
            """,
            (cutoff,),
        ).fetchall()
        return [dict(row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _arguments_preview(self, arguments: Any) -> str:
        if not self.store_tool_args:
            return "[disabled]"
        redacted = _redact_sensitive(arguments)
        try:
            return _truncate(json.dumps(redacted, ensure_ascii=False), 800) or ""
        except TypeError:
            return _truncate(str(redacted), 800) or ""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _truncate(value: Optional[str], limit: int) -> Optional[str]:
    if value is None:
        return None
    value = str(value)
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _redact_sensitive(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return _redact_sensitive(json.loads(value))
        except json.JSONDecodeError:
            return _truncate(value, 800)
    if isinstance(value, list):
        return [_redact_sensitive(item) for item in value]
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            key_str = str(key)
            if any(sensitive in key_str.lower() for sensitive in SENSITIVE_KEYS):
                redacted[key_str] = "[redacted]"
            else:
                redacted[key_str] = _redact_sensitive(item)
        return redacted
    return value


observability_recorder = ObservabilityRecorder(
    enabled=config.observability_enabled,
    db_path=config.observability_db_path,
    queue_size=config.observability_queue_size,
    pricing_catalog=PricingCatalog(config.model_prices_json),
    store_tool_args=config.observability_store_tool_args,
)
