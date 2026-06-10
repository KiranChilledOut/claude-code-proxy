import logging

from src.core.config import config

# Parse log level — take first word and normalise aliases.
_log_raw = (config.log_level or "INFO").split()[0].upper()
LOG_LEVEL = _log_raw if _log_raw in ("DEBUG", "INFO", "WARNING", "WARN", "ERROR", "CRITICAL") else "INFO"

# Numeric level for comparisons.
LOG_LEVEL_NUM = getattr(logging, "WARN" if LOG_LEVEL == "WARN" else LOG_LEVEL)

# Root application logger.
logging.basicConfig(
    level=LOG_LEVEL_NUM,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class UvicornAccessFilter(logging.Filter):
    """Suppress noisy 200 OK access logs for dashboard observability endpoints.

    Keeps failures (non-2xx) visible so errors don't get lost in the noise.
    """

    NOISY_PREFIXES = (
        "/api/observability/sessions",
        "/api/observability/summary",
        "/api/observability/requests",
        "/api/observability/failures",
        "/api/observability/tool-calls",
        "/api/observability/config",
        "/api/observability/context-usage",
        "/dashboard",
        "/dashboard/assets",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        # At WARNING or above: drop all 200 OKs regardless of path.
        if LOG_LEVEL in ("WARNING", "WARN", "ERROR", "CRITICAL"):
            if " 200 OK" in msg:
                return False
            return True  # errors/warnings still pass through.

        # At INFO: suppress dashboard polls only.
        if " 200 OK" not in msg:
            return True
        for prefix in self.NOISY_PREFIXES:
            if prefix in msg:
                return False
        return True


# Uvicorn loggers.
for uvicorn_logger in ["uvicorn", "uvicorn.error"]:
    logging.getLogger(uvicorn_logger).setLevel(
        LOG_LEVEL_NUM if LOG_LEVEL == "DEBUG" else logging.WARNING
    )

_uvicorn_access = logging.getLogger("uvicorn.access")
_uvicorn_access.setLevel(logging.INFO)
_uvicorn_access.addFilter(UvicornAccessFilter())
