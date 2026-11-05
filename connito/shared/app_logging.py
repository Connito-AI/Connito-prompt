# llm_weightnet/shared/logging.py
from __future__ import annotations

import logging
import sys

import structlog

_format = "pretty"  # the last configure_logging's, for log_phase
_chosen = False  # set by a call off the defaults


class _JsonLineFormatter(logging.Formatter):
    """A structlog JSON line alone: it holds the traceback that stdlib's Logger.exception would append again."""

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        # Only structlog's own line, which carries the traceback as "exception": a library's record keeps its own.
        ours = message.startswith("{") and (not record.exc_info or '"exception": ' in message)
        return message if ours else super().format(record)


def configure_logging(fmt: str = "pretty", level: str = "INFO", use_utc: bool = True) -> None:
    """Pretty console (fmt="pretty") or one JSON object per line (fmt="json"), on stdout. The defaults are what every
    caller gets on import; a call with other values replaces the earlier setup, and later default calls keep it."""
    global _format, _chosen
    force = (fmt, level, use_utc) != ("pretty", "INFO", True)
    if _chosen and not force:  # a module imported late must not undo the entry point's choice
        return
    _format, _chosen = fmt, force
    # # 1) stdlib baseline so third-party libs (uvicorn, requests) show up
    # Forced only off the defaults, so the import-time calls keep any handler added between them.
    logging.basicConfig(stream=sys.stdout, level=level, format="%(message)s", force=force)

    # Silence noisy third-party logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("datasets.builder").setLevel(logging.WARNING)

    # Drop huggingface_hub's per-download "Could not set the permissions" warning.
    # It fires from `file_download._chmod_and_move` when chmod loses a race with
    # the `.incomplete` tmp file being cleaned up — harmless, but emits two
    # multi-line warnings per fetched file. Over a validator's day that's tens
    # of thousands of lines that drown the rest of the log. We target the
    # specific message rather than silencing the whole logger so genuine HF
    # warnings still surface.
    class _HfPermissionsNoise(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "Could not set the permissions on the file" not in record.getMessage()

    logging.getLogger("huggingface_hub.file_download").addFilter(_HfPermissionsNoise())

    # 2) structlog processors (common)
    timestamper = structlog.processors.TimeStamper(fmt="iso" if fmt == "json" else "%Y-%m-%d %H:%M:%S", utc=use_utc)
    common = [
        structlog.stdlib.add_log_level,
        timestamper,
        structlog.processors.StackInfoRenderer(),
    ]

    if fmt == "json":
        for handler in logging.getLogger().handlers:  # basicConfig's, replaced above
            handler.setFormatter(_JsonLineFormatter("%(message)s"))
        # format_exc_info renders a plain traceback.print_exception: no frame's locals.
        processors = common + [
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Pretty, aligned columns. pad_event aligns the event text; key=value follow neatly.
        processors = common + [
            structlog.dev.ConsoleRenderer(
                colors=True,
                pad_event=28,  # adjust to your taste
                exception_formatter=structlog.dev.plain_traceback,
            )
        ]

    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


# Module-level logger you can import directly
structlog.configure_once(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# ANSI escape for dim dark-gray text — bypasses structlog so the style isn't overridden
_DIM = "\033[90m"
_RESET = "\033[0m"


def log_phase(msg: str, **kwargs) -> None:
    """Log a message that always prints in dim (darker) colour; under JSON, an ordinary info line."""
    if _format == "json":
        # A fresh proxy: one cached under the old setup would print it pretty.
        structlog.get_logger().info(msg, **kwargs)
        return
    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
    line = f"{ts} [info     ] {msg}"
    if extra:
        line += f" {extra}"
    print(f"{_DIM}{line}{_RESET}", flush=True)
