"""
Centralised logging configuration for File Fellow.

Call `setup_logging()` once at application start-up (in main.py).
All other modules obtain their logger via the standard
``logging.getLogger(__name__)`` pattern — no direct imports of this
module are required beyond the single bootstrap call.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_dir: Path | None = None,
    log_filename: str = "app.log",
) -> None:
    """
    Configure the root logger with a console handler and an optional
    rotating file handler.

    Args:
        level:        Log level string, e.g. "INFO", "DEBUG", "WARNING".
        log_dir:      Directory for the log file.  When *None* only the
                      console handler is attached.
        log_filename: Name of the log file (used only when log_dir is set).
    """
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s: %(message)s"
    datefmt = "%H:%M:%S"
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
    ]

    if log_dir is not None:
        # Lazy import so the module is usable even if RotatingFileHandler
        # is unavailable in some minimal environments.
        from logging.handlers import RotatingFileHandler

        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / log_filename,
            maxBytes=5 * 1024 * 1024,   # 5 MB per file
            backupCount=3,
            encoding="utf-8",
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=numeric_level,
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,   # Override any prior basicConfig() calls
    )
    logging.getLogger(__name__).debug(
        "Logging initialised at level=%s, log_dir=%s", level, log_dir
    )