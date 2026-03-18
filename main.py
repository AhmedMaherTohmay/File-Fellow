"""
Application entrypoint.

Usage:
    python main.py          # Launches BOTH FastAPI backend and Gradio UI
    python main.py --ui     # Gradio UI only
    python main.py --api    # FastAPI backend only

Startup sequence
----------------
  1. Logging is configured first (before any other import that might log).
  2. init_db() creates the connection pool and runs any pending migrations.
     If PostgreSQL is unreachable the process exits here with a clear message.
  3. History purge runs (non-fatal — a failure does not block startup).
  4. FastAPI and/or Gradio start.
"""
from __future__ import annotations

import argparse
import logging
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.core.logger import setup_logging
from config.settings import settings

setup_logging(
    level=settings.LOG_LEVEL,
    log_dir=settings.LOG_DIR,
    log_filename="app.log",
)
logger = logging.getLogger(__name__)


def run_db_startup() -> None:
    """
    Initialise the connection pool and run pending migrations.

    Uses the new src.db.engine module. Raises RuntimeError on failure
    so the process exits immediately rather than serving errors at runtime.
    """
    from src.db.engine import init_db
    init_db()


def run_history_purge() -> None:
    """Delete conversation turns older than HISTORY_TTL_DAYS. Non-fatal."""
    try:
        from src.db.repositories.history_repo import purge_old_turns
        purged = purge_old_turns()
        if purged:
            logger.info("History purge: removed %d old turn(s).", purged)
    except Exception as exc:
        logger.warning("History purge failed (non-fatal): %s", exc)


def run_api() -> None:
    import uvicorn
    logger.info("Starting FastAPI on http://%s:%d", settings.API_HOST, settings.API_PORT)
    uvicorn.run(
        "src.api.app:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False,
        log_level="warning",
    )


def run_ui() -> None:
    from src.ui.app import launch
    logger.info("Starting Gradio UI on http://%s:%d", settings.GRADIO_HOST, settings.GRADIO_PORT)
    launch()


def main() -> None:
    parser = argparse.ArgumentParser(description="Document Q&A Assistant")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ui",  action="store_true", help="Run Gradio UI only")
    group.add_argument("--api", action="store_true", help="Run FastAPI backend only")
    args = parser.parse_args()

    run_db_startup()
    run_history_purge()

    if args.api:
        run_api()
    elif args.ui:
        run_ui()
    else:
        api_thread = threading.Thread(target=run_api, daemon=True)
        api_thread.start()
        run_ui()


if __name__ == "__main__":
    main()
