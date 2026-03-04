"""
Application entrypoint.

Usage:
    python main.py          # Launches BOTH the FastAPI backend and Gradio UI
    python main.py --ui     # Gradio UI only (no API server)
    python main.py --api    # FastAPI backend only
"""
from __future__ import annotations

import argparse
import logging
import sys
import threading
from pathlib import Path

# Ensure project root is on sys.path regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_migrations() -> None:
    """
    Run any pending one-time data migrations before the servers start.
    """
    try:
        from src.ingestion.vector_store import migrate_per_doc_collections
        removed = migrate_per_doc_collections()
        if removed:
            logger.info("Startup migration: cleaned up %d stale collection(s).", removed)
    except Exception as exc:
        logger.warning("Startup migration failed (non-fatal): %s", exc)


def run_history_purge() -> None:
    """
    Purge history turns older than HISTORY_TTL_DAYS at startup.
    """
    try:
        from src.memory.history_store import purge_old_turns
        purged = purge_old_turns()
        if purged:
            logger.info("Startup history purge: removed %d old turn(s).", purged)
    except Exception as exc:
        logger.warning("Startup history purge failed (non-fatal): %s", exc)


def run_api() -> None:
    import uvicorn
    from config.settings import API_HOST, API_PORT

    logger.info("Starting FastAPI server on http://%s:%d", API_HOST, API_PORT)
    uvicorn.run(
        "src.api.server:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="warning",
    )


def run_ui() -> None:
    from src.ui.gradio_app import launch

    logger.info("Starting Gradio UI...")
    launch()


def main() -> None:
    parser = argparse.ArgumentParser(description="File Fellow — Document Q&A Assistant")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ui", action="store_true", help="Run Gradio UI only")
    group.add_argument("--api", action="store_true", help="Run FastAPI backend only")
    args = parser.parse_args()

    # Always run migrations and cleanup first, before any server starts
    run_migrations()
    run_history_purge()

    if args.api:
        run_api()
    elif args.ui:
        run_ui()
    else:
        # Launch API in a daemon thread; UI runs on the main thread so that
        # Gradio's blocking .launch() call keeps the process alive.
        api_thread = threading.Thread(target=run_api, daemon=True)
        api_thread.start()
        run_ui()


if __name__ == "__main__":
    main()