"""
Application entrypoint.

Usage:
    python main.py          # Launches BOTH FastAPI backend and Gradio UI
    python main.py --ui     # Gradio UI only
    python main.py --api    # FastAPI backend only
"""
from __future__ import annotations

import argparse
import logging
import sys
import threading
from pathlib import Path
from src.core.logger import setup_logging

sys.path.insert(0, str(Path(__file__).resolve().parent))

# setup_logging() returns None, so we need to get the logger separately
setup_logging(level="INFO", log_dir="logs", log_filename="app.log")
logger = logging.getLogger(__name__)  # Get the logger properly


def run_migrations() -> None:
    """
    Run any pending one-time data migrations before the servers start.
    """
    try:
        from src.storage.document_store import migrate_per_doc_collections
        removed = migrate_per_doc_collections()
        if removed:
            logger.info("Migration: cleaned up %d stale collection(s).", removed)
    except Exception as exc:
        logger.warning("Migration failed (non-fatal): %s", exc)


def run_history_purge() -> None:
    try:
        from src.storage.history_store import purge_old_turns
        purged = purge_old_turns()
        if purged:
            logger.info("History purge: removed %d old turn(s).", purged)
    except Exception as exc:
        logger.warning("History purge failed (non-fatal): %s", exc)


def run_api() -> None:
    import uvicorn
    from config.settings import API_HOST, API_PORT
    logger.info("Starting FastAPI on http://%s:%d", API_HOST, API_PORT)
    uvicorn.run("src.api.app:app", host=API_HOST, port=API_PORT, reload=False, log_level="warning")


def run_ui() -> None:
    from src.ui.app import launch
    logger.info("Starting Gradio UI...")
    launch()


def main() -> None:
    parser = argparse.ArgumentParser(description="Document Q&A Assistant")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ui",  action="store_true", help="Run Gradio UI only")
    group.add_argument("--api", action="store_true", help="Run FastAPI backend only")
    args = parser.parse_args()

    run_migrations()
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