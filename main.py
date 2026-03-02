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

    Currently handles:
      - Removing stale per-document Chroma collections left over from the
        old dual-store architecture (data is preserved in the global store).
    """
    try:
        from src.ingestion.vector_store import migrate_per_doc_collections
        removed = migrate_per_doc_collections()
        if removed:
            logger.info(
                "Startup migration: cleaned up %d stale collection(s).", removed
            )
    except Exception as exc:
        # A migration failure must never prevent the application from starting
        logger.warning("Startup migration failed (non-fatal): %s", exc)


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

    # Always run migrations first, before any server starts
    run_migrations()

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