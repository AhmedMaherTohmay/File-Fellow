"""
Database initialisation script.

This is the single command you run to set up or update the database.
It is safe to run at any time — nothing happens if everything is already
up to date.

─────────────────────────────────────────────────────────────────────────────
FIRST-TIME SETUP (run these once, in order)
─────────────────────────────────────────────────────────────────────────────

1.  Create the PostgreSQL database (if it doesn't exist):

        createdb filefellow

    Or using psql:

        psql -U postgres -c "CREATE DATABASE filefellow;"

2.  Enable the pgvector extension inside that database:

        psql -U postgres -d filefellow \
             -c "CREATE EXTENSION IF NOT EXISTS vector;"

    pgvector must be installed on your PostgreSQL server first.
    Download from https://github.com/pgvector/pgvector/releases
    and follow the installation README for your OS.

3.  Set your DATABASE_URL in .env:

        DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/filefellow

4.  Run this script:

        python scripts/init_db.py

─────────────────────────────────────────────────────────────────────────────
AFTER PULLING NEW CODE
─────────────────────────────────────────────────────────────────────────────

    python scripts/init_db.py

The migration runner checks which files have already been applied and only
runs new ones. If nothing is new, it exits immediately. Always safe to run.

─────────────────────────────────────────────────────────────────────────────
HOW MIGRATIONS WORK IN THIS PROJECT
─────────────────────────────────────────────────────────────────────────────

Migration files live in  migrations/  at the project root.
They are numbered SQL files:  001_initial.sql, 002_..., 003_...

On every run, the engine:

  1. Creates a schema_migrations tracking table (if it doesn't exist yet).
  2. Reads all .sql files from migrations/ sorted by name.
  3. For each file:
       Already recorded in schema_migrations? → skip.
       Not recorded? → execute it, record it as done.

Rules:
  - Each migration runs exactly once, ever, on every machine.
  - Never edit a committed migration file. Write a new numbered one.
  - Never run ALTER TABLE or CREATE TABLE in pgAdmin manually.
    Other machines won't know, and your environments will silently diverge.

─────────────────────────────────────────────────────────────────────────────
WHAT THIS SCRIPT ACTUALLY DOES
─────────────────────────────────────────────────────────────────────────────

  1. Creates the connection pool (tests that PostgreSQL is reachable).
  2. Creates schema_migrations table if needed.
  3. Applies any pending .sql files from migrations/.
  4. Creates / verifies HNSW vector indexes on embedding columns.
     (HNSW parameters come from settings.py — they can't be in a .sql file.)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Starting database initialisation...")

    try:
        from src.db.engine import init_db
        init_db()
        logger.info("Done. Database is ready.")
    except RuntimeError as exc:
        logger.error("Initialisation failed:\n\n%s", exc)
        logger.error(
            "\n"
            "─────────────────────────────────────\n"
            " Troubleshooting checklist\n"
            "─────────────────────────────────────\n"
            " 1. Is PostgreSQL running?\n"
            "      Windows: check Services or pgAdmin server status\n"
            "\n"
            " 2. Does the database exist?\n"
            "      createdb filefellow\n"
            "      # or in psql:\n"
            "      CREATE DATABASE filefellow;\n"
            "\n"
            " 3. Is pgvector installed and enabled?\n"
            "      psql -d filefellow\n"
            "      CREATE EXTENSION IF NOT EXISTS vector;\n"
            "\n"
            " 4. Is DATABASE_URL correct in .env?\n"
            "      postgresql://user:password@localhost:5432/filefellow\n"
            "─────────────────────────────────────\n"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
