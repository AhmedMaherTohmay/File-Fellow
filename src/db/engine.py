"""
Database engine — connection pool, migration runner, vector index setup.

This is the only module allowed to import psycopg2 directly.
Everything else gets a connection via get_connection().

How migrations work in this project
------------------------------------
Migration files live in  migrations/  at the project root.
They are numbered SQL files:  001_initial.sql, 002_..., 003_...

On every startup, init_db() does this:

  1. Creates a schema_migrations table if it doesn't exist yet.
     This table is the memory of what has already been applied.

  2. Reads the list of .sql files from migrations/ sorted by name.

  3. For each file, checks whether it has a row in schema_migrations.
     If yes  → skip it.
     If no   → execute it, then insert a row recording it as done.

This means:
  - Migrations run exactly once, ever.
  - Adding a new migration file is enough — restart and it runs.
  - You never run SQL manually in production.
  - You never touch a migration file after it has been committed.
    If you need to change something, write a new numbered file.

Connection model
----------------
psycopg2.pool.ThreadedConnectionPool is used because the app is
fundamentally synchronous (blocking calls are wrapped in asyncio.to_thread).
asyncpg would require rewriting every route for no gain at this scale.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import psycopg2
import psycopg2.extras
import psycopg2.pool

logger = logging.getLogger(__name__)

# Single pool — created once by init_db(), used by every get_connection() call
_pool: psycopg2.pool.ThreadedConnectionPool | None = None

# Project root / migrations/
_MIGRATIONS_DIR = Path(__file__).resolve().parent.parent.parent / "migrations"


# ──────────────────────────────────────────────────────────────────────────────
# Public API — used by the rest of the application
# ──────────────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """
    Create the connection pool, run pending migrations, build vector indexes.

    Call exactly once at application startup (main.py) before any
    repository function is used. Fails fast with a clear error message
    if PostgreSQL is unreachable — better than a cryptic failure at
    the first request.
    """
    from config.settings import settings

    global _pool

    logger.info(
        "Connecting to PostgreSQL → %s", _redact_url(settings.DATABASE_URL)
    )

    try:
        _pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=2,
            maxconn=settings.DATABASE_POOL_SIZE,
            dsn=settings.DATABASE_URL,
        )
    except psycopg2.OperationalError as exc:
        raise RuntimeError(
            f"Cannot connect to PostgreSQL at {_redact_url(settings.DATABASE_URL)}.\n"
            f"Is the database running?  Check DATABASE_URL in your .env file.\n"
            f"If using Docker: docker compose up -d\n"
            f"Original error: {exc}"
        ) from exc

    _run_migrations()
    _ensure_vector_indexes()
    logger.info("Database ready.")


@contextmanager
def get_connection() -> Generator[psycopg2.extensions.connection, None, None]:
    """
    Yield a pooled connection scoped to a single unit of work.

    Commits automatically on clean exit.
    Rolls back automatically on any exception.
    Always returns the connection to the pool.

    Usage::

        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT id, doc_name FROM documents WHERE user_id = %s", (uid,))
                rows = cur.fetchall()
        # committed here — no explicit conn.commit() needed
    """
    if _pool is None:
        raise RuntimeError(
            "Database pool not initialised. "
            "Ensure init_db() was called at startup before any DB access."
        )

    conn = _pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _pool.putconn(conn)


def vec_to_literal(v: list[float]) -> str:
    """
    Convert a Python float list to a pgvector literal string.

    pgvector accepts the text format  [x1,x2,...,xn]  which can be
    passed as a %s parameter with a ::vector cast in SQL:

        WHERE embedding <=> %s::vector

    This avoids the pgvector Python adapter dependency while keeping
    psycopg2's parameter escaping intact for the surrounding query.
    """
    return "[" + ",".join(f"{x:.8f}" for x in v) + "]"


# ──────────────────────────────────────────────────────────────────────────────
# Migration runner — the key piece the project was missing
# ──────────────────────────────────────────────────────────────────────────────

def _run_migrations() -> None:
    """
    Apply any migration files that have not been recorded yet.

    The schema_migrations table is the source of truth for what has run.
    Each migration is applied in a separate transaction so a failure in
    migration N does not affect migrations 1..N-1 which already committed.
    """
    _ensure_migrations_table()

    sql_files = sorted(_MIGRATIONS_DIR.glob("*.sql"))
    if not sql_files:
        logger.warning("No migration files found in %s", _MIGRATIONS_DIR)
        return

    applied = _get_applied_migrations()
    pending = [f for f in sql_files if f.name not in applied]

    if not pending:
        logger.info("Database schema is up to date (%d migration(s) already applied).", len(applied))
        return

    logger.info("%d pending migration(s) to apply.", len(pending))

    for migration_file in pending:
        logger.info("Applying migration: %s", migration_file.name)
        sql = migration_file.read_text(encoding="utf-8")

        # Each migration runs in its own transaction.
        # On failure: roll back this migration only, raise to abort startup.
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                # Record this migration as successfully applied
                cur.execute(
                    "INSERT INTO schema_migrations (filename) VALUES (%s)",
                    (migration_file.name,),
                )

        logger.info("Applied: %s", migration_file.name)

    logger.info("All migrations complete.")


def _ensure_migrations_table() -> None:
    """
    Create the schema_migrations tracking table if it doesn't exist.

    This is the one piece of DDL that runs unconditionally on every startup.
    It must be idempotent (IF NOT EXISTS) and it must run BEFORE anything
    else so the migration runner has somewhere to record its work.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    filename    VARCHAR(255) PRIMARY KEY,
                    applied_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
                )
            """)


def _get_applied_migrations() -> set[str]:
    """Return the set of migration filenames already recorded in the DB."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT filename FROM schema_migrations ORDER BY filename")
            return {row[0] for row in cur.fetchall()}


# ──────────────────────────────────────────────────────────────────────────────
# HNSW vector index setup
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_vector_indexes() -> None:
    """
    Create HNSW indexes on vector columns if they don't exist yet.

    Why is this not in a migration file?
    The HNSW index parameters — embedding dimension, m, ef_construction —
    come from settings.py. SQL migration files are static text; they cannot
    read Python config. Interpolating config values into SQL is done safely
    here after pydantic has validated all values as integers.

    These are CREATE INDEX IF NOT EXISTS so re-running on every startup
    is safe. Index creation on an already-indexed column is a no-op.
    """
    from config.settings import settings

    dim = int(settings.EMBEDDING_DIMENSION)
    m   = int(settings.PGVECTOR_HNSW_M)
    ef  = int(settings.PGVECTOR_HNSW_EF_CONSTRUCTION)

    indexes = [
        (
            "idx_chunks_embedding_hnsw",
            f"""
            CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
                ON document_chunks
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = {m}, ef_construction = {ef})
            """,
            "document_chunks",
        ),
        (
            "idx_turns_embedding_hnsw",
            f"""
            CREATE INDEX IF NOT EXISTS idx_turns_embedding_hnsw
                ON conversation_turns
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = {m}, ef_construction = {ef})
            """,
            "conversation_turns",
        ),
    ]

    with get_connection() as conn:
        with conn.cursor() as cur:
            for index_name, ddl, table in indexes:
                try:
                    cur.execute(ddl)
                    logger.debug("Vector index ready: %s on %s", index_name, table)
                except psycopg2.errors.UndefinedTable:
                    # Table doesn't exist yet — shouldn't happen after migrations,
                    # but guard defensively.
                    conn.rollback()
                    logger.warning("Table not found when creating index %s — migrations may not have run yet.", index_name)
                except psycopg2.Error as exc:
                    conn.rollback()
                    logger.warning("Could not create index %s: %s", index_name, exc)

    logger.info(
        "Vector indexes ready (dim=%d, m=%d, ef_construction=%d).", dim, m, ef
    )


# ──────────────────────────────────────────────────────────────────────────────
# Internal utility
# ──────────────────────────────────────────────────────────────────────────────

def _redact_url(url: str) -> str:
    """Replace the password in a DATABASE_URL with *** before logging it."""
    try:
        from urllib.parse import urlparse, urlunparse
        parsed = urlparse(url)
        if parsed.password:
            netloc = parsed.netloc.replace(f":{parsed.password}@", ":***@")
            return urlunparse(parsed._replace(netloc=netloc))
    except Exception:
        pass
    return url
