-- ============================================================
-- Smart Contract Assistant — Initial Schema
-- PostgreSQL + pgvector
--
-- Run via: python -m scripts.init_db
-- Or automatically at startup via database.init_db()
-- ============================================================

-- pgvector extension — must be installed in PostgreSQL first:
--   CREATE EXTENSION IF NOT EXISTS vector;
-- (requires superuser or pg_extension_owner privilege)
CREATE EXTENSION IF NOT EXISTS vector;


-- ============================================================
-- Documents registry
-- Replaces the JSON file (doc_registry.json) from the
-- previous Chroma-based architecture.
-- ============================================================
CREATE TABLE IF NOT EXISTS documents (
    id              SERIAL          PRIMARY KEY,
    user_id         VARCHAR(64)     NOT NULL,
    doc_name        VARCHAR(255)    NOT NULL,
    content_hash    VARCHAR(64)     NOT NULL,
    num_pages       INTEGER         NOT NULL DEFAULT 0,
    num_chunks      INTEGER         NOT NULL DEFAULT 0,
    file_path       TEXT,
    ingested_at     TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_user_doc UNIQUE (user_id, doc_name)
);

CREATE INDEX IF NOT EXISTS idx_documents_user_id
    ON documents (user_id);

CREATE INDEX IF NOT EXISTS idx_documents_content_hash
    ON documents (content_hash);


-- ============================================================
-- Document chunks + embeddings
-- Replaces the global Chroma collection.
-- Embedding dimension is set at index-creation time by
-- database.py using settings.EMBEDDING_DIMENSION.
-- ============================================================
CREATE TABLE IF NOT EXISTS document_chunks (
    id                  SERIAL          PRIMARY KEY,
    chunk_id            VARCHAR(40)     NOT NULL UNIQUE,   -- SHA-256 truncated
    doc_fk              INTEGER         NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    user_id             VARCHAR(64)     NOT NULL,          -- denormalised for fast WHERE
    doc_name            VARCHAR(255)    NOT NULL,          -- denormalised (= documents.doc_name)
    page_content        TEXT            NOT NULL,
    embedding           vector,                            -- dimension set per-index (see database.py)
    source              VARCHAR(255),                      -- original filename
    file_type           VARCHAR(10),
    page                INTEGER,
    chunk_index         INTEGER,
    global_chunk_index  INTEGER,
    chunk_size          INTEGER,
    chunk_overlap       INTEGER,
    uploaded_at         TIMESTAMPTZ
);

-- Filter indexes — used by every retrieval and history query
CREATE INDEX IF NOT EXISTS idx_chunks_user_id
    ON document_chunks (user_id);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_fk
    ON document_chunks (doc_fk);

CREATE INDEX IF NOT EXISTS idx_chunks_user_source
    ON document_chunks (user_id, source);

-- NOTE: The HNSW vector index is created separately in database.py
-- using EMBEDDING_DIMENSION and HNSW parameters from settings.


-- ============================================================
-- Conversation turns (chat history)
-- Replaces the Chroma chat_history collection.
-- Supports both recency-based and semantic retrieval.
-- ============================================================
CREATE TABLE IF NOT EXISTS conversation_turns (
    id                  SERIAL          PRIMARY KEY,
    user_id             VARCHAR(64)     NOT NULL,
    conversation_id     VARCHAR(64),
    role                VARCHAR(20)     NOT NULL CHECK (role IN ('user', 'assistant')),
    content             TEXT            NOT NULL,
    embedding           vector,                            -- for semantic history search
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_turns_user_id
    ON conversation_turns (user_id);

CREATE INDEX IF NOT EXISTS idx_turns_conversation_id
    ON conversation_turns (conversation_id);

CREATE INDEX IF NOT EXISTS idx_turns_created_at
    ON conversation_turns (created_at);

-- NOTE: The HNSW index on conversation_turns.embedding is also
-- created in database.py alongside the document_chunks index.
