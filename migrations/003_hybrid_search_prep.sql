-- ============================================================
-- Migration 003 — Hybrid Search Preparation
--
-- Adds a generated tsvector column to document_chunks and a GIN
-- index over it. No application code change is needed now.
--
-- When hybrid search is implemented, queries will combine:
--
--     vector_score  (pgvector cosine similarity)
--   + keyword_score (full-text search via tsvector)
--
-- Example future query shape:
--
--     SELECT chunk_id, page_content,
--            (1 - (embedding <=> $1::vector)) * 0.7
--          + ts_rank(content_tsv, plainto_tsquery('english', $2)) * 0.3
--            AS combined_score
--     FROM document_chunks
--     WHERE user_id = $3
--     ORDER BY combined_score DESC
--     LIMIT 10;
--
-- The column is GENERATED ALWAYS AS ... STORED which means
-- PostgreSQL maintains it automatically on every INSERT/UPDATE.
-- No application code needs to populate it.
-- ============================================================

ALTER TABLE document_chunks
    ADD COLUMN IF NOT EXISTS content_tsv TSVECTOR
    GENERATED ALWAYS AS (to_tsvector('english', page_content)) STORED;

CREATE INDEX IF NOT EXISTS idx_chunks_content_tsv
    ON document_chunks
    USING GIN (content_tsv);
