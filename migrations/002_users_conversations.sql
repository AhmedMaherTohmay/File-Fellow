-- ============================================================
-- Migration 002 — Users and Conversations
--
-- Adds proper relational anchors for user_id and conversation_id.
-- Both were previously unanchored VARCHAR strings. This migration
-- creates the home rows and wires up the foreign keys.
--
-- Why VARCHAR(64) primary keys instead of UUID?
--   - Existing user_id values are Gradio session strings (e.g. "default")
--   - chunk_id is a content-derived SHA-256 hash, not a random UUID
--   - Switching to UUID would require a data migration on existing rows
--   - VARCHAR(64) is large enough for UUIDs if we add auth later
-- ============================================================


-- ── Users ────────────────────────────────────────────────────
-- Minimal anchor table. When authentication is added, expand
-- this with email, password_hash, role, etc.
-- We do NOT add NOT NULL on created_at because it has a default.

CREATE TABLE IF NOT EXISTS users (
    id          VARCHAR(64)     PRIMARY KEY,
    created_at  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);


-- ── Conversations ─────────────────────────────────────────────
-- Represents a single chat session belonging to a user.
-- Cascades delete so removing a user wipes their conversations
-- and — via the turns FK below — their full message history.

CREATE TABLE IF NOT EXISTS conversations (
    id          VARCHAR(64)     PRIMARY KEY,
    user_id     VARCHAR(64)     NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    created_at  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_conversations_user_id
    ON conversations (user_id);


-- ── Back-fill users for any existing data ────────────────────
-- documents and conversation_turns may already have user_id values
-- from the previous unanchored schema. Insert a user row for each
-- unique value so the FK constraints below don't fail on existing data.
-- The ON CONFLICT DO NOTHING makes this safe to re-run.

INSERT INTO users (id)
    SELECT DISTINCT user_id FROM documents
    ON CONFLICT DO NOTHING;

INSERT INTO users (id)
    SELECT DISTINCT user_id FROM conversation_turns
    WHERE user_id IS NOT NULL
    ON CONFLICT DO NOTHING;


-- ── Back-fill conversations for existing turns ────────────────
-- conversation_turns may reference conversation_ids that don't
-- exist as rows yet. Create placeholder rows for them.

INSERT INTO conversations (id, user_id)
    SELECT DISTINCT
        t.conversation_id,
        t.user_id
    FROM conversation_turns t
    WHERE t.conversation_id IS NOT NULL
      AND NOT EXISTS (
          SELECT 1 FROM conversations c WHERE c.id = t.conversation_id
      )
    ON CONFLICT DO NOTHING;


-- ── Add foreign key constraints ───────────────────────────────
-- Only add if they don't already exist to keep this idempotent.

DO $$
BEGIN
    -- documents.user_id → users.id
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'fk_documents_user'
    ) THEN
        ALTER TABLE documents
            ADD CONSTRAINT fk_documents_user
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE;
    END IF;

    -- conversation_turns.conversation_id → conversations.id
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'fk_turns_conversation'
    ) THEN
        ALTER TABLE conversation_turns
            ADD CONSTRAINT fk_turns_conversation
            FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE;
    END IF;
END
$$;
