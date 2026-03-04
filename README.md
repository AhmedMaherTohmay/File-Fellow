
# Document Q&A Assistant

A production-grade RAG (Retrieval-Augmented Generation) system for querying PDF and DOCX documents through a conversational interface. Ask questions grounded in your uploaded documents, get cited answers, and maintain context across sessions.

---

## Features

* **Multi-document RAG** — upload multiple PDFs or DOCX files; query across all of them or scope to a single document
* **Source citations** — every answer includes the source filename, page number, and a relevance score
* **Persistent session memory** — conversation history is embedded and retrieved semantically across browser sessions using a stable User ID
* **Duplicate detection** — SHA-256 content hashing prevents the same document being ingested twice under different filenames
* **Document summarization** — map-reduce summarization that respects sentence boundaries
* **LLM-as-Judge evaluation** — automated pipeline that generates synthetic Q&A pairs and scores RAG answers on faithfulness, relevance, and correctness
* **REST API** — full FastAPI backend with LangServe playground for direct integration
* **Thread-safe writes** — a process-level write lock prevents SQLite contention when the API and UI run concurrently

---

## Architecture

```
smart_contract_assistant/
├── config/
│   ├── config.yaml          # All tuneable parameters
│   └── settings.py          # Loads config + .env into typed constants
│
├── src/
│   ├── core/                # Shared utilities and exception types
│   │   ├── utils.py         # sanitize_filename, normalise_score, file_content_hash
│   │   └── exceptions.py    # Typed exception hierarchy
│   │
│   ├── ingestion/           # Document processing pipeline
│   │   ├── pipeline.py      # Orchestrates: sanitize → dedup → parse → chunk → store
│   │   ├── parser.py        # PDF (PyMuPDF + pdfplumber fallback) and DOCX parsing
│   │   ├── chunker.py       # RecursiveCharacterTextSplitter with stable chunk IDs
│   │   └── embedder.py      # SentenceTransformer embeddings (local, no API key)
│   │
│   ├── storage/             # All persistence — single source of truth for Chroma
│   │   ├── document_store.py  # Global vector store, registry, write lock, migrations
│   │   └── history_store.py   # Per-user conversation history with TTL purge
│   │
│   ├── retrieval/
│   │   └── retriever.py     # Semantic search with score normalisation and threshold
│   │
│   ├── llm/                 # LLM infrastructure only
│   │   ├── llm_factory.py       # Cached Groq client (get_llm, get_llm_for_eval)
│   │   └── prompts.py       # Q&A, summarization, question-gen, and judge prompts
│   │
│   ├── services/            # Business logic — orchestrates storage + retrieval + LLM
│   │   ├── qa.py            # answer_question(): RAG chain with history injection
│   │   └── summary.py       # summarize_document(): chunk-aware map-reduce
│   │
│   ├── api/
│   │   ├── server.py        # FastAPI routes
│   │   └── schema.py        # Pydantic request/response models
│   │
│   └── ui/
│       ├── app.py           # gr.Blocks layout — mounts tabs, owns shared state
│       ├── styles.py        # CSS design system + header HTML
│       ├── formatters.py    # Pure HTML rendering functions (no Gradio imports)
│       ├── session.py       # User ID and conversation ID management
│       └── tabs/
│           ├── upload.py    # Upload & Manage tab
│           ├── chat.py      # Chat tab
│           └── summary.py   # Summary tab
│
├── scripts/
│   └── evaluate.py          # LLM-as-Judge evaluation pipeline
│
├── main.py                  # Entrypoint — launches API + UI, runs startup tasks
└── .env.example
```

### Key design decisions

**Single global vector store.** All document chunks live in one Chroma collection tagged with `source` metadata. Per-document scoping is handled at query time via metadata filtering. This avoids the N-collections problem where N uploaded documents creates N Chroma stores.

**Two-tier history.** In-session history (Gradio state, ephemeral) handles the current conversation. Semantic history (Chroma, persistent) retrieves relevant turns from *previous* sessions. The current conversation is explicitly excluded from semantic retrieval to prevent the LLM seeing the same dialogue twice.

**Score normalisation.** LangChain's Chroma relevance scores can be negative when vectors are not unit-normalised. All raw scores are mapped to `[0, 1]` via `(raw + 1) / 2` before thresholding. Both the document retriever and the history store use the same normalisation function from `core/utils.py`, so threshold constants mean the same thing in both places.

---

## Requirements

* Python 3.10+
* A [Groq](https://console.groq.com/) API key (free tier is sufficient)

---

## Installation

```bash
git clone https://github.com/AhmedMaherTohmay/File-Fellow
cd File-Fellow

# Create a virtual environment and install all dependencies
uv sync
```

Copy the environment file and add your Groq API key:

```bash
cp .env.example .env
```

```env
LLM_KEY=gsk_your_groq_api_key_here
```

---

## Usage

### Launch both servers (default)

```bash
uv run main.py
```

* Gradio UI → `http://localhost:7860`
* FastAPI + docs → `http://localhost:8000/docs`
* LangServe playground → `http://localhost:8000/qa-langserve/playground`

### Launch individually

```bash
uv run main.py --ui     # Gradio only
uv run main.py --api    # FastAPI only
```

### Workflow

1. Go to the **Upload & Manage** tab and drop one or more PDF or DOCX files
2. Switch to  **Chat** , enter a User ID or leave blank to generate one — **save this ID** to continue your session later
3. Ask questions; answers include cited sources with page numbers and relevance scores
4. Click **New Conversation** to start fresh while keeping your session history available as context
5. Use the **Summary** tab for a structured overview of any ingested document

---

## Configuration

All parameters live in `config/config.yaml`. Any value can be overridden with an environment variable of the same name in uppercase (e.g. `TOP_K=10`).

| Parameter                   | Default                     | Description                                    |
| --------------------------- | --------------------------- | ---------------------------------------------- |
| `groq_model_id`           | `llama-3.3-70b-versatile` | Groq model to use                              |
| `llm_temperature`         | `0.0`                     | LLM temperature (0 = deterministic)            |
| `chunk_size`              | `800`                     | Max characters per chunk                       |
| `chunk_overlap`           | `150`                     | Overlap between adjacent chunks                |
| `top_k`                   | `5`                       | Chunks retrieved per query                     |
| `similarity_threshold`    | `0.30`                    | Min normalised score (0–1) to include a chunk |
| `history_score_threshold` | `0.25`                    | Min score for semantic history retrieval       |
| `history_ttl_days`        | `7`                       | Days before history turns are purged           |
| `max_session_turns`       | `6`                       | Recent turns injected into the LLM prompt      |
| `vector_store_backend`    | `chroma`                  | `chroma`or `faiss`                         |
| `api_port`                | `8000`                    | FastAPI port                                   |
| `gradio_port`             | `7860`                    | Gradio UI port                                 |

---

## API Reference

All endpoints accept and return JSON. Full interactive docs at `/docs`.

| Method     | Endpoint              | Description                      |
| ---------- | --------------------- | -------------------------------- |
| `GET`    | `/health`           | System status and document count |
| `POST`   | `/ingest`           | Upload and ingest a single file  |
| `POST`   | `/ingest/batch`     | Upload and ingest multiple files |
| `GET`    | `/documents`        | List all ingested documents      |
| `DELETE` | `/documents/{name}` | Remove a document                |
| `POST`   | `/qa`               | Ask a question                   |
| `POST`   | `/summarize`        | Summarize a document             |

### POST /qa

```json
{
  "question": "What are the termination conditions?",
  "doc_name": "contract.pdf",
  "user_id": "abc123",
  "conversation_id": "uuid-here",
  "history": [],
  "session_id": "default"
}
```

`doc_name` is optional — omit to search across all documents. `user_id` and `conversation_id` are optional but enable persistent session memory.

---

## Evaluation

The evaluation pipeline measures RAG quality using an LLM-as-judge approach. It requires at least one ingested document.

```bash
python scripts/evaluate.py
```

```bash
# Custom options
python scripts/evaluate.py \
  --doc_name contract.pdf \
  --num_samples 10 \
  --qa_per_sample 2 \
  --output reports/eval.json
```

**Pipeline per sample:**

1. Draw two random chunks from the vector store
2. Feed both chunks to the LLM → generate synthetic (question, ground-truth answer) pairs
3. Run the same question through the live RAG pipeline → candidate answer
4. Feed (question, ground truth, retrieved context, candidate answer) to a judge LLM
5. Judge scores on  **Faithfulness** ,  **Relevance** , and **Correctness** (0–10 each) and flags hallucinations

Results are printed to stdout and saved as a JSON report.

---

## Session Memory

Each user receives a stable **User ID** (12-character UUID prefix) on first connect. This ID persists across browser sessions and is the key to long-term memory.

* **Enter your User ID** when reconnecting to resume context from previous conversations
* **Leave it blank** to receive a new ID (a fresh user with no history)
* **New Conversation** starts a clean chat while preserving your ID — past sessions remain available as semantic context

History turns older than `history_ttl_days` (default: 7 days) are purged automatically at startup. Document chunks are never affected by this purge.

---

## Concurrency

The FastAPI server and Gradio UI run as threads in the same OS process. All Chroma write operations are serialised by a `threading.RLock` defined in `src/storage/document_store.py`. This prevents SQLite `database is locked` errors under concurrent upload and query traffic.

> **Note:** If you deploy with multiple OS processes (e.g. gunicorn workers), replace `PersistentClient` with `chromadb.HttpClient` pointing at a dedicated Chroma server process — a threading lock cannot protect across process boundaries.

---

## Supported File Types

| Format | Parser                                   | Notes                                                       |
| ------ | ---------------------------------------- | ----------------------------------------------------------- |
| PDF    | PyMuPDF (primary), pdfplumber (fallback) | Falls back automatically if PyMuPDF returns empty text      |
| DOCX   | python-docx                              | Extracts paragraphs and tables; groups into synthetic pages |

Maximum file size: 50 MB per document.
