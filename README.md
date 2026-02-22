# 📜 File Fellow

A local, multi-document RAG (Retrieval-Augmented Generation) assistant for contracts, insurance policies, and reports. Upload PDF or DOCX files and interact with them through a conversational chat interface — with grounded answers, source citations, and persistent session memory.

---

## Features

- **Multi-document ingestion** — Upload multiple PDF/DOCX files at once. Each document gets its own vector store collection, plus a global store for cross-document queries.
- **Conversational Q&A** — Ask questions about your documents or just chat naturally. The assistant handles both without switching modes.
- **Source citations** — Every document-grounded answer includes the filename, page number, and a relevance score.
- **Document summarization** — One-click structured summaries using map-reduce for large files.
- **Semantic session memory** — Conversation history is embedded and persisted in a vector store. Relevant past turns are retrieved automatically in future sessions.
- **Per-user session isolation** — Each browser session gets a unique ID; history is scoped to that session.
- **FastAPI backend + LangServe playground** — REST endpoints for all operations, plus an interactive `/qa-langserve/playground` for testing the RAG chain directly.
- **LLM-as-Judge evaluation** — A full synthetic evaluation pipeline that samples document chunks, generates ground-truth Q&A pairs, runs the RAG agent, and scores the results.

---

## Architecture

```
main.py
├── src/
│   ├── api/
│   │   └── server.py          # FastAPI + LangServe endpoints
│   ├── ingestion/
│   │   ├── __init__.py        # Orchestrates parse → chunk → embed → store
│   │   ├── parser.py          # PDF (PyMuPDF / pdfplumber) and DOCX parsing
│   │   ├── chunker.py         # Recursive character text splitting
│   │   ├── embedder.py        # SentenceTransformers (local, no API key)
│   │   └── vector_store.py    # Chroma/FAISS multi-store management + registry
│   ├── retrieval/
│   │   └── retriever.py       # Semantic similarity search with threshold filtering
│   ├── llm/
│   │   ├── llm_factory.py     # Groq LLM factory (cached)
│   │   ├── prompts.py         # Q&A, summary, question-gen, and judge prompts
│   │   ├── qa_chain.py        # Full RAG chain with history and citations
│   │   └── summarizer.py      # Map-reduce document summarization
│   ├── memory/
│   │   └── history_store.py   # Semantic chat history persistence (Chroma)
│   └── ui/
│       └── gradio_app.py      # Gradio web interface (3 tabs)
├── scripts/
│   └── evaluate.py            # LLM-as-Judge evaluation pipeline
└── config/
    ├── config.yaml            # Default settings
    └── settings.py            # Loads yaml → .env → env vars
```

### RAG Pipeline

```
User Message
     │
     ▼
 HistoryStore.retrieve_relevant()      ← semantic search over past turns
     │
     ▼
 retrieve_chunks()                     ← similarity search (per-doc or global)
     │
     ▼
 QA_PROMPT (context + history + question)
     │
     ▼
 Groq LLM (llama-3.3-70b-versatile)
     │
     ▼
 Answer + Source Citations
     │
     ▼
 HistoryStore.add_turn()               ← persist this turn for future retrieval
```

### Vector Store Design

| Collection            | Contents                    | Used for                   |
| --------------------- | --------------------------- | -------------------------- |
| `contract_<doc_name>` | Chunks from one document    | Per-document Q&A           |
| `all_contract_s`      | Chunks from all documents   | Cross-document Q&A         |
| `chat_history`        | Embedded conversation turns | Semantic history retrieval |

---

## Setup

### Prerequisites

- Python 3.10+
- uv
- A [Groq API key](https://console.groq.com/) (free tier available)

### Installation

```bash
# Clone the repository
git clone https://github.com/AhmedMaherTohmay/File-Fellow.git
cd file-fellow

# Create a virtual environment and install all dependencies
uv sync
```

### Configuration

Create a `.env` file in the project root:

```env
LLM_KEY=your_groq_api_key_here
```

All other settings have sensible defaults in `config/config.yaml`. Override any of them via environment variables or the yaml file:

```yaml
# config/config.yaml
llm_provider: groq
groq_model_id: llama-3.3-70b-versatile
embedding_provider: sentence_transformers
sentence_transformer_model: all-MiniLM-L6-v2
chunk_size: 800
chunk_overlap: 150
top_k: 5
similarity_threshold: 0.25
```

---

## Running

### Full application (API + UI)

```bash
uv run python main.py (uv)
python main.py
```

- Gradio UI: [http://localhost:7860](http://localhost:7860/)
- FastAPI docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- LangServe playground: [http://localhost:8000/qa-langserve/playground](http://localhost:8000/qa-langserve/playground)

### UI only

```bash
uv run python main.py --ui
python main.py --ui
```

### API only

```bash
uv run python main.py --api
python main.py --api
```

---

## Using the Interface

### Tab 1 — Upload & Manage

1. Drag and drop one or more PDF or DOCX files onto the upload area.
2. Files are ingested automatically on drop, or click **Ingest Uploaded Files** .
3. The document table shows each file with its page and chunk counts.
4. To remove a document, enter its exact filename and click **Remove** .

### Tab 2 — Chat

- Select a document scope from the dropdown (`All Documents` or a specific file).
- Type your question and press Enter or click **Send** .
- Works for casual conversation too — no document required.
- Click **New Session** to clear history and start a fresh session ID.

### Tab 3 — Summary

- Select a document from the dropdown.
- Click **Generate Summary** for a structured overview including parties, obligations, dates, and risks.
- Large documents are summarized in segments using map-reduce then merged.

---

## API Reference

| Method   | Endpoint            | Description                          |
| -------- | ------------------- | ------------------------------------ |
| `GET`    | `/health`           | System status and document count     |
| `POST`   | `/ingest`           | Upload and ingest a single file      |
| `POST`   | `/ingest/batch`     | Upload and ingest multiple files     |
| `GET`    | `/documents`        | List all ingested documents          |
| `DELETE` | `/documents/{name}` | Remove a document                    |
| `POST`   | `/qa`               | Ask a question (single or cross-doc) |
| `POST`   | `/summarize`        | Summarize a document                 |

**Example Q&A request:**

```bash
curl -X POST http://localhost:8000/qa \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the termination conditions?",
    "doc_name": "contract.pdf",
    "session_id": "user-123"
  }'
```

---

## Evaluation

The evaluation pipeline implements an **LLM-as-Judge** formulation across four steps:

1. **Sample** — Two random chunks are drawn from the document's vector store.
2. **Generate** — A generator LLM produces a synthetic `(question, ground_truth_answer)` pair grounded in those chunks.
3. **Retrieve & Answer** — The live RAG agent answers the same question, returning a candidate answer and the chunks it retrieved.
4. **Judge** — A judge LLM compares the candidate answer against the ground truth and scores it on three axes:
   - **Faithfulness** (0–10): Is every claim grounded in the retrieved context?
   - **Relevance** (0–10): Does the answer address the question directly?
   - **Correctness** (0–10): How well does it align with the ground-truth answer?
   - **Hallucination flag** : `true` if any claim is unsupported by the retrieved context.

### Running the evaluation

First, ingest the target document through the UI or API. Then:

```bash
# Default: 5 samples from Fraud_Detection_System_Design.pdf
python scripts/evaluate.py

# Custom options
python scripts/evaluate.py \
  --doc_name Fraud_Detection_System_Design.pdf \
  --num_samples 10 \
  --qa_per_sample 2 \
  --output reports/eval_report.json
```

### Sample output

```
=======================================================
  LLM-AS-JUDGE EVALUATION REPORT
=======================================================
  Num Samples Evaluated          10
  Avg Faithfulness               8.4
  Avg Relevance                  7.9
  Avg Correctness                7.6
  Hallucination Rate             0.1
  Avg Composite Score            7.97
=======================================================
```

The full per-question report is saved to the `--output` path as JSON.

---

## Project Structure Notes

- **No external vector DB required** — Chroma runs locally and persists to `vector_store/`.
- **No API key for embeddings** — `all-MiniLM-L6-v2` runs fully locally via SentenceTransformers.
- **Only Groq API key needed** — Used for both the chat LLM and the evaluation judge.
- **Stateless vector store access** — Each Chroma store is opened fresh per request (Windows-safe; avoids file lock conflicts).
- **Modular design** — Each concern (parsing, chunking, embedding, retrieval, LLM, memory, UI) is a separate module with a clean interface.

---

## Configuration Reference

| Setting                      | Default                   | Description                   |
| ---------------------------- | ------------------------- | ----------------------------- |
| `LLM_KEY`                    | _(required)_              | Groq API key                  |
| `GROQ_MODEL_ID`              | `llama-3.3-70b-versatile` | Groq model to use             |
| `LLM_TEMPERATURE`            | `0.0`                     | LLM sampling temperature      |
| `LLM_MAX_TOKENS`             | `1024`                    | Max tokens per response       |
| `SENTENCE_TRANSFORMER_MODEL` | `all-MiniLM-L6-v2`        | Local embedding model         |
| `CHUNK_SIZE`                 | `800`                     | Characters per chunk          |
| `CHUNK_OVERLAP`              | `150`                     | Overlap between chunks        |
| `TOP_K`                      | `5`                       | Chunks to retrieve per query  |
| `SIMILARITY_THRESHOLD`       | `0.25`                    | Minimum relevance score (0–1) |
| `GRADIO_PORT`                | `7860`                    | UI server port                |
| `API_PORT`                   | `8000`                    | FastAPI server port           |

---

## Tech Stack

| Component     | Library                                                                                           |
| ------------- | ------------------------------------------------------------------------------------------------- |
| LLM           | [Groq](https://groq.com/)via `langchain-groq`                                                     |
| Embeddings    | [SentenceTransformers](https://www.sbert.net/)via `langchain-community`                           |
| Vector store  | [Chroma](https://www.trychroma.com/)(default) / FAISS                                             |
| RAG framework | [LangChain](https://www.langchain.com/)                                                           |
| API server    | [FastAPI](https://fastapi.tiangolo.com/)+[LangServe](https://python.langchain.com/docs/langserve) |
| Web UI        | [Gradio](https://www.gradio.app/)                                                                 |
| PDF parsing   | [PyMuPDF](https://pymupdf.readthedocs.io/)/[pdfplumber](https://github.com/jsvine/pdfplumber)     |
| DOCX parsing  | [python-docx](https://python-docx.readthedocs.io/)                                                |
