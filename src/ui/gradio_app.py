"""
Gradio web interface for the Smart Contract Assistant v2.

Tabs:
  1. Upload & Manage  – Multi-file upload, document registry, removal
  2. Chat             – Conversational Q&A with session memory and source citations
  3. Summary          – One-click document summary
"""
from __future__ import annotations

import json
from pathlib import Path
import logging
import uuid
from typing import List

import gradio as gr

from config.settings import GRADIO_HOST, GRADIO_PORT, GRADIO_SHARE, UPLOAD_DIR
from src.ingestion import ingest_document
from src.ingestion.vector_store import (
    store_is_ready,
    get_ingested_documents,
    get_document_registry,
    remove_document,
)
from src.llm.qa_chain import answer_question
from src.llm.summarizer import summarize_document

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ── Tab 1: Upload & Manage ─────────────────────────────────────────────────

def handle_upload(file_objs) -> tuple[str, str]:
    """Handle single or multiple file uploads."""
    if not file_objs:
        return "⚠️ No files selected.", _refresh_doc_list()

    if not isinstance(file_objs, list):
        file_objs = [file_objs]

    messages = []
    for file_obj in file_objs:
        if file_obj is None:
            continue
        try:
            result = ingest_document(file_obj.name)
            messages.append(
                f"✅ **{result['filename']}** — "
                f"{result['num_pages']} pages, "
                f"{result['num_chunks']} chunks"
            )
        except Exception as e:
            logger.error("Upload error for %s: %s", file_obj.name, e)
            messages.append(f"❌ **{Path(file_obj.name).name}** — Error: {e}")

    status = "\n".join(messages) or "⚠️ No valid files processed."
    if all(m.startswith("✅") for m in messages):
        status = "### Ingestion Complete!\n\n" + status + "\n\nYou can now ask questions in the **Chat** tab."

    return status, _refresh_doc_list()


def _refresh_doc_list() -> str:
    """Return formatted markdown table of all ingested documents."""
    registry = get_document_registry()
    if not registry:
        return "*No documents ingested yet.*"

    lines = ["| Document | Pages | Chunks |", "|---|---|---|"]
    for name, meta in registry.items():
        pages = meta.get("num_pages", "?")
        chunks = meta.get("num_chunks", "?")
        lines.append(f"| `{name}` | {pages} | {chunks} |")

    return "\n".join(lines)


def handle_remove_doc(doc_name: str) -> tuple[str, str]:
    """Remove a document from the system."""
    if not doc_name.strip():
        return "⚠️ Please enter a document name.", _refresh_doc_list()
    success = remove_document(doc_name.strip())
    if success:
        return f"✅ Document `{doc_name}` removed.", _refresh_doc_list()
    return f"❌ Document `{doc_name}` not found.", _refresh_doc_list()


def get_doc_choices() -> List[str]:
    """Return list of document names for the dropdown."""
    return ["🌐 All Documents"] + get_ingested_documents()


# ── Tab 2: Chat ────────────────────────────────────────────────────────────

def _format_sources(sources: list) -> str:
    if not sources:
        return ""
    lines = ["\n\n---\n**📎 Sources:**"]
    for s in sources:
        lines.append(
            f"- 📄 `{s['source']}` | Page {s['page']} | Score: {s['score']:.3f}"
        )
        lines.append(f"  > *{s['snippet'][:140].strip()}…*")
    return "\n".join(lines)


def chat(
    user_message: str,
    chat_history: list,
    doc_selector: str,
    session_id: str,
) -> tuple[str, list]:
    """Process a chat message."""
    if not user_message.strip():
        return "", chat_history


    # Resolve doc_name from selector
    doc_name = None if doc_selector == "🌐 All Documents" else doc_selector

    # Gradio's history is ALREADY a list of dicts now! 
    # We can just copy it directly for the QA chain.
    lc_history = chat_history.copy()

    try:
        result = answer_question(
            question=user_message,
            history=lc_history,
            doc_name=doc_name,
            session_id=session_id,
        )
        answer = result["answer"]
        sources_text = _format_sources(result["sources"])
        bot_reply = answer + sources_text
    except Exception as e:
        logger.error("Chat error: %s", e)
        bot_reply = f"❌ Error: {e}"

    # Append dictionaries instead of lists
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": bot_reply})
    
    return "", chat_history


def clear_chat() -> tuple[str, list, str, str]:
    new_sid = str(uuid.uuid4())[:8]
    return "", [], f"🔑 New session: `{new_sid}`", new_sid


# ── Tab 3: Summary ─────────────────────────────────────────────────────────

def handle_summarize(doc_selector: str) -> str:
    """Generate summary for the selected document."""
    docs = get_ingested_documents()
    if not docs:
        return "⚠️ No documents ingested. Please upload a document first."

    # If "all" is selected, summarize the first document
    fname = docs[0] if doc_selector == "🌐 All Documents" else doc_selector

    try:
        summary = summarize_document(fname)
        return f"## 📋 Summary: `{fname}`\n\n{summary}"
    except Exception as e:
        logger.error("Summary error: %s", e)
        return f"❌ Summarization failed: {e}"



# ── Build UI ───────────────────────────────────────────────────────────────

CSS = """
.gradio-container { max-width: 1080px !important; margin: auto; }
#chatbot { height: 520px; overflow-y: auto; }
footer { display: none !important; }
.doc-table td, .doc-table th { font-size: 0.85rem; }
"""

DESCRIPTION = """
# 📜 Smart Contract Q&A Assistant v2
Multi-document RAG pipeline — upload contracts, ask questions, get grounded answers with citations.
"""

with gr.Blocks(theme=gr.themes.Soft(), css=CSS, title="Smart Contract Assistant v2") as demo:
    gr.Markdown(DESCRIPTION)

    # Per-user session state
    session_state = gr.State(lambda: str(uuid.uuid4())[:8])

    with gr.Tabs():

        # ── Tab 1: Upload & Manage ─────────────────────────────────────────
        with gr.Tab("📤 Upload & Manage"):
            gr.Markdown("### Upload Documents (PDF or DOCX)")
            gr.Markdown(
                "You can upload **multiple files at once**. Each gets its own vector store "
                "collection (hierarchical indexing). All are queryable together or individually."
            )
            with gr.Row():
                file_input = gr.File(
                    label="Drop PDF / DOCX files here",
                    file_types=[".pdf", ".docx", ".doc"],
                    file_count="multiple",
                )
                with gr.Column():
                    upload_btn = gr.Button("📥 Ingest All Uploaded Files", variant="primary")
                    upload_status = gr.Markdown("*No files uploaded yet.*")

            gr.Markdown("---\n### Ingested Documents")
            doc_table = gr.Markdown(_refresh_doc_list())
            refresh_btn = gr.Button("🔄 Refresh List", variant="secondary")

            with gr.Row():
                remove_input = gr.Textbox(
                    label="Remove document (enter exact filename)",
                    placeholder="e.g., contract.pdf",
                    scale=4,
                )
                remove_btn = gr.Button("🗑 Remove", variant="stop", scale=1)
            remove_status = gr.Markdown("")

            upload_btn.click(
                fn=handle_upload,
                inputs=[file_input],
                outputs=[upload_status, doc_table],
            )
            file_input.upload(
                fn=handle_upload,
                inputs=[file_input],
                outputs=[upload_status, doc_table],
            )
            refresh_btn.click(fn=_refresh_doc_list, outputs=[doc_table])
            remove_btn.click(
                fn=handle_remove_doc,
                inputs=[remove_input],
                outputs=[remove_status, doc_table],
            )

        # ── Tab 2: Chat ────────────────────────────────────────────────────
        with gr.Tab("💬 Chat"):
            gr.Markdown("### Ask Questions About Your Contracts")
            with gr.Row():
                doc_selector_chat = gr.Dropdown(
                    choices=get_doc_choices(),
                    value="🌐 All Documents",
                    label="📂 Document scope",
                    interactive=True,
                )
                refresh_docs_btn = gr.Button("🔄", scale=0)

            chatbot = gr.Chatbot(
                elem_id="chatbot",
                label="Contract Assistant",
            )

            session_info = gr.Markdown("🔑 Session active")

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="e.g. What are the termination conditions?",
                    label="Your question",
                    lines=2,
                    scale=5,
                )
                send_btn = gr.Button("Send ↵", variant="primary", scale=1)

            with gr.Row():
                clear_btn = gr.Button("🗑 New Session", variant="secondary")
                gr.Markdown(
                    "*Conversation history is persisted semantically — relevant past turns "
                    "from previous sessions are retrieved automatically.*"
                )

            def _refresh_dropdown():
                return gr.Dropdown(choices=get_doc_choices())

            send_btn.click(
                fn=chat,
                inputs=[msg_input, chatbot, doc_selector_chat, session_state],
                outputs=[msg_input, chatbot],
            )
            msg_input.submit(
                fn=chat,
                inputs=[msg_input, chatbot, doc_selector_chat, session_state],
                outputs=[msg_input, chatbot],
            )
            clear_btn.click(
                fn=clear_chat,
                outputs=[msg_input, chatbot, session_info, session_state],
            )
            refresh_docs_btn.click(
                fn=_refresh_dropdown,
                outputs=[doc_selector_chat],
            )

        # ── Tab 3: Summary ─────────────────────────────────────────────────
        with gr.Tab("📋 Summary"):
            gr.Markdown("### One-Click Contract Summary")
            gr.Markdown(
                "Generates a structured overview of key parties, obligations, "
                "dates, and risks. Uses map-reduce for large documents."
            )
            with gr.Row():
                doc_selector_summary = gr.Dropdown(
                    choices=get_doc_choices(),
                    value="🌐 All Documents",
                    label="📂 Select document",
                    interactive=True,
                )
                refresh_summary_btn = gr.Button("🔄", scale=0)
            summarize_btn = gr.Button("🔍 Generate Summary", variant="primary")
            summary_output = gr.Markdown("*Click the button above to generate a summary.*")

            summarize_btn.click(
                fn=handle_summarize,
                inputs=[doc_selector_summary],
                outputs=[summary_output],
            )
            refresh_summary_btn.click(
                fn=_refresh_dropdown,
                outputs=[doc_selector_summary],
            )

def launch():
    demo.launch(
        server_name=GRADIO_HOST,
        server_port=GRADIO_PORT,
        share=GRADIO_SHARE,
        show_error=True,
    )


if __name__ == "__main__":
    launch()
