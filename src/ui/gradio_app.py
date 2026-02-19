"""
Gradio web interface for the Smart Contract Assistant.

Tabs:
  1. Upload & Ingest  – drag-and-drop PDF/DOCX
  2. Chat             – conversational Q&A with source citations
  3. Summary          – one-click contract summary
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List

import gradio as gr

# Ensure project root is on PYTHONPATH when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import GRADIO_HOST, GRADIO_PORT, GRADIO_SHARE, UPLOAD_DIR
from src.ingestion import ingest_document
from src.llm.qa_chain import answer_question
from src.llm.summarizer import summarize_document
from src.ingestion.vector_store import store_is_ready

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── State helpers ─────────────────────────────────────────────────────────

_current_filename: str | None = None


def _set_filename(name: str | None) -> None:
    global _current_filename
    _current_filename = name


def _get_filename() -> str | None:
    return _current_filename


# ── Tab 1: Upload & Ingest ─────────────────────────────────────────────────

def handle_upload(file_obj) -> str:
    """Called when the user uploads a file."""
    if file_obj is None:
        return "⚠️ No file selected."
    try:
        result = ingest_document(file_obj.name)
        _set_filename(result["filename"])
        return (
            f"✅ **Ingestion complete!**\n\n"
            f"- 📄 File: `{result['filename']}`\n"
            f"- 📑 Pages extracted: **{result['num_pages']}**\n"
            f"- 🧩 Chunks created: **{result['num_chunks']}**\n\n"
            f"You can now ask questions in the **Chat** tab."
        )
    except Exception as e:
        logger.error("Upload error: %s", e)
        return f"❌ Ingestion failed: {e}"


# ── Tab 2: Chat ────────────────────────────────────────────────────────────

def _format_sources(sources: list) -> str:
    if not sources:
        return ""
    lines = ["\n\n---\n**📎 Sources:**"]
    for s in sources:
        lines.append(
            f"- `{s['source']}` | Page {s['page']} | Chunk {s['chunk_id']} | Score: {s['score']:.3f}"
        )
        lines.append(f"  > *{s['snippet'][:120]}…*")
    return "\n".join(lines)


def chat(user_message: str, chat_history: list) -> tuple[str, list]:
    """Process a chat message and return updated history."""
    if not user_message.strip():
        return "", chat_history

    if not store_is_ready():
        bot_reply = "⚠️ Please upload and ingest a document first (see the **Upload** tab)."
        chat_history.append((user_message, bot_reply))
        return "", chat_history

    # Convert Gradio history to LangChain-style
    lc_history = []
    for human, ai in chat_history:
        lc_history.append({"role": "user", "content": human})
        lc_history.append({"role": "assistant", "content": ai})

    try:
        result = answer_question(user_message, lc_history)
        answer = result["answer"]
        sources_text = _format_sources(result["sources"])
        bot_reply = answer + sources_text
    except Exception as e:
        logger.error("Chat error: %s", e)
        bot_reply = f"❌ Error: {e}"

    chat_history.append((user_message, bot_reply))
    return "", chat_history


def clear_chat() -> tuple[str, list]:
    return "", []


# ── Tab 3: Summary ─────────────────────────────────────────────────────────

def handle_summarize() -> str:
    fname = _get_filename()
    if not fname:
        return "⚠️ No document ingested yet. Please upload a document first."
    try:
        summary = summarize_document(fname)
        return f"## 📋 Contract Summary: `{fname}`\n\n{summary}"
    except Exception as e:
        logger.error("Summary error: %s", e)
        return f"❌ Summarization failed: {e}"


# ── Build UI ───────────────────────────────────────────────────────────────

CSS = """
.gradio-container { max-width: 960px !important; margin: auto; }
#chatbot { height: 500px; overflow-y: auto; }
footer { display: none !important; }
"""

DESCRIPTION = """
# 📜 Smart Contract Q&A Assistant

Upload a PDF or DOCX contract, then ask questions about it — grounded answers with source citations.
"""

with gr.Blocks(theme=gr.themes.Soft(), css=CSS, title="Smart Contract Assistant") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Tabs():
        # ── Upload Tab ─────────────────────────────────────────────────────
        with gr.Tab("📤 Upload & Ingest"):
            gr.Markdown("### Step 1: Upload your contract")
            file_input = gr.File(
                label="Drop a PDF or DOCX file here",
                file_types=[".pdf", ".docx", ".doc"],
            )
            upload_btn = gr.Button("Ingest Document", variant="primary")
            upload_status = gr.Markdown("*No document ingested yet.*")

            upload_btn.click(
                fn=handle_upload,
                inputs=[file_input],
                outputs=[upload_status],
            )
            file_input.upload(
                fn=handle_upload,
                inputs=[file_input],
                outputs=[upload_status],
            )

        # ── Chat Tab ───────────────────────────────────────────────────────
        with gr.Tab("💬 Chat"):
            gr.Markdown("### Step 2: Ask questions about your contract")
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                label="Contract Assistant",
                bubble_full_width=False,
            )
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="e.g. What are the termination conditions?",
                    label="Your question",
                    lines=2,
                    scale=5,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            clear_btn = gr.Button("🗑 Clear conversation", variant="secondary")

            send_btn.click(
                fn=chat,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot],
            )
            msg_input.submit(
                fn=chat,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot],
            )
            clear_btn.click(fn=clear_chat, outputs=[msg_input, chatbot])

        # ── Summary Tab ────────────────────────────────────────────────────
        with gr.Tab("📋 Summary"):
            gr.Markdown("### One-click contract summary")
            gr.Markdown(
                "Generates a structured overview of key parties, obligations, dates, and risks."
            )
            summarize_btn = gr.Button("Generate Summary", variant="primary")
            summary_output = gr.Markdown("*Click the button above to generate a summary.*")

            summarize_btn.click(
                fn=handle_summarize,
                inputs=[],
                outputs=[summary_output],
            )

    gr.Markdown(
        "---\n*⚠️ This tool is for informational purposes only and does not constitute legal advice.*"
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
