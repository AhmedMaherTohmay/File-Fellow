from __future__ import annotations

import html
import logging

import gradio as gr

from src.storage.document_store import get_ingested_documents
from src.services.summary import summarize_document
from src.ui.formatters import get_doc_choices, status_html

logger = logging.getLogger(__name__)


def handle_summarize(doc_selector: str, user_id: str) -> str:
    uid = user_id or "default"
    docs = get_ingested_documents(user_id=uid)
    if not docs:
        return status_html("warning", "No documents ingested — upload one first.")

    fname = docs[0] if doc_selector == "All Documents" else doc_selector
    try:
        summary = summarize_document(fname, user_id=uid)
        safe_fname   = html.escape(fname)
        safe_summary = html.escape(summary)
        return (
            f'<div class="summary-wrap">'
            f'<div class="summary-title">{safe_fname}</div>'
            f'<div class="summary-body">{safe_summary}</div>'
            f'</div>'
        )
    except Exception as e:
        logger.error("Summary error: %s", e)
        return status_html("error", f"Summarization failed: {e}")


def build_summary_tab(user_id_state):
    with gr.Tab("Summary"):
        gr.HTML('<div class="section-title">Generate Document Summary</div>')
        with gr.Row():
            doc_selector = gr.Dropdown(
                choices=get_doc_choices(), value="All Documents",
                label="Document", scale=5, interactive=True,
            )
            refresh_btn   = gr.Button("↺", variant="secondary", scale=0)
            summarize_btn = gr.Button("Generate Summary", variant="primary", scale=2)

        summary_output = gr.HTML(
            '<div class="doc-empty">Select a document and click Generate Summary.</div>'
        )

        summarize_btn.click(
            fn=handle_summarize,
            inputs=[doc_selector, user_id_state],
            outputs=[summary_output],
        )
        refresh_btn.click(
            fn=lambda uid: gr.Dropdown(choices=get_doc_choices(user_id=uid or None)),
            inputs=[user_id_state],
            outputs=[doc_selector],
        )
