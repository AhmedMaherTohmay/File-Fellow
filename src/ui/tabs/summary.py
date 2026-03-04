from __future__ import annotations

import logging

import gradio as gr

from src.storage.document_store import get_ingested_documents
from src.services.summary import summarize_document
from src.ui.formatters import get_doc_choices, status_html

logger = logging.getLogger(__name__)


def handle_summarize(doc_selector: str) -> str:
    docs = get_ingested_documents()
    if not docs:
        return status_html("warning", "No documents ingested — upload one first.")

    fname = docs[0] if doc_selector == "All Documents" else doc_selector
    try:
        summary = summarize_document(fname)
        return (
            f'<div class="summary-wrap">'
            f'<div class="summary-title">{fname}</div>'
            f'<div class="summary-body">{summary}</div>'
            f'</div>'
        )
    except Exception as e:
        logger.error("Summary error: %s", e)
        return status_html("error", f"Summarization failed: {e}")


def build_summary_tab():
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

        summarize_btn.click(fn=handle_summarize, inputs=[doc_selector], outputs=[summary_output])
        refresh_btn.click(
            fn=lambda: gr.Dropdown(choices=get_doc_choices()),
            outputs=[doc_selector],
        )
