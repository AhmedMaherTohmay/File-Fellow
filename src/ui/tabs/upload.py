from __future__ import annotations

import logging
from pathlib import Path

import gradio as gr

from src.db.repositories.document_repo import remove_document
from src.ingestion.pipeline import ingest_document
from src.ui.formatters import doc_list_html, status_html

logger = logging.getLogger(__name__)


def handle_upload(file_objs, user_id: str):
    """
    Ingest each uploaded file and return status HTML + updated doc list.

    Ingestion logic lives here — not inside a formatter helper. Formatters
    only receive data to render; they don't call the ingestion pipeline.
    """
    uid = user_id or "default"

    if not file_objs:
        return status_html("warning", "No files selected."), doc_list_html(user_id=uid)

    if not isinstance(file_objs, list):
        file_objs = [file_objs]

    import html as html_mod
    messages = []
    for file_obj in file_objs:
        if file_obj is None:
            continue
        file_path = Path(file_obj.name)
        try:
            result = ingest_document(file_path, user_id=uid)
            if result.get("duplicate"):
                messages.append(("warning",
                    f"{file_path.name} — duplicate of '{result['duplicate_of']}', skipped."))
            else:
                messages.append(("success",
                    f"{result['filename']} — "
                    f"{result['num_pages']} pages · {result['num_chunks']} chunks"))
        except Exception as e:
            logger.error("Upload error for %s: %s", file_path.name, e)
            messages.append(("error", f"{file_path.name} — {e}"))

    if not messages:
        return status_html("warning", "No valid files processed."), doc_list_html(user_id=uid)

    rows = "".join(
        f'<div class="msg-row msg-{kind}">'
        f'<span class="msg-icon">{"✓" if kind=="success" else "⚠" if kind=="warning" else "✕"}</span>'
        f'<span>{html_mod.escape(text)}</span></div>'
        for kind, text in messages
    )
    status = f'<div class="status-block">{rows}</div>'
    return status, doc_list_html(user_id=uid)


def handle_remove(doc_name: str, user_id: str):
    uid = user_id or "default"
    if not doc_name.strip():
        return status_html("warning", "Enter a document name first."), doc_list_html(user_id=uid)
    if remove_document(doc_name.strip(), user_id=uid):
        return status_html("success", f"'{doc_name}' removed."), doc_list_html(user_id=uid)
    return status_html("error", f"'{doc_name}' not found."), doc_list_html(user_id=uid)


def build_upload_tab(user_id_state):
    with gr.Tab("Upload & Manage"):
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                file_input = gr.File(
                    label="Drop PDF or DOCX files here",
                    file_types=[".pdf", ".docx", ".doc"],
                    file_count="multiple",
                )
            with gr.Column(scale=3):
                upload_btn    = gr.Button("Ingest Files", variant="primary")
                upload_status = gr.HTML('<div class="doc-empty">No files uploaded yet.</div>')

        gr.HTML('<hr class="section-divider">')
        gr.HTML('<div class="section-title">Ingested Documents</div>')
        # Initial render uses user_id_state value — at page load this is the
        # generated session ID, so it correctly shows only this user's documents.
        doc_table   = gr.HTML(doc_list_html(user_id=None))
        refresh_btn = gr.Button("Refresh", variant="secondary")

        gr.HTML('<hr class="section-divider">')
        gr.HTML('<div class="section-title">Remove Document</div>')
        with gr.Row():
            remove_input = gr.Textbox(
                placeholder="Exact filename, e.g. contract.pdf",
                label="", scale=5, container=False,
            )
            remove_btn = gr.Button("Remove", variant="stop", scale=1)
        remove_status = gr.HTML("")

        upload_btn.click(
            fn=handle_upload,
            inputs=[file_input, user_id_state],
            outputs=[upload_status, doc_table],
        )
        refresh_btn.click(
            fn=lambda uid: doc_list_html(user_id=uid or "default"),
            inputs=[user_id_state],
            outputs=[doc_table],
        )
        remove_btn.click(
            fn=handle_remove,
            inputs=[remove_input, user_id_state],
            outputs=[remove_status, doc_table],
        )
