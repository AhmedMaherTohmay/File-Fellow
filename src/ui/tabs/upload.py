from __future__ import annotations

import gradio as gr

from src.storage.document_store import remove_document
from src.ui.formatters import doc_list_html, status_html, upload_status_html


def handle_upload(file_objs, user_id: str):
    uid = user_id or "default"
    return upload_status_html(file_objs, user_id=uid), doc_list_html(user_id=uid)


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
        doc_table   = gr.HTML(doc_list_html())
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