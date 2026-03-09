from __future__ import annotations

import logging

import gradio as gr

from config.settings import GRADIO_HOST, GRADIO_PORT, GRADIO_SHARE
from src.ui.styles import CSS, HEADER_HTML
from src.ui.session import new_conversation_id
from src.ui.tabs.upload import build_upload_tab
from src.ui.tabs.chat import build_chat_tab
from src.ui.tabs.summary import build_summary_tab

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.orange,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Sora"),
    ),
    css=CSS,
    title="Document Q&A Assistant",
) as demo:
    user_id_state      = gr.State("")
    conversation_state = gr.State(new_conversation_id)

    gr.HTML(HEADER_HTML)

    with gr.Tabs():
        build_upload_tab(user_id_state)
        build_chat_tab(user_id_state, conversation_state)
        build_summary_tab(user_id_state)


def launch():
    demo.launch(
        server_name=GRADIO_HOST,
        server_port=GRADIO_PORT,
        share=GRADIO_SHARE,
        show_error=True,
    )


if __name__ == "__main__":
    launch()
