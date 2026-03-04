from __future__ import annotations

import logging

import gradio as gr

from src.services.qa import answer_question
from src.ui.formatters import get_doc_choices, sources_html
from src.ui.session import connect_user, new_conversation

logger = logging.getLogger(__name__)


def chat(user_message, chat_history, doc_selector, user_id, conversation_id):
    if not user_message.strip():
        return "", chat_history, ""

    doc_name = None if doc_selector == "All Documents" else doc_selector

    try:
        result = answer_question(
            question=user_message,
            history=chat_history.copy(),
            doc_name=doc_name,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        bot_reply    = result["answer"]
        sources_text = sources_html(result["sources"])
    except Exception as e:
        logger.error("Chat error: %s", e)
        bot_reply    = f"Error: {e}"
        sources_text = ""

    chat_history.append({"role": "user",      "content": user_message})
    chat_history.append({"role": "assistant",  "content": bot_reply})
    return "", chat_history, sources_text


def _auto_connect_and_chat(msg, history, doc_sel, uid, conv_id):
    if not uid:
        uid, conv_id, _, _ = connect_user("")
    return chat(msg, history, doc_sel, uid, conv_id)


def build_chat_tab(user_id_state, conversation_state):
    with gr.Tab("Chat"):
        with gr.Group(elem_id="session-connect-box"):
            gr.HTML('<div class="connect-label">Session</div>')
            with gr.Row():
                user_id_input = gr.Textbox(
                    placeholder="User ID — leave blank for a new one",
                    label="", scale=6, container=False, elem_id="uid-input",
                )
                connect_btn = gr.Button("Connect", variant="primary", scale=1)
            session_info = gr.HTML(
                '<div class="session-pill">'
                '<span class="pill-dot"></span>'
                'Not connected — click Connect or just send a message'
                '</div>'
            )

        with gr.Row():
            doc_selector = gr.Dropdown(
                choices=get_doc_choices(), value="All Documents",
                label="Scope", scale=4, interactive=True,
            )
            refresh_docs_btn = gr.Button("↺", variant="secondary", scale=0)

        chatbot = gr.Chatbot(
            elem_id="chatbot", label="File Fellow")
        sources_display = gr.HTML(elem_id="sources-panel", value="")

        with gr.Row():
            msg_input = gr.Textbox(
                placeholder="Ask a question about your documents…",
                label="", lines=2, scale=6, container=False, elem_id="msg-input",
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)

        new_conv_btn = gr.Button("New Conversation", variant="secondary")

        connect_btn.click(
            fn=connect_user, inputs=[user_id_input],
            outputs=[user_id_state, conversation_state, session_info, user_id_input],
        )
        send_btn.click(
            fn=_auto_connect_and_chat,
            inputs=[msg_input, chatbot, doc_selector, user_id_state, conversation_state],
            outputs=[msg_input, chatbot, sources_display],
        )
        msg_input.submit(
            fn=_auto_connect_and_chat,
            inputs=[msg_input, chatbot, doc_selector, user_id_state, conversation_state],
            outputs=[msg_input, chatbot, sources_display],
        )
        new_conv_btn.click(
            fn=new_conversation, inputs=[user_id_state],
            outputs=[msg_input, chatbot, session_info, conversation_state],
        )
        new_conv_btn.click(fn=lambda: "", outputs=[sources_display])
        refresh_docs_btn.click(
            fn=lambda: gr.Dropdown(choices=get_doc_choices()),
            outputs=[doc_selector],
        )
