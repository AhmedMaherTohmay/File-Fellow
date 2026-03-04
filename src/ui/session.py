from __future__ import annotations

import uuid


def new_user_id() -> str:
    return str(uuid.uuid4())[:12]


def new_conversation_id() -> str:
    return str(uuid.uuid4())


def connect_user(typed_id: str) -> tuple:
    user_id = typed_id.strip() if typed_id.strip() else new_user_id()
    conversation_id = new_conversation_id()

    if typed_id.strip():
        html = (
            f'<div class="session-pill pill-connected">'
            f'<span class="pill-dot"></span>'
            f'Reconnected · <code>{user_id}</code>'
            f'</div>'
        )
    else:
        html = (
            f'<div class="session-pill pill-new">'
            f'<span class="pill-dot"></span>'
            f'New session · <code>{user_id}</code>'
            f'<span class="pill-hint"> — save this ID to continue later</span>'
            f'</div>'
        )
    return user_id, conversation_id, html, ""


def new_conversation(user_id: str) -> tuple:
    conv_id = new_conversation_id()
    html = (
        f'<div class="session-pill pill-new">'
        f'<span class="pill-dot"></span>'
        f'New conversation · <code>{user_id or "—"}</code>'
        f'</div>'
    )
    return "", [], html, conv_id
