from __future__ import annotations

import html
import logging
from pathlib import Path
from typing import List, Optional

from src.storage.document_store import get_ingested_documents, get_document_registry

logger = logging.getLogger(__name__)


def get_doc_choices(user_id: Optional[str] = None) -> List[str]:
    return ["All Documents"] + get_ingested_documents(user_id=user_id)


def status_html(kind: str, text: str) -> str:
    icon = {"success": "✓", "warning": "⚠", "error": "✕"}.get(kind, "·")
    safe_text = html.escape(text)
    return (
        f'<div class="status-block">'
        f'<div class="msg-row msg-{kind}">'
        f'<span class="msg-icon">{icon}</span><span>{safe_text}</span>'
        f'</div></div>'
    )


def doc_list_html(user_id: Optional[str] = None) -> str:
    registry = get_document_registry(user_id=user_id)
    if not registry:
        return '<div class="doc-empty">No documents ingested yet.</div>'

    cards = ""
    for _key, meta in registry.items():
        name   = html.escape(meta.get("doc_name", _key.split(":", 1)[-1]))
        pages  = meta.get("num_pages", "?")
        chunks = meta.get("num_chunks", "?")
        ftype  = name.rsplit(".", 1)[-1].upper() if "." in name else "DOC"
        cards += (
            f'<div class="doc-card">'
            f'<span class="doc-badge">{ftype}</span>'
            f'<span class="doc-name">{name}</span>'
            f'<span class="doc-meta">{pages} pages · {chunks} chunks</span>'
            f'</div>'
        )
    return f'<div class="doc-grid">{cards}</div>'


def sources_html(sources: list) -> str:
    if not sources:
        return ""
    cards = ""
    for i, s in enumerate(sources, 1):
        score_pct = int(s["score"] * 100)
        bar_color = (
            "#d4a017" if score_pct >= 60
            else "#6b8aad" if score_pct >= 40
            else "#4a5a6e"
        )
        safe_source  = html.escape(str(s.get("source", "?")))
        safe_snippet = html.escape(s.get("snippet", "")[:160].strip())
        cards += (
            f'<div class="src-card">'
            f'<div class="src-header">'
            f'<span class="src-num">[{i}]</span>'
            f'<span class="src-file">{safe_source}</span>'
            f'<span class="src-page">p. {s.get("page", "?")}</span>'
            f'<div class="src-score-wrap">'
            f'<div class="src-score-bar" style="width:{score_pct}%;background:{bar_color}"></div>'
            f'</div>'
            f'<span class="src-score-num">{score_pct}%</span>'
            f'</div>'
            f'<div class="src-snippet">{safe_snippet}…</div>'
            f'</div>'
        )
    return f'<div class="sources-wrap"><div class="sources-label">Sources</div>{cards}</div>'


def upload_status_html(file_objs, user_id: str = "default") -> str:
    from src.ingestion.pipeline import ingest_document

    if not file_objs:
        return status_html("warning", "No files selected.")

    if not isinstance(file_objs, list):
        file_objs = [file_objs]

    messages = []
    for file_obj in file_objs:
        if file_obj is None:
            continue
        file_path = Path(file_obj.name)
        try:
            result = ingest_document(file_path, user_id=user_id)
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
        return status_html("warning", "No valid files processed.")

    rows = "".join(
        f'<div class="msg-row msg-{kind}">'
        f'<span class="msg-icon">{"✓" if kind=="success" else "⚠" if kind=="warning" else "✕"}</span>'
        f'<span>{html.escape(text)}</span></div>'
        for kind, text in messages
    )
    return f'<div class="status-block">{rows}</div>'
