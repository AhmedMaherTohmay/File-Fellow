from __future__ import annotations

from pathlib import Path
from typing import List

from src.storage.document_store import get_ingested_documents, get_document_registry


def get_doc_choices() -> List[str]:
    return ["All Documents"] + get_ingested_documents()


def status_html(kind: str, text: str) -> str:
    icon = {"success": "✓", "warning": "⚠", "error": "✕"}.get(kind, "·")
    return (
        f'<div class="status-block">'
        f'<div class="msg-row msg-{kind}">'
        f'<span class="msg-icon">{icon}</span><span>{text}</span>'
        f'</div></div>'
    )


def doc_list_html() -> str:
    registry = get_document_registry()
    if not registry:
        return '<div class="doc-empty">No documents ingested yet.</div>'

    cards = ""
    for name, meta in registry.items():
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
        bar_color = "#d4a017" if score_pct >= 60 else "#6b8aad" if score_pct >= 40 else "#4a5a6e"
        cards += (
            f'<div class="src-card">'
            f'<div class="src-header">'
            f'<span class="src-num">[{i}]</span>'
            f'<span class="src-file">{s["source"]}</span>'
            f'<span class="src-page">p. {s["page"]}</span>'
            f'<div class="src-score-wrap"><div class="src-score-bar" style="width:{score_pct}%;background:{bar_color}"></div></div>'
            f'<span class="src-score-num">{score_pct}%</span>'
            f'</div>'
            f'<div class="src-snippet">{s["snippet"][:160].strip()}…</div>'
            f'</div>'
        )
    return f'<div class="sources-wrap"><div class="sources-label">Sources</div>{cards}</div>'


def upload_status_html(file_objs) -> str:
    from src.ingestion.pipeline import ingest_document
    import logging
    logger = logging.getLogger(__name__)

    if not file_objs:
        return status_html("warning", "No files selected.")

    if not isinstance(file_objs, list):
        file_objs = [file_objs]

    messages = []
    for file_obj in file_objs:
        if file_obj is None:
            continue
        try:
            result = ingest_document(file_obj.name)
            if result.get("duplicate"):
                messages.append(("warning",
                    f"{Path(file_obj.name).name} — duplicate of '{result['duplicate_of']}', skipped."))
            else:
                messages.append(("success",
                    f"{result['filename']} — {result['num_pages']} pages · {result['num_chunks']} chunks"))
        except Exception as e:
            logger.error("Upload error for %s: %s", file_obj.name, e)
            messages.append(("error", f"{Path(file_obj.name).name} — {e}"))

    if not messages:
        return status_html("warning", "No valid files processed.")

    rows = "".join(
        f'<div class="msg-row msg-{kind}">'
        f'<span class="msg-icon">{"✓" if kind=="success" else "⚠" if kind=="warning" else "✕"}</span>'
        f'<span>{text}</span></div>'
        for kind, text in messages
    )
    return f'<div class="status-block">{rows}</div>'
