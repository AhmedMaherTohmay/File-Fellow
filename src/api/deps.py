from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import Depends, HTTPException, UploadFile

from config.settings import settings
from src.core.utils import sanitize_filename
from src.storage.document_store import store_is_ready

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc"}


async def require_store() -> None:
    if not store_is_ready():
        raise HTTPException(status_code=400, detail="No documents ingested yet.")


async def save_upload(file: UploadFile) -> Path:
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    safe_name = sanitize_filename(file.filename)
    dest = settings.UPLOAD_DIR / safe_name

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not save file: {exc}") from exc

    return dest
