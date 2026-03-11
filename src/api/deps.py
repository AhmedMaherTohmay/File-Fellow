from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import HTTPException, UploadFile

from config.settings import settings
from src.core.exceptions import UnsupportedFileType
from src.core.utils import sanitize_filename
from src.ingestion.validators import ALLOWED_EXTENSIONS 
from src.storage.document_store import store_is_ready


async def require_store() -> None:
    if not store_is_ready():
        raise HTTPException(status_code=400, detail="No documents ingested yet.")


async def save_upload(file: UploadFile) -> Path:
    """
    Save an uploaded file to UPLOAD_DIR and return its destination path.

    Raises HTTPException 422 for unsupported file types, 500 for I/O errors.
    """
    try:
        _validate_upload_extension(file.filename)
    except UnsupportedFileType as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    safe_name = sanitize_filename(file.filename)
    dest = settings.UPLOAD_DIR / safe_name

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not save file: {exc}") from exc

    return dest


def _validate_upload_extension(filename: str) -> None:
    """Raise UnsupportedFileType if the extension is not allowed."""
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise UnsupportedFileType(suffix)
