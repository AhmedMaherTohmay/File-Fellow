from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config.settings import UPLOAD_DIR
from src.core.utils import sanitize_filename, file_content_hash

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc"}
MAX_FILE_MB = 50


@dataclass
class PreparedFile:
    dest: Path
    safe_name: str
    content_hash: str
    is_duplicate: bool
    duplicate_of: Optional[str]


def _validate_extension(filename: str) -> None:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )


def _validate_size(file_path: Path) -> None:
    size_mb = file_path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        raise ValueError(
            f"File is {size_mb:.1f} MB — exceeds the {MAX_FILE_MB} MB limit."
        )


def _check_duplicate(content_hash: str, registry: dict) -> Optional[str]:
    for existing_key, meta in registry.items():
        if meta.get("content_hash") == content_hash:
            return meta.get("doc_name", existing_key.split(":", 1)[-1])
    return None


def _resolve_name(safe_name: str, content_hash: str, registry: dict, user_id: str) -> str:
    registry_key = f"{user_id}:{safe_name}"
    if registry_key not in registry:
        return safe_name
    if registry[registry_key].get("content_hash") == content_hash:
        return safe_name
    stem = Path(safe_name).stem
    ext = Path(safe_name).suffix
    resolved = f"{stem}_{content_hash[:8]}{ext}"
    logger.info(
        "Name collision resolved for user '%s': '%s' already exists with different content — using '%s'.",
        user_id, safe_name, resolved,
    )
    return resolved


def prepare_upload(file_path: Path, registry: dict, user_id: str = "default") -> PreparedFile:
    """
    Full file preparation pipeline before parsing begins.

    Steps:
      1. Validate extension
      2. Validate file size
      3. Sanitize filename
      4. Compute content hash
      5. Check for exact duplicate (same bytes already in store)
      6. Resolve name collision (same name, different content)
      7. Copy to UPLOAD_DIR under the resolved safe name

    Returns PreparedFile. If is_duplicate is True, pipeline.py should return
    early — no parsing or embedding is needed.
    """
    _validate_extension(file_path.name)
    _validate_size(file_path)

    safe_name = sanitize_filename(file_path.name)
    content_hash = file_content_hash(file_path)

    existing = _check_duplicate(content_hash, registry)
    if existing:
        logger.info(
            "Duplicate detected: '%s' has the same content as '%s' — skipping.",
            file_path.name, existing,
        )
        return PreparedFile(
            dest=file_path,
            safe_name=existing,
            content_hash=content_hash,
            is_duplicate=True,
            duplicate_of=existing,
        )

    safe_name = _resolve_name(safe_name, content_hash, registry, user_id)

    dest = UPLOAD_DIR / safe_name
    if file_path.resolve() != dest.resolve():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, dest)
        logger.debug("Copied '%s' → '%s'.", file_path, dest)

        if file_path.parent.resolve() == UPLOAD_DIR.resolve():
            try:
                file_path.unlink()
            except OSError as exc:
                logger.warning("Could not remove original '%s': %s", file_path, exc)

    return PreparedFile(
        dest=dest,
        safe_name=safe_name,
        content_hash=content_hash,
        is_duplicate=False,
        duplicate_of=None,
    )
