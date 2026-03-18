"""
General-purpose utilities shared across the application.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path


def sanitize_filename(filename: str, max_stem_len: int = 80) -> str:
    """
    Return a filesystem-safe version of *filename*.
    Args:
        filename:     Original filename (may include directory components).
        max_stem_len: Maximum characters kept from the stem (default: 80).

    Returns:
        A sanitised filename string, e.g. ``"Annual_Report_2024.pdf"``.
    """
    # Step 1 — strip directory components
    bare = Path(filename).name

    # Step 2 — split stem / extension; normalise extension case
    stem = Path(bare).stem
    suffix = Path(bare).suffix.lower()          # ".PDF" → ".pdf"

    # Step 3 — replace unsafe chars, collapse underscores, strip edges
    safe_stem = re.sub(r"[^\w\-]", "_", stem)   # \w = [a-zA-Z0-9_]
    safe_stem = re.sub(r"_+", "_", safe_stem).strip("_")

    # Steps 4–5 — length cap and empty-stem fallback
    safe_stem = (safe_stem[:max_stem_len] or "unnamed")

    return f"{safe_stem}{suffix}"


def file_content_hash(file_path: "str | Path") -> str:
    """
    Compute the SHA-256 digest of a file's content.
    """
    sha = hashlib.sha256()
    with open(file_path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            sha.update(block)
    return sha.hexdigest()


def normalise_score(raw: float) -> float:
    """
    Map similarity score to the [0, 1] range.
    """
    return max(0.0, min(1.0, (raw + 1.0) / 2.0))
