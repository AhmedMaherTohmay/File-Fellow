"""
General-purpose utilities shared across the application.
"""
from __future__ import annotations

import re
from pathlib import Path


def sanitize_filename(filename: str) -> str:
    """
    Return a filesystem-safe version of *filename*.
    """
    # Step 1 — strip directory components
    bare = Path(filename).name

    # Step 2 — split stem / extension; normalise extension case
    stem = Path(bare).stem
    suffix = Path(bare).suffix.lower()          # ".PDF" → ".pdf"

    # Step 3–4 — replace unsafe chars, collapse underscores, strip edges
    safe_stem = re.sub(r"[^\w\-]", "_", stem)   # \w = [a-zA-Z0-9_]
    safe_stem = re.sub(r"_+", "_", safe_stem).strip("_")

    # Steps 5–7 — length cap and empty-stem fallback
    safe_stem = safe_stem[:100] or "unnamed"

    return f"{safe_stem}{suffix}"