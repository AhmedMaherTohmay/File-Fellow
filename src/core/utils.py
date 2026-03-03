"""
General-purpose utilities shared across the application.
"""
from __future__ import annotations

import re
from pathlib import Path
import hashlib


def sanitize_filename(filename: str, max_len: int = 40) -> str:
    """
    Return a filesystem-safe version of *filename*.
    """
    # Step 1 — strip directory components
    bare = Path(filename).name
    # Create hash from full filename for uniqueness
    name_hash = hashlib.md5(bare.encode()).hexdigest()[:8]

    # Step 2 — split stem / extension; normalise extension case
    stem = Path(bare).stem
    suffix = Path(bare).suffix.lower()          # ".PDF" → ".pdf"

    # Step 3–4 — replace unsafe chars, collapse underscores, strip edges
    safe_stem = re.sub(r"[^\w\-]", "_", stem)   # \w = [a-zA-Z0-9_]
    safe_stem = re.sub(r"_+", "_", safe_stem).strip("_")

    # Step 5-6 — truncate and append hash and extension
    # max_len = 40 - len(name_hash) - 1
    available = max_len - len(name_hash) - 1
    truncated = safe_stem[:available] if len(safe_stem) > available else safe_stem

    return f"{truncated}_{name_hash}{suffix}"

if __name__ == "__main__":
    print(sanitize_filename("test.pdf"))