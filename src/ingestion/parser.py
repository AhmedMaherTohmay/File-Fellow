"""
Document parsing utilities.

Supports:
- PDF (PyMuPDF -> pdfplumber fallback)
- DOCX (paragraphs + tables)

Returns a list of dicts:
{
    "text": str,
    "page": int,
    "source": str,
    "file_type": str,
    "file_path": str,
}
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


# =========================
# Common helpers
# =========================

def _validate_file(file_path: Path, max_mb: int = 50) -> None:
    """
    Validate that a file exists, is a file, and is not too large (> max_mb MB).

    Args:
        file_path: Path to the file to validate.
        max_mb: Maximum file size in MB (default: 50).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a file or is too large.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"Not a file: {file_path}")

    size_mb = file_path.stat().st_size / (1024 * 1024)
    if size_mb > max_mb:
        raise ValueError(f"File too large ({size_mb:.2f} MB). Limit = {max_mb} MB")


def _make_page(
    *,
    text: str,
    page: int,
    file_path: Path,
) -> Dict[str, Any]:
    return {
        "text": text.strip(),
        "page": page,
        "source": file_path.name,
        "file_type": file_path.suffix.lower().lstrip("."),
        "file_path": str(file_path.resolve()),
    }


# =========================
# PDF parsing
# =========================

def parse_pdf(file_path: str | Path) -> List[Dict[str, Any]]:
    """
    Parse a PDF file and return a list of page dictionaries.

    Tries PyMuPDF first, then falls back to pdfplumber if PyMuPDF fails.

    Args:
        file_path: Path to the PDF file to parse.

    Returns:
        List of page dictionaries with keys: text, page, source, file_type, file_path.

    Raises:
        RuntimeError: If both PyMuPDF and pdfplumber fail to parse the PDF.
    """
    file_path = Path(file_path)
    _validate_file(file_path)

    pages: List[Dict[str, Any]] = []

    # ---- Try PyMuPDF first ----
    try:
        import fitz  # PyMuPDF

        # loop through pages extract text add it to a list
        with fitz.open(file_path) as doc:
            for idx, page in enumerate(doc, start=1):
                text = page.get_text("text")
                if text and text.strip():
                    pages.append(
                        _make_page(
                            text=text,
                            page=idx,
                            file_path=file_path,
                        )
                    )

        if pages:
            logger.info(
                "Parsed %d pages from '%s' using PyMuPDF",
                len(pages),
                file_path.name,
            )
            return pages

        logger.warning("PyMuPDF returned empty text, falling back to pdfplumber")

    except Exception as e:
        logger.warning("PyMuPDF failed (%s), falling back to pdfplumber", e)

    # ---- Fallback: pdfplumber ----
    try:
        import pdfplumber

        with pdfplumber.open(file_path) as pdf:
            for idx, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(
                        _make_page(
                            text=text,
                            page=idx,
                            file_path=file_path,
                        )
                    )

        logger.info(
            "Parsed %d pages from '%s' using pdfplumber",
            len(pages),
            file_path.name,
        )
        return pages

    except Exception as e:
        logger.error("Failed to parse PDF '%s': %s", file_path.name, e)
        raise RuntimeError(f"Cannot parse PDF '{file_path.name}'") from e


# =========================
# DOCX parsing
# =========================

def parse_docx(file_path: str | Path) -> List[Dict[str, Any]]:
    from docx import Document

    file_path = Path(file_path)
    _validate_file(file_path)

    doc = Document(file_path)
    blocks: List[str] = []

    # ---- Paragraphs ----
    for p in doc.paragraphs:
        if p.text.strip():
            blocks.append(p.text.strip())

    # ---- Tables ----
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(
                cell.text.strip() for cell in row.cells if cell.text.strip()
            )
            if row_text:
                blocks.append(row_text)

    # ---- Group into synthetic pages ----
    GROUP_SIZE = 30
    pages: List[Dict[str, Any]] = []

    for idx in range(0, len(blocks), GROUP_SIZE):
        chunk = "\n".join(blocks[idx : idx + GROUP_SIZE])
        pages.append(
            _make_page(
                text=chunk,
                page=(idx // GROUP_SIZE) + 1,
                file_path=file_path,
            )
        )

    logger.info(
        "Parsed %d page-groups from '%s'",
        len(pages),
        file_path.name,
    )
    return pages


# =========================
# Auto-dispatch
# =========================

def parse_document(file_path: str | Path) -> List[Dict[str, Any]]:
    file_path = Path(file_path)
    ext = file_path.suffix.lower()

    if ext == ".pdf":
        return parse_pdf(file_path)

    if ext in {".docx", ".doc"}:
        return parse_docx(file_path)

    raise ValueError(
        f"Unsupported file type '{ext}'. Supported: PDF, DOCX"
    )


if __name__ == "__main__":
    import os
    from pathlib import Path

    # get the parent of the parent of the currnet dir
    parent_dir = Path(__file__).parent.parent.parent
    os.chdir(parent_dir)
    parse_pdf("uploads/Syllabus-CSE325-Spring2021.pdf")