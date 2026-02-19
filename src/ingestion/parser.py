"""
Document parsing utilities.
Supports PDF (via PyMuPDF / pdfplumber) and DOCX (via python-docx).
Returns a list of dicts: {text, page, source}.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def parse_pdf(file_path: str | Path) -> List[Dict[str, Any]]:
    """Extract text from a PDF file, page by page.

    Tries PyMuPDF first; falls back to pdfplumber.

    Args:
        file_path: Absolute path to the PDF.

    Returns:
        List of dicts with keys ``text``, ``page``, ``source``.
    """
    file_path = Path(file_path)
    pages: List[Dict[str, Any]] = []

    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(file_path))
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append({"text": text, "page": i, "source": file_path.name})
        doc.close()
        logger.info("PyMuPDF parsed %d pages from '%s'.", len(pages), file_path.name)
        return pages
    except Exception as e:  # noqa: BLE001
        logger.warning("PyMuPDF failed (%s); falling back to pdfplumber.", e)

    try:
        import pdfplumber

        with pdfplumber.open(str(file_path)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    pages.append({"text": text, "page": i, "source": file_path.name})
        logger.info("pdfplumber parsed %d pages from '%s'.", len(pages), file_path.name)
        return pages
    except Exception as e:
        logger.error("pdfplumber also failed: %s", e)
        raise RuntimeError(f"Cannot parse PDF '{file_path.name}': {e}") from e


def parse_docx(file_path: str | Path) -> List[Dict[str, Any]]:
    """Extract text from a DOCX file paragraph by paragraph.

    Groups paragraphs into synthetic 'pages' of ~40 lines each.

    Args:
        file_path: Absolute path to the DOCX.

    Returns:
        List of dicts with keys ``text``, ``page``, ``source``.
    """
    from docx import Document  # python-docx

    file_path = Path(file_path)
    doc = Document(str(file_path))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

    # Group ~40 paragraphs per synthetic page
    GROUP_SIZE = 40
    pages: List[Dict[str, Any]] = []
    for idx in range(0, len(paragraphs), GROUP_SIZE):
        chunk_text = "\n".join(paragraphs[idx : idx + GROUP_SIZE])
        pages.append(
            {
                "text": chunk_text,
                "page": (idx // GROUP_SIZE) + 1,
                "source": file_path.name,
            }
        )

    logger.info("python-docx parsed %d page-groups from '%s'.", len(pages), file_path.name)
    return pages


def parse_document(file_path: str | Path) -> List[Dict[str, Any]]:
    """Auto-detect file type and parse.

    Args:
        file_path: Path to PDF or DOCX file.

    Returns:
        List of page dicts.

    Raises:
        ValueError: If the file extension is not supported.
    """
    file_path = Path(file_path)
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        return parse_pdf(file_path)
    elif ext in {".docx", ".doc"}:
        return parse_docx(file_path)
    else:
        raise ValueError(f"Unsupported file type: '{ext}'. Only PDF and DOCX are supported.")
