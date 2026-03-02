"""
Custom exception hierarchy for File Fellow.

Having typed exceptions lets callers (API, UI) pattern-match on the error kind
rather than inspecting string messages, which is fragile and hard to maintain.

Hierarchy
---------
FileFellowError          — base for every application-level error
  IngestionError         — something went wrong during document ingestion
    UnsupportedFileType  — file extension not in the allowed set
    FileTooLarge         — file exceeds the configured size limit
    ExtractionError      — parser returned no usable text
  RetrievalError         — vector-store or similarity-search failure
  LLMError               — LLM initialisation or invocation failure
  SessionError           — history / session management failure
"""
from __future__ import annotations


class FileFellowError(Exception):
    """Base class for all application-level errors."""


# ── Ingestion ──────────────────────────────────────────────────────────────

class IngestionError(FileFellowError):
    """Raised when document ingestion fails for any reason."""


class UnsupportedFileType(IngestionError):
    """Raised when the uploaded file has an extension we cannot process."""

    def __init__(self, ext: str) -> None:
        super().__init__(
            f"Unsupported file type '{ext}'. Supported formats: PDF, DOCX."
        )
        self.ext = ext


class FileTooLarge(IngestionError):
    """Raised when the uploaded file exceeds the configured size limit."""

    def __init__(self, size_mb: float, limit_mb: int) -> None:
        super().__init__(
            f"File is {size_mb:.1f} MB — exceeds the {limit_mb} MB limit."
        )
        self.size_mb = size_mb
        self.limit_mb = limit_mb


class ExtractionError(IngestionError):
    """Raised when the parser returns no usable text from a document."""

    def __init__(self, filename: str) -> None:
        super().__init__(
            f"No text could be extracted from '{filename}'. "
            "The file may be empty, image-only, or password-protected."
        )
        self.filename = filename


# ── Retrieval ──────────────────────────────────────────────────────────────

class RetrievalError(FileFellowError):
    """Raised when the vector store is unavailable or the search fails."""


# ── LLM ───────────────────────────────────────────────────────────────────

class LLMError(FileFellowError):
    """Raised when the LLM cannot be initialised or a call fails."""


# ── Session / memory ──────────────────────────────────────────────────────

class SessionError(FileFellowError):
    """Raised when session history persistence or retrieval fails."""