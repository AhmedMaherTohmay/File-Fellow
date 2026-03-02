"""
src/ingestion — document ingestion sub-package.
"""
# Re-export so all existing callers keep working without any changes.
from src.ingestion.pipeline import ingest_document
from src.core.utils import sanitize_filename

__all__ = ["ingest_document", "sanitize_filename"]