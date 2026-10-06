"""Format-independent document ingestion; no RAG framework dependency."""
from .normalization.schema import CanonicalDocument, Element, Page
from .pipeline import DocumentPipeline

__all__ = ["CanonicalDocument", "Element", "Page", "DocumentPipeline"]
