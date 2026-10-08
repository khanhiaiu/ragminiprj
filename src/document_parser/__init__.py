"""Format-independent document ingestion; no RAG framework dependency."""

from .config import ParserConfig, PDFAnalyzerConfig, ProjectConfig
from .normalization.schema import CanonicalDocument, Element, Page
from .ocr import DocumentParser, LayoutDetector, OCRDetector, OCRRecognizer, TableRecognizer
from .pipeline import DocumentPipeline
from .retrieval.models import RetrievalDocument, RetrievalElement
from .retrieval.preprocessor import RetrievalPreprocessor

__all__ = [
    "CanonicalDocument",
    "Element",
    "Page",
    "DocumentPipeline",
    "DocumentParser",
    "LayoutDetector",
    "OCRDetector",
    "OCRRecognizer",
    "TableRecognizer",
    "ParserConfig",
    "PDFAnalyzerConfig",
    "ProjectConfig",
    "RetrievalDocument",
    "RetrievalElement",
    "RetrievalPreprocessor",
]
