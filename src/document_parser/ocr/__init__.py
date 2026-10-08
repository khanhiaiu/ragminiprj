"""Composable PP-StructureV3 OCR architecture."""

from .detection import OCRDetector
from .document import DocumentParser, recover_unassigned_ocr_lines
from .layout import LayoutDetector, require_requested_gpu
from .recognition import OCRRecognizer
from .table import TableRecognizer

__all__ = [
    "DocumentParser",
    "LayoutDetector",
    "OCRDetector",
    "OCRRecognizer",
    "TableRecognizer",
    "recover_unassigned_ocr_lines",
    "require_requested_gpu",
]
