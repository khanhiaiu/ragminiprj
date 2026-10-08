from ..config import PDFAnalyzerConfig
from ..normalization.schema import CanonicalDocument
from ..router.pdf_analyzer import text_metrics


class ParseQualityValidator:
    def __init__(self, config: PDFAnalyzerConfig | None = None):
        self.config = config or PDFAnalyzerConfig()

    def page_needs_ocr(self, elements: list[dict], analysis) -> bool:
        if analysis.blank:
            return False
        text = "\n".join(e.get("text", "") for e in elements if e["element_type"] != "image")
        printable, invalid, quality = text_metrics(text)
        return (
            not text.strip()
            or printable < self.config.minimum_printable_ratio
            or invalid > self.config.max_invalid_char_ratio
            or quality < self.config.minimum_text_quality
        )

    def validate(self, document: CanonicalDocument) -> dict:
        text = "\n".join(e.text for e in document.elements if e.element_type != "image")
        _, invalid, quality = text_metrics(text)
        failed = [p.page_number for p in document.pages if p.status == "failed"]
        warnings = []
        if not document.elements:
            warnings.append("Document contains no elements")
        if not text.strip():
            warnings.append("No text extracted; inspect image assets or blank pages")
        if invalid > self.config.max_invalid_char_ratio:
            warnings.append("Invalid character ratio exceeds threshold")
        if failed:
            warnings.append(f"Pages failed: {failed}")
        if document.metadata.get("conversion_status") == "partial_success":
            warnings.append("Docling reported partial conversion")
        return {
            "status": "partial" if warnings else "ok",
            "element_count": len(document.elements),
            "text_length": len(text),
            "invalid_char_ratio": invalid,
            "text_quality_score": quality,
            "page_count": len(document.pages),
            "successful_pages": len(document.pages) - len(failed),
            "failed_pages": failed,
            "warnings": warnings,
        }
