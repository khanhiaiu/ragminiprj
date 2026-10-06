import logging
from pathlib import Path

import pymupdf as fitz

from ...config import ParserConfig
from ...normalization.normalizer import Normalizer
from ...normalization.schema import CanonicalDocument
from ...quality.validator import ParseQualityValidator
from ...router.pdf_analyzer import PDFAnalyzer
from ...utils.file_utils import document_data
from ..image.paddle_image_parser import PaddleEngine
from .native_pdf_parser import NativePDFParser
from .paddle_pdf_parser import PaddlePDFParser

logger = logging.getLogger(__name__)


class PDFParser:
    """Page-level routing and quality fallback; partial documents survive errors."""
    def __init__(self, assets_dir: Path, config: ParserConfig | None = None, engine: PaddleEngine | None = None):
        self.config = config or ParserConfig()
        self.analyzer = PDFAnalyzer(self.config.pdf)
        self.validator = ParseQualityValidator(self.config.pdf)
        self.native = NativePDFParser(assets_dir)
        self.ocr = PaddlePDFParser(assets_dir, self.config, engine)

    def _validate_page_elements(self, elements: list[dict], data: dict, record: dict) -> list[dict]:
        # Validate at the page boundary, so one malformed backend bbox or
        # non-JSON metadata object cannot invalidate otherwise successful pages.
        page_document = Normalizer().normalize({**data, "pages": [record], "elements": elements})
        return [element.model_dump(exclude={"element_id", "order", "section_id"})
                for element in page_document.elements]

    def parse(self, path: Path) -> CanonicalDocument:
        data = document_data(path, "page_routed_pdf")
        with fitz.open(path) as pdf:
            data["metadata"]["pdf_metadata"] = pdf.metadata
            for page in pdf:
                number = page.number + 1
                record = {"page_number": number, "width": page.rect.width, "height": page.rect.height, "metadata": {}}
                elements = []
                try:
                    analysis = self.analyzer.analyze_page(page)
                    record.update(page_type=analysis.page_type)
                    record["metadata"]["analysis"] = analysis.to_dict()
                    needs_ocr = analysis.page_type == "scanned"
                    if not needs_ocr:
                        try:
                            native_elements = self.native.parse_page(page)
                            elements = self._validate_page_elements(native_elements, data, record)
                            needs_ocr = self.validator.page_needs_ocr(elements, analysis)
                        except Exception as exc:
                            logger.exception("Native parser failed on %s page %d; trying OCR", path.name, number)
                            record["metadata"]["native_error"] = str(exc)
                            needs_ocr = True
                    if needs_ocr:
                        record["metadata"].update(parser="paddleocr", ocr_fallback=True,
                                                  rendered_asset=f"assets/p{number:04d}_render.png")
                        ocr_elements = self._validate_page_elements(self.ocr.parse_page(page), data, record)
                        if self.validator.page_needs_ocr(ocr_elements, analysis):
                            raise ValueError("OCR returned empty or unusable text")
                        # Replace native text rather than appending duplicate OCR text.
                        elements = ocr_elements
                    else:
                        record["metadata"].update(parser="pymupdf", ocr_fallback=False)
                    logger.info("%s page %d: %s -> %s (%d elements)", path.name, number,
                                analysis.page_type, record["metadata"]["parser"], len(elements))
                except Exception as exc:
                    logger.exception("Page %d failed in %s", number, path.name)
                    record.update(status="failed")
                    record["metadata"]["error"] = str(exc)
                    # A failed fallback still retains any available native extraction.
                    record["metadata"]["retained_native_elements"] = bool(elements)
                data["elements"].extend(elements)
                data["pages"].append(record)
        data["metadata"]["failed_pages"] = [p["page_number"] for p in data["pages"] if p.get("status") == "failed"]
        return Normalizer().normalize(data)
