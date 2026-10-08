import logging
from pathlib import Path

import pymupdf as fitz

from ...config import ParserConfig
from ...normalization.normalizer import Normalizer
from ...normalization.schema import CanonicalDocument
from ...utils.file_utils import document_data
from ..image.paddle_image_parser import PaddleEngine
from .paddle_pdf_parser import PaddlePDFParser

logger = logging.getLogger(__name__)


class PDFParser:
    """OCR every PDF page; native text extraction is deliberately not a route."""

    def __init__(
        self,
        assets_dir: Path,
        config: ParserConfig | None = None,
        engine: PaddleEngine | None = None,
    ):
        self.config = config or ParserConfig()
        self.ocr = PaddlePDFParser(assets_dir, self.config, engine)

    def _validate_page_elements(self, elements: list[dict], data: dict, record: dict) -> list[dict]:
        # Validate at the page boundary, so one malformed backend bbox or
        # non-JSON metadata object cannot invalidate otherwise successful pages.
        page_document = Normalizer().normalize({**data, "pages": [record], "elements": elements})
        return [
            element.model_dump(exclude={"element_id", "order", "section_id"})
            for element in page_document.elements
        ]

    def parse(self, path: Path) -> CanonicalDocument:
        data = document_data(path, "paddleocr")
        with fitz.open(path) as pdf:
            data["metadata"]["pdf_metadata"] = pdf.metadata
            for page in pdf:
                number = page.number + 1
                record = {
                    "page_number": number,
                    "width": page.rect.width,
                    "height": page.rect.height,
                    "page_type": "scanned",
                    "metadata": {
                        "parser": "paddleocr",
                        "ocr_forced": True,
                        "ocr_fallback": False,
                        "rendered_asset": f"assets/p{number:04d}_render.png",
                    },
                }
                try:
                    elements = self._validate_page_elements(
                        self.ocr.parse_page(page), data, record
                    )
                except Exception as exc:
                    logger.exception("OCR failed on page %d in %s", number, path.name)
                    record.update(status="failed")
                    record["metadata"]["error"] = str(exc)
                    elements = []
                data["elements"].extend(elements)
                data["pages"].append(record)
        data["metadata"]["failed_pages"] = [
            p["page_number"] for p in data["pages"] if p.get("status") == "failed"
        ]
        return Normalizer().normalize(data)
