import logging
from pathlib import Path

import pymupdf as fitz
from PIL import Image

from ...config import ParserConfig
from ...normalization.normalizer import Normalizer
from ...normalization.schema import CanonicalDocument
from ...utils.file_utils import document_data
from ..image.paddle_image_parser import PaddleEngine

logger = logging.getLogger(__name__)


class PaddlePDFParser:
    def __init__(self, assets_dir: Path, config: ParserConfig | None = None, engine: PaddleEngine | None = None):
        self.assets_dir = assets_dir
        self.config = config or ParserConfig()
        self.engine = engine or PaddleEngine(self.config)

    def parse_page(self, page) -> list[dict]:
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        number = page.number + 1
        filename = f"p{number:04d}_render.png"
        path = self.assets_dir / filename
        page.get_pixmap(dpi=self.config.ocr_dpi, alpha=False).save(path)
        with Image.open(path) as image:
            scale_x = page.rect.width / image.width
            scale_y = page.rect.height / image.height
        elements = self.engine.predict(path, self.assets_dir, f"p{number:04d}")
        for element in elements:
            bbox = element.get("bbox")
            if bbox:
                element["bbox"] = (bbox[0] * scale_x, bbox[1] * scale_y, bbox[2] * scale_x, bbox[3] * scale_y)
            element["page_number"] = number
            element["metadata"].update(coordinate_unit="pt", rendered_asset=f"assets/{filename}", dpi=self.config.ocr_dpi)
        return elements

    def parse(self, path: Path) -> CanonicalDocument:
        data = document_data(path, "paddleocr")
        with fitz.open(path) as pdf:
            for page in pdf:
                record = {"page_number": page.number + 1, "width": page.rect.width, "height": page.rect.height, "page_type": "scanned"}
                try:
                    data["elements"].extend(self.parse_page(page))
                except Exception as exc:
                    logger.exception("OCR failed on page %d", page.number + 1)
                    record.update(status="failed", metadata={"error": str(exc)})
                data["pages"].append(record)
        data["metadata"]["failed_pages"] = [p["page_number"] for p in data["pages"] if p.get("status") == "failed"]
        return Normalizer().normalize(data)
