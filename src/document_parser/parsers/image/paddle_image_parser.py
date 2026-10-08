"""Image adapter for the composable PP-StructureV3 document parser."""

import logging
import shutil
from pathlib import Path

from PIL import Image

from ...normalization.normalizer import Normalizer
from ...normalization.schema import CanonicalDocument
from ...ocr import DocumentParser, recover_unassigned_ocr_lines, require_requested_gpu
from ...utils.file_utils import document_data

logger = logging.getLogger(__name__)


class PaddleEngine(DocumentParser):
    """Backward-compatible name for the new document-level OCR orchestrator."""


class PaddleImageParser:
    def __init__(self, assets_dir: Path | None = None, engine: PaddleEngine | None = None):
        self.assets_dir = assets_dir
        self.engine = engine or PaddleEngine()

    def parse(self, path: Path) -> CanonicalDocument:
        data = document_data(path, "paddleocr")
        original_ref = str(path.resolve())
        if self.assets_dir:
            self.assets_dir.mkdir(parents=True, exist_ok=True)
            filename = "original" + path.suffix.lower()
            shutil.copy2(path, self.assets_dir / filename)
            original_ref = f"assets/{filename}"
        with Image.open(path) as image:
            width, height = image.size
        data["pages"] = [
            {
                "page_number": 1,
                "width": width,
                "height": height,
                "page_type": "scanned",
                "metadata": {"coordinate_unit": "px", "original_asset": original_ref},
            }
        ]
        data["elements"].append(
            {
                "element_type": "image",
                "page_number": 1,
                "bbox": (0, 0, width, height),
                "metadata": {"parser": "source", "asset_path": original_ref, "original": True},
            }
        )
        try:
            for element in self.engine.predict(path, self.assets_dir):
                element["page_number"] = 1
                table = element.get("metadata", {}).get("table")
                if table:
                    table["page"] = 1
                data["elements"].append(element)
        except Exception as exc:
            logger.exception("Image OCR failed: %s", path)
            data["pages"][0]["status"] = "failed"
            data["pages"][0]["metadata"]["error"] = str(exc)
            data["metadata"]["failed_pages"] = [1]
        return Normalizer().normalize(data)


__all__ = [
    "PaddleEngine",
    "PaddleImageParser",
    "recover_unassigned_ocr_lines",
    "require_requested_gpu",
]
