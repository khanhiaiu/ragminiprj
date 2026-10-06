import json
import logging
import shutil
import unicodedata
from pathlib import Path

from PIL import Image

from ...config import ParserConfig
from ...errors import DependencyUnavailableError, DocumentParseError
from ...normalization.normalizer import Normalizer
from ...normalization.schema import CanonicalDocument
from ...utils.file_utils import document_data

logger = logging.getLogger(__name__)


def _compact_text(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).split())


def recover_unassigned_ocr_lines(elements: list[dict], ocr: dict) -> list[dict]:
    """Retain OCR lines omitted by layout, without duplicating assigned text.

    Coverage requires both matching text and geometry. Tables/formulas use
    geometry because their structured representation differs from OCR text.
    Layout order remains authoritative; recovered lines use nearby column anchors.
    """
    recovered = []
    boxes = ocr.get("rec_boxes", [])
    polygons = ocr.get("rec_polys", [])
    scores = ocr.get("rec_scores", [])
    for index, text in enumerate(ocr.get("rec_texts", [])):
        if not text or not text.strip():
            continue
        if index < len(boxes):
            bbox = tuple(float(v) for v in boxes[index])
        elif index < len(polygons) and polygons[index]:
            points = polygons[index]
            bbox = (min(p[0] for p in points), min(p[1] for p in points),
                    max(p[0] for p in points), max(p[1] for p in points))
        else:
            continue
        x0, y0, x1, y1 = bbox
        area = max(0, x1 - x0) * max(0, y1 - y0)
        if not area:
            continue
        compact = _compact_text(text)
        covered = False
        for element in elements:
            ex0, ey0, ex1, ey1 = element["bbox"]
            intersection = max(0, min(x1, ex1) - max(x0, ex0)) * max(0, min(y1, ey1) - max(y0, ey0))
            if intersection / area < 0.5:
                continue
            if (element["element_type"] in {"table", "formula", "equation"}
                    and element.get("text")) or compact in _compact_text(element.get("text", "")):
                covered = True
                break
        if covered:
            continue
        metadata = {"parser": "paddleocr", "original_label": "ocr_line", "coordinate_unit": "px",
                    "layout_fallback": True}
        if index < len(scores):
            metadata["ocr_confidence"] = float(scores[index])
        recovered.append({"element_type": "paragraph", "text": text, "bbox": bbox, "metadata": metadata})

    merged = list(elements)
    for line in sorted(recovered, key=lambda item: (item["bbox"][1], item["bbox"][0])):
        x0, y0, x1, _ = line["bbox"]
        anchors = [i for i, element in enumerate(merged)
                   if min(x1, element["bbox"][2]) > max(x0, element["bbox"][0])]
        if not anchors:
            anchors = list(range(len(merged)))
        following = next((i for i in anchors if merged[i]["bbox"][1] > y0), None)
        position = following if following is not None else (anchors[-1] + 1 if anchors else 0)
        merged.insert(position, line)
    if recovered:
        logger.info("Retained %d OCR lines omitted by layout", len(recovered))
    return merged


class PaddleEngine:
    """Lazy, injectable PPStructureV3 adapter shared by one pipeline instance."""
    def __init__(self, config: ParserConfig | None = None, backend=None):
        self.config = config or ParserConfig()
        self._backend = backend
        self._initialization_error: str | None = None

    def predict(self, path: Path, assets_dir: Path | None = None, prefix: str = "ocr") -> list[dict]:
        if self._initialization_error:
            raise DependencyUnavailableError(self._initialization_error)
        if self._backend is None:
            try:
                from paddleocr import PPStructureV3
                self._backend = PPStructureV3(**self.config.ocr_options)
            except Exception as exc:
                self._initialization_error = f"Cannot initialize PPStructureV3: {exc}. Install .[ocr] and provision model cache."
                raise DependencyUnavailableError(self._initialization_error) from exc
        elements = []
        with Image.open(path) as source:
            image = source.convert("RGB")
            for result in self._backend.predict(str(path)):
                payload = result.json
                if callable(payload):
                    payload = payload()
                if isinstance(payload, str):
                    payload = json.loads(payload)
                payload = payload.get("res", payload)
                if "parsing_res_list" not in payload:
                    raise DocumentParseError("Unsupported Paddle result: missing parsing_res_list")
                page_elements = []
                for block in payload["parsing_res_list"]:
                    bbox = tuple(float(v) for v in block["block_bbox"])
                    label = block["block_label"]
                    content = block.get("block_content") or ""
                    metadata = {"parser": "paddleocr", "original_label": label, "coordinate_unit": "px",
                                "block_id": block.get("block_id"), "block_order": block.get("block_order")}
                    if label == "table":
                        metadata["table_html"] = content
                    if assets_dir and label in {"image", "figure", "chart", "seal"}:
                        assets_dir.mkdir(parents=True, exist_ok=True)
                        filename = f"{prefix}_region_{len(elements) + len(page_elements) + 1:03d}.png"
                        image.crop(bbox).save(assets_dir / filename)
                        metadata["asset_path"] = f"assets/{filename}"
                    page_elements.append({"element_type": label, "text": content, "bbox": bbox, "metadata": metadata})
                elements.extend(recover_unassigned_ocr_lines(page_elements, payload.get("overall_ocr_res", {})))
        return elements


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
        data["pages"] = [{"page_number": 1, "width": width, "height": height, "page_type": "scanned",
                          "metadata": {"coordinate_unit": "px", "original_asset": original_ref}}]
        data["elements"].append({"element_type": "image", "page_number": 1, "bbox": (0, 0, width, height),
                                 "metadata": {"parser": "source", "asset_path": original_ref, "original": True}})
        try:
            for element in self.engine.predict(path, self.assets_dir):
                element["page_number"] = 1
                data["elements"].append(element)
        except Exception as exc:
            logger.exception("Image OCR failed: %s", path)
            data["pages"][0]["status"] = "failed"
            data["pages"][0]["metadata"]["error"] = str(exc)
            data["metadata"]["failed_pages"] = [1]
        return Normalizer().normalize(data)
