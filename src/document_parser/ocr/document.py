"""Document-level routing for PP-StructureV3 layout and specialized OCR components."""

import logging
import unicodedata
from pathlib import Path

from PIL import Image

from ..config import ParserConfig
from ..normalization.ocr_postprocessing import overlap, prepare_blocks, process_page
from .detection import OCRDetector
from .layout import LayoutDetector
from .models import RecognizedText
from .recognition import OCRRecognizer
from .table import TableRecognizer

logger = logging.getLogger(__name__)


def _compact_text(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).split())


def recover_unassigned_ocr_lines(
    elements: list[dict], ocr: dict, duplicate_overlap: float = 0.8, coverage_ratio: float = 0.5
) -> list[dict]:
    """Retain OCR lines omitted by layout without flattening table contents."""
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
            bbox = (
                min(p[0] for p in points),
                min(p[1] for p in points),
                max(p[0] for p in points),
                max(p[1] for p in points),
            )
        else:
            continue
        x0, y0, x1, y1 = bbox
        area = max(0, x1 - x0) * max(0, y1 - y0)
        if not area:
            continue
        compact = _compact_text(text)
        covered = False
        for element in elements:
            coverage, containment = overlap(bbox, element["bbox"])
            if (
                compact == _compact_text(element.get("text", ""))
                and containment >= duplicate_overlap
            ):
                element["metadata"].setdefault("duplicate_ocr_lines", []).append(
                    {"text": text, "bbox": list(bbox), "coordinate_unit": "px"}
                )
                covered = True
                break
            if coverage < coverage_ratio:
                continue
            if element["element_type"] in {
                "table",
                "formula",
                "equation",
            } or compact in _compact_text(element.get("text", "")):
                covered = True
                break
        if covered:
            continue
        metadata = {
            "parser": "paddleocr",
            "pipeline": "PP-StructureV3",
            "original_label": "ocr_line",
            "coordinate_unit": "px",
            "layout_fallback": True,
        }
        if index < len(scores):
            metadata["ocr_confidence"] = float(scores[index])
        metadata["ocr_lines"] = [
            {
                "text": text,
                "bbox": list(bbox),
                "coordinate_unit": "px",
                "confidence": metadata.get("ocr_confidence"),
            }
        ]
        recovered.append(
            {"element_type": "paragraph", "text": text, "bbox": bbox, "metadata": metadata}
        )

    merged = list(elements)
    for line in sorted(recovered, key=lambda item: (item["bbox"][1], item["bbox"][0])):
        x0, y0, x1, _ = line["bbox"]
        anchors = [
            index
            for index, element in enumerate(merged)
            if min(x1, element["bbox"][2]) > max(x0, element["bbox"][0])
        ]
        if not anchors:
            anchors = list(range(len(merged)))
        following = next((i for i in anchors if merged[i]["bbox"][1] > y0), None)
        position = following if following is not None else (anchors[-1] + 1 if anchors else 0)
        merged.insert(position, line)
    if recovered:
        logger.info("Retained %d OCR lines omitted by layout", len(recovered))
    return merged


class DocumentParser:
    """Coordinate layout, text detection/recognition, tables, images and assembly."""

    IMAGE_TYPES = {"image", "figure", "chart", "seal", "picture"}

    def __init__(
        self,
        config: ParserConfig | None = None,
        backend=None,
        *,
        layout_detector: LayoutDetector | None = None,
        ocr_detector: OCRDetector | None = None,
        ocr_recognizer: OCRRecognizer | None = None,
        table_recognizer: TableRecognizer | None = None,
    ):
        self.config = config or ParserConfig()
        self.layout_detector = layout_detector or LayoutDetector(self.config, backend)
        self.ocr_detector = ocr_detector or OCRDetector(self.config)
        self.ocr_recognizer = ocr_recognizer or OCRRecognizer(self.config)
        self.table_recognizer = table_recognizer or TableRecognizer()

    def retry_heading(self, element, match, scales):
        """Re-read only a suspect heading using the configured recognition model."""
        metadata = element.metadata
        source = metadata.get("rendered_source")
        if not source:
            return []
        prefix = "Điều" if match.group(1) else "CHƯƠNG"
        line = next(
            (
                line
                for line in metadata.get("ocr_lines", [])
                if line["text"].casefold().startswith(prefix.casefold())
            ),
            None,
        )
        with Image.open(source) as image:
            if line:
                box = line["bbox"]
            elif metadata.get("coordinate_unit") == "px":
                box = element.bbox
            else:
                return []
            padding = self.config.postprocessing["retry_crop_padding"]
            box = (
                max(0, int(box[0]) - padding),
                max(0, int(box[1]) - padding),
                min(image.width, int(box[2]) + padding),
                min(image.height, int(box[3]) + padding),
            )
            crop = image.crop(box).convert("RGB")
            readings = []
            for scale in dict.fromkeys(scales):
                resized = crop.resize(
                    (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
                    Image.Resampling.LANCZOS,
                )
                recognized = self.ocr_recognizer.recognize(resized)
                readings.append(
                    {"scale": scale, "text": recognized.text, "confidence": recognized.confidence}
                )
            return readings

    def predict(
        self, path: Path, assets_dir: Path | None = None, prefix: str = "ocr"
    ) -> list[dict]:
        elements = []
        with Image.open(path) as source:
            image = source.convert("RGB")
            for payload in self.layout_detector.analyze(path):
                page_elements = self._route_blocks(payload, image, assets_dir, prefix, len(elements))
                settings = self.config.postprocessing
                ocr = self._normalized_ocr(payload)
                page_elements = prepare_blocks(page_elements, ocr, settings)
                page_elements = recover_unassigned_ocr_lines(
                    page_elements,
                    ocr,
                    settings["duplicate_overlap"],
                    settings["line_assignment_ratio"],
                )
                for element in page_elements:
                    element["metadata"]["rendered_source"] = str(path.resolve())
                elements.extend(process_page(page_elements, image.width, image.height, settings))
        return elements

    def _route_blocks(self, payload, image, assets_dir, prefix, offset):
        blocks = self.layout_detector.detect(payload)
        overall_ocr = payload.get("overall_ocr_res") or {}
        detections = self.ocr_detector.detect(overall_ocr)
        recognized = self.ocr_recognizer.collect(overall_ocr, detections)
        table_results = payload.get("table_res_list") or []
        table_blocks = [block for block in blocks if block.block_type == "table"]
        matched_table_results = self.table_recognizer.match_results(table_blocks, table_results)
        table_index = 0
        elements = []
        for block in blocks:
            metadata = {
                "parser": "paddleocr",
                "pipeline": "PP-StructureV3",
                "original_label": block.block_type,
                "coordinate_unit": "px",
                "block_id": block.block_id,
                "reading_order": block.reading_order,
                "block_order": block.reading_order,
                "layout_confidence": block.confidence,
                "recognition_model": self.ocr_recognizer.model_name,
            }
            content = block.content
            if block.block_type == "table":
                table_result = (
                    matched_table_results[table_index]
                    if table_index < len(matched_table_results)
                    else None
                )
                table_index += 1
                table = self.table_recognizer.process(block, table_result, recognized)
                table["recognition_model"] = self.ocr_recognizer.model_name
                metadata.update(
                    table=table,
                    table_html=table["html"],
                    table_markdown=table["markdown"],
                    cells=table["cells"],
                    rows=table["rows"],
                    columns=table["columns"],
                )
                # Old/injected payloads may only contain layout block HTML. Preserve
                # that compatibility; real PP-StructureV3 table results use the
                # reconstructed Markdown/HTML representation above.
                if table_result is not None:
                    content = table["markdown"] or table["html"]
                else:
                    content = block.content
            elif block.block_type in self.IMAGE_TYPES and assets_dir:
                assets_dir.mkdir(parents=True, exist_ok=True)
                filename = f"{prefix}_region_{offset + len(elements) + 1:03d}.png"
                image.crop(block.bbox).save(assets_dir / filename)
                metadata["asset_path"] = f"assets/{filename}"
            elements.append(
                {
                    "element_type": block.block_type,
                    "text": content,
                    "bbox": block.bbox,
                    "metadata": metadata,
                }
            )
        return elements

    def _normalized_ocr(self, payload: dict) -> dict:
        raw = payload.get("overall_ocr_res") or {}
        detections = self.ocr_detector.detect(raw)
        lines = self.ocr_recognizer.collect(raw, detections)
        return {
            "rec_texts": [line.text for line in lines],
            "rec_boxes": [list(line.bbox) for line in lines],
            "rec_polys": [line.polygon for line in lines],
            "rec_scores": [line.confidence if line.confidence is not None else 0.0 for line in lines],
        }
