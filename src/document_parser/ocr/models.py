"""Small, dependency-free value objects shared by the OCR components."""

from dataclasses import dataclass, field
from typing import Any


BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class LayoutBlock:
    block_type: str
    bbox: BBox
    confidence: float | None
    reading_order: int
    content: str = ""
    block_id: int | str | None = None


@dataclass(frozen=True)
class DetectionBox:
    bbox: BBox
    polygon: list[list[float]]
    confidence: float | None = None


@dataclass(frozen=True)
class RecognizedText:
    bbox: BBox
    text: str
    confidence: float | None = None
    polygon: list[list[float]] = field(default_factory=list)

    def as_ocr_line(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "bbox": list(self.bbox),
            "coordinate_unit": "px",
            "confidence": self.confidence,
        }


def bbox_from_geometry(value: Any) -> BBox:
    """Convert a rectangle or polygon from Paddle into an axis-aligned bbox."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("Missing or invalid bounding box")
    if len(value) == 4 and all(isinstance(item, (int, float)) for item in value):
        box = tuple(float(item) for item in value)
    else:
        points = value
        if len(value) == 8 and all(isinstance(item, (int, float)) for item in value):
            points = list(zip(value[::2], value[1::2]))
        if not all(isinstance(point, (list, tuple)) and len(point) >= 2 for point in points):
            raise ValueError("Bounding box must be [x0, y0, x1, y1] or a polygon")
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        box = (min(xs), min(ys), max(xs), max(ys))
    if box[2] < box[0] or box[3] < box[1]:
        raise ValueError("Bounding box must be [left, top, right, bottom]")
    return box


def polygon_from_geometry(value: Any) -> list[list[float]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if len(value) == 4 and all(isinstance(item, (int, float)) for item in value):
        x0, y0, x1, y1 = map(float, value)
        return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    if len(value) == 8 and all(isinstance(item, (int, float)) for item in value):
        return [[float(x), float(y)] for x, y in zip(value[::2], value[1::2])]
    return [[float(point[0]), float(point[1])] for point in value]
