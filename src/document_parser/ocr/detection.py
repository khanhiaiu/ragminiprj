"""Text detection geometry, intentionally independent from recognition."""

import json
from typing import Any

from ..config import ParserConfig
from .models import DetectionBox, bbox_from_geometry, polygon_from_geometry


class OCRDetector:
    """Read only text locations and detector scores from a PP-Structure result."""

    def __init__(self, config: ParserConfig | None = None, backend=None):
        self.config = config or ParserConfig()
        self._backend = backend

    @property
    def model_name(self) -> str:
        return str(self.config.ocr_options["text_detection_model_name"])

    def _get_backend(self):
        if self._backend is None:
            from paddleocr import TextDetection

            options = self.config.ocr_options
            detector_names = {
                "text_det_limit_side_len": "limit_side_len",
                "text_det_limit_type": "limit_type",
                "text_det_thresh": "thresh",
                "text_det_box_thresh": "box_thresh",
                "text_det_unclip_ratio": "unclip_ratio",
            }
            self._backend = TextDetection(
                **{
                    name: options[name]
                    for name in (
                        "device",
                        "cpu_threads",
                        "enable_mkldnn",
                        "mkldnn_cache_capacity",
                        "enable_hpi",
                        "use_tensorrt",
                        "precision",
                    )
                    if name in options
                }
                | {
                    target: options[source]
                    for source, target in detector_names.items()
                    if source in options
                }
                | {
                    "model_name": options["text_detection_model_name"],
                    "model_dir": options.get("text_detection_model_dir"),
                }
            )
        return self._backend

    def predict(self, image: Any) -> list[DetectionBox]:
        """Run detection only on an image; no recognition model is initialized."""
        result = self._get_backend().predict(image)[0]
        payload = result.json if hasattr(result, "json") else result
        if callable(payload):
            payload = payload()
        if isinstance(payload, str):
            payload = json.loads(payload)
        return self.detect(payload.get("res", payload))

    def detect(self, ocr_payload: dict | None) -> list[DetectionBox]:
        payload = ocr_payload or {}
        polygons = payload.get("dt_polys")
        if polygons is None:
            polygons = payload.get("rec_polys")
        if polygons is None:
            polygons = []
        scores = payload.get("dt_scores")
        if scores is None:
            scores = []
        boxes = []
        for index, polygon in enumerate(polygons):
            confidence = float(scores[index]) if index < len(scores) else None
            boxes.append(
                DetectionBox(
                    bbox=bbox_from_geometry(polygon),
                    polygon=polygon_from_geometry(polygon),
                    confidence=confidence,
                )
            )
        return boxes
