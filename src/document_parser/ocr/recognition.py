"""Vietnamese-capable text recognition for already detected/cropped text."""

import json
from typing import Any

from ..config import ParserConfig
from .models import DetectionBox, RecognizedText, bbox_from_geometry, polygon_from_geometry


class OCRRecognizer:
    """Recognize text only; this component never infers layout or table structure."""

    def __init__(self, config: ParserConfig | None = None, backend=None):
        self.config = config or ParserConfig()
        self._backend = backend

    @property
    def model_name(self) -> str:
        return self.config.recognition_model

    def _get_backend(self):
        if self._backend is None:
            from paddleocr import TextRecognition

            options = self.config.ocr_options
            self._backend = TextRecognition(
                **{
                    name: options[name]
                    for name in (
                        "device",
                        "cpu_threads",
                        "enable_mkldnn",
                        "mkldnn_cache_capacity",
                    )
                    if name in options
                }
                | {
                    "model_name": options["text_recognition_model_name"],
                    "model_dir": options.get("text_recognition_model_dir"),
                }
            )
        return self._backend

    def recognize(self, cropped_image: Any) -> RecognizedText:
        """Recognize one cropped text/cell image with the configured recognition model."""
        import numpy as np

        image = np.asarray(cropped_image)
        if image.ndim == 3:
            image = image[:, :, ::-1].copy()
        result = self._get_backend().predict(image)[0]
        payload = result.json if hasattr(result, "json") else result
        if callable(payload):
            payload = payload()
        if isinstance(payload, str):
            payload = json.loads(payload)
        payload = payload.get("res", payload)
        height, width = image.shape[:2]
        return RecognizedText(
            bbox=(0.0, 0.0, float(width), float(height)),
            text=str(payload.get("rec_text") or ""),
            confidence=(
                float(payload["rec_score"]) if payload.get("rec_score") is not None else None
            ),
        )

    def collect(
        self, ocr_payload: dict | None, detections: list[DetectionBox] | None = None
    ) -> list[RecognizedText]:
        """Pair PP's recognition output with geometry without changing the detector API."""
        payload = ocr_payload or {}
        texts = payload.get("rec_texts")
        if texts is None:
            texts = []
        scores = payload.get("rec_scores")
        if scores is None:
            scores = []
        geometries = payload.get("rec_boxes")
        if geometries is None:
            geometries = payload.get("rec_polys")
        if geometries is None:
            geometries = []
        lines = []
        for index, text in enumerate(texts):
            if not str(text).strip():
                continue
            if index < len(geometries):
                geometry = geometries[index]
                bbox = bbox_from_geometry(geometry)
                polygon = polygon_from_geometry(geometry)
            elif detections and index < len(detections):
                bbox = detections[index].bbox
                polygon = detections[index].polygon
            else:
                continue
            lines.append(
                RecognizedText(
                    bbox=bbox,
                    polygon=polygon,
                    text=str(text),
                    confidence=float(scores[index]) if index < len(scores) else None,
                )
            )
        return lines
