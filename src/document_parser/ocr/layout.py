"""PP-StructureV3-backed layout analysis."""

import json
from pathlib import Path
from typing import Any, Iterable

from ..config import ParserConfig
from ..errors import DependencyUnavailableError, DocumentParseError
from .models import LayoutBlock, bbox_from_geometry


def require_requested_gpu(device: str, paddle) -> None:
    """Reject silent CPU fallback when a caller explicitly requests CUDA."""
    if not device.startswith("gpu"):
        return
    if not paddle.is_compiled_with_cuda():
        raise DependencyUnavailableError(
            "GPU requested but this environment has CPU-only Paddle. "
            "Use the GPU environment and install paddlepaddle-gpu; see docs/README.md."
        )
    count = paddle.device.cuda.device_count()
    try:
        indices = (
            [int(value) for value in device.split(":", 1)[1].split(",")]
            if ":" in device
            else [0]
        )
    except ValueError as exc:
        raise DependencyUnavailableError(f"Invalid CUDA device: {device}") from exc
    if not indices or any(index < 0 or index >= count for index in indices):
        raise DependencyUnavailableError(
            f"GPU requested ({device}) but only {count} CUDA device(s) are visible. "
            "Check NVIDIA driver/device permissions; sandbox sessions may hide /dev/nvidia*."
        )


def result_payload(result: Any) -> dict:
    payload = result.json if hasattr(result, "json") else result
    if callable(payload):
        payload = payload()
    if isinstance(payload, str):
        payload = json.loads(payload)
    payload = payload.get("res", payload)
    if not isinstance(payload, dict):
        raise DocumentParseError("Unsupported Paddle result: expected a mapping")
    return payload


class LayoutDetector:
    """Run PP-StructureV3 and expose layout blocks in reading order."""

    def __init__(self, config: ParserConfig | None = None, backend=None):
        self.config = config or ParserConfig()
        self._backend = backend
        self._initialization_error: str | None = None

    def _get_backend(self):
        if self._initialization_error:
            raise DependencyUnavailableError(self._initialization_error)
        if self._backend is None:
            try:
                self.config.apply_environment()
                device = str(self.config.ocr_options.get("device") or "cpu")
                if device.startswith("gpu"):
                    import paddle

                    require_requested_gpu(device, paddle)
                from paddleocr import PPStructureV3

                self._backend = PPStructureV3(**self.config.ocr_options)
            except Exception as exc:
                self._initialization_error = (
                    f"Cannot initialize PPStructureV3: {exc}. Install requirements.txt "
                    "and a Paddle backend; see docs/README.md."
                )
                raise DependencyUnavailableError(self._initialization_error) from exc
        return self._backend

    def analyze(self, path: Path | str) -> Iterable[dict]:
        for result in self._get_backend().predict(str(path)):
            payload = result_payload(result)
            if "parsing_res_list" not in payload:
                raise DocumentParseError("Unsupported Paddle result: missing parsing_res_list")
            yield payload

    def detect(self, payload: dict) -> list[LayoutBlock]:
        blocks = []
        for index, item in enumerate(payload.get("parsing_res_list", [])):
            order = item.get("block_order")
            order = index if order is None else int(order)
            bbox = bbox_from_geometry(item.get("block_bbox"))
            confidence = item.get("block_score", item.get("score"))
            if confidence is None:
                confidence = self._layout_confidence(payload, item, bbox, index)
            blocks.append(
                LayoutBlock(
                    block_type=str(item.get("block_label") or "unknown").lower(),
                    bbox=bbox,
                    confidence=float(confidence) if confidence is not None else None,
                    reading_order=order,
                    content=str(item.get("block_content") or ""),
                    block_id=item.get("block_id"),
                )
            )
        return blocks

    @staticmethod
    def _layout_confidence(payload: dict, item: dict, bbox, index: int):
        candidates = payload.get("layout_det_res", {}).get("boxes", [])
        block_id = item.get("block_id")
        for candidate_index, candidate in enumerate(candidates):
            candidate_id = candidate.get("block_id", candidate_index)
            if block_id is not None and candidate_id == block_id:
                return candidate.get("score")
        label = str(item.get("block_label") or "").casefold()
        best = (0.0, None)
        for candidate in candidates:
            if label and str(candidate.get("label") or "").casefold() != label:
                continue
            geometry = candidate.get("coordinate", candidate.get("bbox"))
            if geometry is None:
                continue
            other = bbox_from_geometry(geometry)
            intersection = max(0.0, min(bbox[2], other[2]) - max(bbox[0], other[0])) * max(
                0.0, min(bbox[3], other[3]) - max(bbox[1], other[1])
            )
            union = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) + (
                other[2] - other[0]
            ) * (other[3] - other[1]) - intersection
            iou = intersection / union if union else 0.0
            if iou > best[0]:
                best = (iou, candidate.get("score"))
        if best[1] is not None:
            return best[1]
        if index < len(candidates):
            return candidates[index].get("score")
        return None
