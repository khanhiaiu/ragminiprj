"""Document-level orchestration for conservative OCR correction."""

import logging
from collections import Counter
from typing import Any

from ..normalization.ocr_postprocessing import record_text_change
from ..normalization.schema import CanonicalDocument, Element
from .base import TextCorrector
from .detector import CorrectionDetector
from .safety import CorrectionSafetyGate

logger = logging.getLogger(__name__)


class TextCorrectionService:
    def __init__(self, config: dict[str, Any], corrector: TextCorrector | None = None):
        self.config = config
        self.detector = CorrectionDetector(config["detector"])
        self.safety = CorrectionSafetyGate(config["safety"])
        self.corrector = corrector if config["enabled"] else None
        if config["enabled"] and self.corrector is None:
            raise ValueError("Enabled text correction requires one selected corrector")

    def _eligible(self, element: Element) -> bool:
        if element.element_type not in self.config["eligible_element_types"]:
            return False
        if element.metadata.get("exclude_from_content"):
            return False
        if self.config["only_ocr"] and element.metadata.get("parser") != "paddleocr":
            return False
        return bool(element.text.strip())

    def process(self, document: CanonicalDocument) -> CanonicalDocument:
        if not self.config["enabled"]:
            return document

        statuses = Counter()
        considered = 0
        for element in document.elements:
            if not self._eligible(element):
                continue
            decision = self.detector.decide(element.text, element.metadata)
            logger.debug(
                "Correction decision element=%s should_correct=%s reasons=%s",
                element.element_id,
                decision.should_correct,
                decision.reasons,
            )
            if not decision.should_correct:
                continue
            considered += 1
            try:
                candidate = self.corrector.correct(element.text)
                safety = self.safety.evaluate(
                    candidate,
                    allowed_edit_tokens=decision.suspicious_tokens,
                    replacement_options=decision.replacement_options,
                    allow_punctuation_edits="punctuation_anomaly" in decision.reasons,
                )
                audit = {
                    "status": safety.status,
                    "backend": candidate.backend,
                    "model": candidate.model_name,
                    "model_name": candidate.model_name,
                    "edit_ratio": safety.edit_ratio,
                    "protected_tokens_preserved": safety.protected_tokens_preserved,
                    "structure_preserved": safety.structure_preserved,
                    "detector_reasons": decision.reasons,
                    "reason": safety.reason,
                }
                if safety.accepted:
                    raw = {"text": element.text, "metadata": element.metadata}
                    record_text_change(
                        raw, candidate.corrected, f"{candidate.backend}_ocr_correction"
                    )
                    element.text = raw["text"]
                    logger.debug(
                        "Accepted correction element=%s edit_ratio=%.4f",
                        element.element_id,
                        safety.edit_ratio,
                    )
                elif safety.status != "not_needed":
                    audit["candidate"] = candidate.corrected
                    logger.warning(
                        "Rejected correction element=%s status=%s reason=%s edit_ratio=%.4f",
                        element.element_id,
                        safety.status,
                        safety.reason,
                        safety.edit_ratio,
                    )
                element.metadata["text_correction"] = audit
                statuses[safety.status] += 1
            except Exception as exc:
                logger.warning("Text correction failed for element %s: %s", element.element_id, exc)
                element.metadata["text_correction"] = {
                    "status": "error",
                    "backend": self.corrector.backend,
                    "model": self.corrector.model_name,
                    "model_name": self.corrector.model_name,
                    "detector_reasons": decision.reasons,
                    "reason": "backend_error",
                    "error": str(exc),
                }
                statuses["error"] += 1

        document.metadata["text_correction"] = {
            "enabled": True,
            "backend": self.corrector.backend,
            "model": self.corrector.model_name,
            "model_name": self.corrector.model_name,
            "considered_elements": considered,
            "accepted_elements": statuses["accepted"],
            "status_counts": dict(statuses),
        }
        logger.info(
            "Text correction considered %d element(s), accepted %d",
            considered,
            statuses["accepted"],
        )
        return document
