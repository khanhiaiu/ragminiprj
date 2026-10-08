"""Conservative, auditable OCR text correction."""

from .base import CorrectionCandidate, TextCorrector
from .detector import CorrectionDecision, CorrectionDetector
from .factory import CorrectorFactory
from .safety import CorrectionSafetyGate, SafetyDecision
from .service import TextCorrectionService

__all__ = [
    "BMD1905Corrector",
    "BravendCorrector",
    "CorrectionCandidate",
    "CorrectionDecision",
    "CorrectionDetector",
    "CorrectionSafetyGate",
    "CorrectorFactory",
    "ProtonXCorrector",
    "ProtonXLegalCorrector",
    "SafetyDecision",
    "TextCorrectionService",
    "TextCorrector",
]


def __getattr__(name: str):
    """Keep concrete adapters lazy when importing the correction package."""
    if name == "ProtonXCorrector":
        from .protonx import ProtonXCorrector

        return ProtonXCorrector
    if name == "ProtonXLegalCorrector":
        from .protonx_legal import ProtonXLegalCorrector

        return ProtonXLegalCorrector
    if name == "BMD1905Corrector":
        from .bmd1905 import BMD1905Corrector

        return BMD1905Corrector
    if name == "BravendCorrector":
        from .bravend import BravendCorrector

        return BravendCorrector
    raise AttributeError(name)
