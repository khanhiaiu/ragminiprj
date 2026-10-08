"""Backend-neutral correction contracts."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class CorrectionCandidate:
    original: str
    corrected: str
    backend: str
    model_name: str
    changed: bool


class TextCorrector(Protocol):
    """A correction backend returns a candidate, never an authoritative edit."""

    backend: str
    model_name: str

    @property
    def initialized(self) -> bool: ...

    def correct(self, text: str) -> CorrectionCandidate: ...
