"""Create exactly one configured correction backend."""

from importlib import import_module
from typing import Any

from .base import TextCorrector

_BACKENDS = {
    "protonx": ("document_parser.correction.protonx", "ProtonXCorrector"),
    "protonx_legal": (
        "document_parser.correction.protonx_legal",
        "ProtonXLegalCorrector",
    ),
    "bmd1905": ("document_parser.correction.bmd1905", "BMD1905Corrector"),
    "bravend": ("document_parser.correction.bravend", "BravendCorrector"),
}


class CorrectorFactory:
    @staticmethod
    def create(
        config: dict[str, Any], override: TextCorrector | None = None
    ) -> TextCorrector | None:
        if not config["enabled"]:
            return None
        if override is not None:
            return override
        backend = config["backend"]
        try:
            module_name, class_name = _BACKENDS[backend]
        except KeyError as exc:
            raise ValueError(f"Unsupported text correction backend: {backend}") from exc
        corrector_class = getattr(import_module(module_name), class_name)
        return corrector_class(config)
