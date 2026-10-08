from pathlib import Path
from typing import Callable

from ..errors import UnsupportedFormatError
from ..parsers.base import BaseDocumentParser


class DocumentRouter:
    FORMATS = {
        ".pdf": "pdf",
        ".docx": "docx",
        ".xlsx": "excel",
        ".png": "image",
        ".jpg": "image",
        ".jpeg": "image",
    }

    def __init__(self, factories: dict[str, Callable[[], BaseDocumentParser]] | None = None):
        self.factories = factories or {}

    def format_for(self, path: Path) -> str:
        try:
            return self.FORMATS[path.suffix.lower()]
        except KeyError as exc:
            raise UnsupportedFormatError(f"Unsupported extension: {path.suffix}") from exc

    def route(self, path: Path) -> BaseDocumentParser:
        name = self.format_for(path)
        if name not in self.factories:
            raise UnsupportedFormatError(f"Parser not configured: {name}")
        return self.factories[name]()
