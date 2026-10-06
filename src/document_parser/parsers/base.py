from pathlib import Path
from typing import Protocol

from ..normalization.schema import CanonicalDocument


class BaseDocumentParser(Protocol):
    def parse(self, path: Path) -> CanonicalDocument: ...
