from pathlib import Path

from ..errors import DocumentParseError, UnsupportedFormatError

SUPPORTED_EXTENSIONS = frozenset({".pdf", ".docx", ".xlsx", ".png", ".jpg", ".jpeg"})


class FileLoader:
    def load(self, path: Path | str) -> Path:
        path = Path(path).resolve()
        if not path.is_file():
            raise DocumentParseError(f"File does not exist: {path}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFormatError(f"Unsupported extension: {path.suffix}")
        if path.stat().st_size == 0:
            raise DocumentParseError(f"Empty file: {path}")
        return path

    def discover(self, directory: Path | str) -> list[Path]:
        directory = Path(directory).resolve()
        if not directory.is_dir():
            raise DocumentParseError(f"Input directory does not exist: {directory}")
        return sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
