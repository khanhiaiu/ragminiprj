import hashlib
from pathlib import Path


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def document_data(path: Path, parser: str) -> dict:
    return {
        "document_id": file_digest(path)[:20],
        "filename": path.name,
        "file_type": path.suffix.lower().lstrip("."),
        "source_path": str(path.resolve()),
        "metadata": {"parser": parser, "size_bytes": path.stat().st_size},
        "pages": [],
        "elements": [],
    }


def table_markdown(rows: list[list]) -> str:
    if not rows:
        return ""
    width = max(map(len, rows))

    def line(row):
        cells = [
            str(v if v is not None else "").replace("|", "\\|").replace("\n", "<br>") for v in row
        ]
        return "| " + " | ".join(cells + [""] * (width - len(cells))) + " |"

    return "\n".join([line(rows[0]), line(["---"] * width), *map(line, rows[1:])])
