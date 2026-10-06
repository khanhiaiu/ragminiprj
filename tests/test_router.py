from pathlib import Path

import pytest

from document_parser.errors import UnsupportedFormatError
from document_parser.router.document_router import DocumentRouter


@pytest.mark.parametrize("suffix,expected", [(".pdf", "pdf"), (".PDF", "pdf"), (".docx", "docx"),
    (".xlsx", "excel"), (".png", "image"), (".jpg", "image"), (".jpeg", "image")])
def test_router(suffix, expected):
    assert DocumentRouter().format_for(Path("fixture" + suffix)) == expected


def test_unsupported():
    with pytest.raises(UnsupportedFormatError):
        DocumentRouter().format_for(Path("file.txt"))
