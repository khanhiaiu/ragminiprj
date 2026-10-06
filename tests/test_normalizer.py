import json

import pytest

from document_parser.normalization.normalizer import Normalizer
from document_parser.normalization.schema import CanonicalDocument


@pytest.mark.parametrize("label,expected", [("text", "paragraph"), ("section_header", "heading"),
    ("picture", "image"), ("table", "table"), ("future-label", "unknown")])
def test_plain_parser_outputs(label, expected):
    doc = Normalizer().normalize({"document_id": "doc", "filename": "x", "file_type": "pdf", "source_path": "/x",
        "elements": [{"element_type": label, "text": " Text ", "metadata": {"parser": "example"}}]})
    assert isinstance(doc, CanonicalDocument)
    assert doc.elements[0].element_type == expected
    assert json.loads(doc.model_dump_json())["elements"][0]["text"] == "Text"
    assert Normalizer().normalize(doc) == doc


def test_section_and_ids():
    doc = Normalizer().normalize({"document_id": "doc", "filename": "x", "file_type": "pdf", "source_path": "/x",
        "elements": [{"element_type": "heading", "text": "Title"}, {"element_type": "paragraph", "text": "Body"}]})
    assert doc.elements[0].section_id == doc.elements[1].section_id
    assert len({e.element_id for e in doc.elements}) == 2
