from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from document_parser.parsers.docx.docling_parser import DoclingDOCXParser
from document_parser.parsers.image.paddle_image_parser import PaddleEngine, PaddleImageParser


def test_image_preserves_original_and_table_structure(tmp_path):
    path = tmp_path / "source.png"
    Image.new("RGB", (200, 100), "white").save(path)

    class Backend:
        def predict(self, path):
            yield SimpleNamespace(json={"res": {"parsing_res_list": [
                {"block_label": "table", "block_content": "<table><tr><td>Value</td></tr></table>",
                 "block_bbox": [10, 10, 100, 80], "block_order": 0},
                {"block_label": "image", "block_content": "", "block_bbox": [100, 0, 200, 100], "block_order": 1},
            ]}})

    assets = tmp_path / "output/assets"
    doc = PaddleImageParser(assets, PaddleEngine(backend=Backend())).parse(path)
    assert (assets / "original.png").read_bytes() == path.read_bytes()
    table = next(e for e in doc.elements if e.element_type == "table")
    assert "<table>" in table.metadata["table_html"]
    crop = next(e for e in doc.elements if e.metadata.get("original_label") == "image")
    assert (assets.parent / crop.metadata["asset_path"]).is_file()


def test_ocr_recovers_layout_omissions_without_duplicate_table_or_text(tmp_path):
    path = tmp_path / "scan.png"
    Image.new("RGB", (200, 200), "white").save(path)

    class Backend:
        def predict(self, path):
            yield SimpleNamespace(json={"res": {
                "parsing_res_list": [
                    {"block_label": "text", "block_content": "Body content", "block_bbox": [10, 50, 190, 80]},
                    {"block_label": "table", "block_content": "<table><tr><td>42</td></tr></table>",
                     "block_bbox": [10, 100, 190, 150]},
                ],
                "overall_ocr_res": {
                    "rec_texts": ["Missing title", "Body content", "Missing middle", "42", "Missing tail"],
                    "rec_boxes": [[10, 10, 180, 30], [10, 50, 180, 70], [10, 85, 180, 95],
                                  [10, 105, 40, 120], [10, 160, 180, 180]],
                    "rec_scores": [0.9] * 5,
                },
            }})

    parsed = PaddleEngine(backend=Backend()).predict(path)
    assert [e["text"] for e in parsed] == ["Missing title", "Body content", "Missing middle",
                                           "<table><tr><td>42</td></tr></table>", "Missing tail"]
    recovered = [e for e in parsed if e["metadata"].get("layout_fallback")]
    assert len(recovered) == 3
    assert all(e["metadata"]["ocr_confidence"] == 0.9 for e in recovered)


def test_docling_adapter_uses_structural_order(tmp_path):
    path = tmp_path / "x.docx"
    path.write_bytes(b"fake fixture for injected converter")
    items = [SimpleNamespace(label=SimpleNamespace(value="section_header"), text="Heading", level=1,
                              self_ref="#/texts/0", parent=None, prov=[]),
             SimpleNamespace(label=SimpleNamespace(value="text"), text="Body", self_ref="#/texts/1",
                              parent=SimpleNamespace(cref="#/texts/0"), prov=[])]

    class Document:
        pages = {}
        def iterate_items(self):
            return [(item, 1) for item in items]

    class Converter:
        def convert(self, path):
            return SimpleNamespace(document=Document(), status=SimpleNamespace(value="success"), errors=[])

    doc = DoclingDOCXParser(converter=Converter()).parse(path)
    assert [e.element_type for e in doc.elements] == ["heading", "paragraph"]
    assert doc.elements[1].parent_id == doc.elements[0].element_id
    assert doc.elements[0].page_number is None


def test_real_docling_images_and_tables(tmp_path):
    docx = pytest.importorskip("docx")
    pytest.importorskip("docling")
    source = tmp_path / "source.docx"
    picture = tmp_path / "picture.png"
    Image.new("RGB", (40, 40), "blue").save(picture)
    word = docx.Document()
    word.add_heading("Structured document", level=1)
    word.add_paragraph("Paragraph content")
    word.add_paragraph("List content", style="List Bullet")
    table = word.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Product"
    table.cell(0, 1).text = "Value"
    table.cell(1, 0).text = "A"
    table.cell(1, 1).text = "100"
    word.add_picture(str(picture))
    word.save(source)
    assets = tmp_path / "output/assets"
    parsed = DoclingDOCXParser(assets).parse(source)
    assert {"heading", "paragraph", "list", "table", "image"} <= {e.element_type for e in parsed.elements}
    table_element = next(e for e in parsed.elements if e.element_type == "table")
    assert table_element.metadata["table_data"]["num_rows"] == 2
    image_element = next(e for e in parsed.elements if e.element_type == "image")
    assert (assets.parent / image_element.metadata["asset_path"]).is_file()
