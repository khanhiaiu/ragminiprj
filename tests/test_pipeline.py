import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pymupdf as fitz
import pytest
from openpyxl import Workbook
from PIL import Image

from document_parser.parsers.excel.excel_parser import ExcelParser
from document_parser.parsers.image.paddle_image_parser import PaddleEngine
from document_parser.pipeline import DocumentPipeline


def create_pdf(path, scan=False):
    with fitz.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), "Native text layer is usable. " * 3)
        if scan:
            page = pdf.new_page()
            image = path.parent / "scan.png"
            Image.new("RGB", (600, 800), "white").save(image)
            page.insert_image(page.rect, filename=str(image))
        pdf.save(path)


class FakeOCR:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def predict(self, path):
        self.calls.append(path)
        if self.fail:
            raise RuntimeError("Simulated OCR page failure")
        with Image.open(path) as image:
            w, h = image.size
        yield SimpleNamespace(json={"res": {"parsing_res_list": [
            {"block_label": "text", "block_content": "Recognized scan text", "block_bbox": [0, 0, w, h], "block_order": 0}
        ]}})


def test_native_does_not_initialize_ocr(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    path = source / "digital.pdf"
    create_pdf(path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    backend = FakeOCR()
    doc = DocumentPipeline(ocr_engine=PaddleEngine(backend=backend)).parse(path, tmp_path / "out")
    assert not backend.calls
    assert doc.pages[0].metadata["parser"] == "pymupdf"
    assert doc.metadata["quality"]["status"] == "ok"
    output = json.loads((tmp_path / "out/document.json").read_text())
    assert output["elements"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("failure", [False, True])
def test_mixed_page_routing_and_failure_isolation(tmp_path, failure):
    source = tmp_path / "input"
    source.mkdir()
    path = source / "mixed.pdf"
    create_pdf(path, scan=True)
    backend = FakeOCR(fail=failure)
    doc = DocumentPipeline(ocr_engine=PaddleEngine(backend=backend)).parse(path, tmp_path / "out")
    assert len(backend.calls) == 1
    assert doc.pages[0].metadata["parser"] == "pymupdf"
    assert doc.pages[1].page_type == "scanned"
    assert doc.pages[1].status == ("failed" if failure else "ok")
    assert doc.metadata["quality"]["failed_pages"] == ([2] if failure else [])
    assert any(e.page_number == 1 and "Native text" in e.text for e in doc.elements)
    if not failure:
        ocr = next(e for e in doc.elements if e.page_number == 2)
        assert ocr.bbox[2] == pytest.approx(doc.pages[1].width)


def test_excel_regions_formulas_and_merges(tmp_path):
    path = tmp_path / "workbook.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Revenue"
    sheet.append(["Product", "Value"])
    sheet.append(["A", 10])
    sheet["B3"] = "=SUM(B2:B2)"
    sheet["A8"] = "Second table"
    sheet.merge_cells("A8:C8")
    sheet["C9"] = 25
    sheet["F8"] = "Side table"
    sheet["F9"] = 30
    sheet.row_dimensions[2].hidden = True
    sheet.column_dimensions["B"].hidden = True
    workbook.save(path)
    doc = ExcelParser().parse(path)
    assert [e.metadata["range"] for e in doc.elements] == ["A1:B3", "A8:C9", "F8:F9"]
    assert all(e.page_number is None for e in doc.elements)
    assert doc.elements[0].metadata["hidden_rows"] == [2]
    assert next(c for c in doc.elements[0].metadata["cells"] if c["coordinate"] == "B3")["formula"] == "=SUM(B2:B2)"


def test_input_output_guard(tmp_path):
    path = tmp_path / "x.pdf"
    create_pdf(path)
    with pytest.raises(ValueError, match="outside"):
        DocumentPipeline().parse(path, tmp_path / "nested")


def test_batch_continues_after_bad_file_and_disambiguates(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    create_pdf(source / "same.pdf")
    (source / "same.docx").write_bytes(b"invalid zip")
    (source / "ignore.txt").write_text("unsupported")
    summary = DocumentPipeline().parse_all(source, tmp_path / "out")
    assert summary["counts"] == {"ok": 1, "partial": 0, "failed": 1}
    assert len({f["output_dir"] for f in summary["files"]}) == 2


@pytest.mark.integration
def test_repository_document(tmp_path):
    directory = Path(__file__).resolve().parents[1] / "documents"
    pdfs = sorted(directory.glob("*.pdf"))
    if not pdfs:
        pytest.skip("No repository PDF fixture")
    doc = DocumentPipeline().parse(pdfs[0], tmp_path / "repository_parse")
    assert doc.elements
    assert len(doc.pages) > 0


@pytest.mark.integration
def test_repository_docx(tmp_path):
    path = Path(__file__).resolve().parents[1] / "documents/rag_system_docs.docx"
    if not path.exists():
        pytest.skip("No repository DOCX fixture")
    doc = DocumentPipeline().parse(path, tmp_path / "repository_docx")
    assert doc.metadata["parser"] == "docling"
    assert doc.metadata["quality"]["status"] == "ok"
    assert any(e.element_type == "heading" for e in doc.elements)


def test_native_quality_failure_triggers_page_ocr(tmp_path, monkeypatch):
    from document_parser.parsers.pdf.native_pdf_parser import NativePDFParser
    source = tmp_path / "input"
    source.mkdir()
    path = source / "native-bad.pdf"
    create_pdf(path)
    monkeypatch.setattr(NativePDFParser, "parse_page", lambda self, page: [
        {"element_type": "paragraph", "text": "\ufffd" * 100, "page_number": page.number + 1}])
    backend = FakeOCR()
    doc = DocumentPipeline(ocr_engine=PaddleEngine(backend=backend)).parse(path, tmp_path / "out")
    assert len(backend.calls) == 1
    assert doc.pages[0].page_type == "digital"
    assert doc.pages[0].metadata["ocr_fallback"]
    assert not any("\ufffd" in e.text for e in doc.elements)


def test_bad_ocr_bbox_fails_only_affected_page(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    path = source / "mixed.pdf"
    create_pdf(path, scan=True)

    class MalformedOCR:
        def predict(self, path):
            yield SimpleNamespace(json={"res": {"parsing_res_list": [
                {"block_label": "text", "block_content": "Malformed bounding box",
                 "block_bbox": [100, 100, 10, 10], "block_order": 0}
            ]}})

    document = DocumentPipeline(ocr_engine=PaddleEngine(backend=MalformedOCR())).parse(path, tmp_path / "out")
    assert document.metadata["quality"]["failed_pages"] == [2]
    assert document.pages[0].status == "ok"
    assert any("Native text" in element.text for element in document.elements)
