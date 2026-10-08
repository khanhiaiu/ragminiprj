import re

import pytest

from document_parser.retrieval.models import RetrievalDocument, RetrievalElement
from rag.chunking.chunker import ChunkingConfig, TypeAwareChunker


class Tokenizer:
    name_or_path = "test"

    def __init__(self):
        self.forward = {}
        self.reverse = {}

    def encode(self, text, *, add_special_tokens=False):
        result = []
        for value in re.findall(r"\S+", text):
            if value not in self.forward:
                index = len(self.forward) + 1
                self.forward[value] = index
                self.reverse[index] = value
            result.append(self.forward[value])
        return result

    def decode(self, ids, **kwargs):
        return " ".join(self.reverse[index] for index in ids)


def element(element_id, text, *, kind="paragraph", order=0, **kwargs):
    return RetrievalElement(
        element_id=element_id,
        element_type=kind,
        text=text,
        text_for_embedding=text,
        document_id="doc",
        page_number=kwargs.pop("page_number", 1),
        order=order,
        heading_path=kwargs.pop("heading_path", ["Mục"]),
        source_element_ids=[element_id],
        source_spans=[{"element_id": element_id, "order": order}],
        **kwargs,
    )


def document(elements):
    return RetrievalDocument(
        document_id="doc",
        filename="doc.pdf",
        file_type="pdf",
        source_path="doc.pdf",
        elements=elements,
    )


def test_long_text_has_no_content_loss_and_respects_limits():
    tokenizer = Tokenizer()
    source = " ".join(f"token{i}" for i in range(1250))
    chunks = TypeAwareChunker(tokenizer).chunk(document([element("e1", source)]))
    assert len(chunks) == 3
    assert all(chunk.metadata["embedding_token_count"] <= 768 for chunk in chunks)
    # Strip the configured overlap from every later chunk and reconstruct source.
    rebuilt = []
    for index, chunk in enumerate(chunks):
        values = chunk.content.split()
        rebuilt.extend(values if index == 0 else values[50:])
    assert rebuilt == source.split()
    assert chunks[0].source_element_ids == ["e1"]


def test_overlap_never_crosses_heading_section_boundary():
    first = element("e1", " ".join(f"a{i}" for i in range(450)), heading_path=["A"])
    second = element(
        "e2", " ".join(f"b{i}" for i in range(450)), order=1, heading_path=["B"]
    )
    chunks = TypeAwareChunker(Tokenizer()).chunk(document([first, second]))
    b_chunk = next(chunk for chunk in chunks if chunk.heading_path == ["B"])
    assert b_chunk.content.split()[0] == "b0"


def test_long_table_repeats_headers_and_keeps_rows_whole():
    tokenizer = Tokenizer()
    header = "| Tên | Đơn vị | Giá trị |"
    separator = "| --- | --- | --- |"
    rows = [f"| Chỉ-tiêu-{i} | tỷ-đồng | {i} |" for i in range(300)]
    table = element(
        "table",
        "\n".join([header, separator, *rows]),
        kind="table",
        metadata={"sheet": "Sheet1", "range": "A1:C301"},
    )
    chunks = TypeAwareChunker(tokenizer).chunk(document([table]))
    assert len(chunks) > 1
    assert all(chunk.content.startswith(header + "\n" + separator) for chunk in chunks)
    assert all(chunk.sheet == "Sheet1" and chunk.cell_range == "A1:C301" for chunk in chunks)
    observed = [
        line
        for chunk in chunks
        for line in chunk.content.splitlines()[2:]
        if line.strip()
    ]
    assert observed == rows


def test_oversized_single_table_row_is_rejected_without_cell_cut():
    row = "| " + " ".join(f"value{i}" for i in range(800)) + " |"
    table = element("table", "| H |\n| --- |\n" + row, kind="table")
    with pytest.raises(ValueError, match="cannot be cut safely"):
        TypeAwareChunker(Tokenizer()).chunk(document([table]))


def test_figure_chunk_links_asset_and_preserves_uncertainty():
    figure = element(
        "figure",
        "Sơ đồ có hai nút.",
        kind="image",
        asset_path="assets/figure.png",
        image_hash="abc",
        generated_enrichment={"caption": "Sơ đồ có hai nút."},
        uncertainty_flags=["Nhãn nhỏ không đọc được"],
    )
    figure.text_for_embedding = "Mục\nSơ đồ có hai nút.\nVăn bản nhìn thấy: A; B"
    chunk = TypeAwareChunker(Tokenizer()).chunk(document([figure]))[0]
    assert chunk.type == "figure"
    assert chunk.image_path == "assets/figure.png"
    assert chunk.image_hash == "abc"
    assert chunk.source_element_ids == ["figure"]
    assert chunk.uncertainty_flags
    assert chunk.contextual_content


def test_embedding_input_over_hard_limit_never_silently_truncates():
    figure = element(
        "figure",
        "caption",
        kind="image",
        asset_path="x.png",
        generated_enrichment={"caption": "caption"},
    )
    figure.text_for_embedding = " ".join(f"x{i}" for i in range(769))
    with pytest.raises(ValueError, match="hard limit"):
        TypeAwareChunker(Tokenizer()).chunk(document([figure]))
