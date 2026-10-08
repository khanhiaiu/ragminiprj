import json

from rag.bm25 import BM25Index, tokenize_vi
from rag.chunk_contract import chunk_documents
from rag.embeddings import HashEmbedder, make_embedder
from rag.store import LocalVectorStore


def _doc():
    return {
        "document_id": "bieu-phi-2026",
        "filename": "bieu-phi.pdf",
        "elements": [
            {"element_type": "heading", "text": "Biểu phí thẻ tín dụng", "page_number": 12, "order": 0},
            {"element_type": "paragraph", "text": "Phí thường niên thẻ chuẩn là 300.000đ.", "page_number": 12, "order": 1},
            {
                "element_type": "table",
                "text": "| Hạng thẻ | Phí thường niên |\n|---|---|\n| Chuẩn | 300.000đ |",
                "page_number": 12,
                "order": 2,
            },
            {
                "element_type": "image",
                "text": "",
                "page_number": 13,
                "order": 3,
                "metadata": {"asset_path": "assets/image_001.png"},
            },
        ],
    }


def test_chunk_contract_types_and_hash():
    chunks = chunk_documents(_doc())
    assert [c.type for c in chunks] == ["text", "table", "figure"]
    assert chunks[1].section is not None  # table keeps parent section
    assert chunks[2].image_path == "assets/image_001.png"
    assert chunks[2].content.startswith("[FIGURE")
    hashes = [c.content_hash for c in chunks]
    assert len(set(hashes)) == len(hashes) and all(len(h) == 64 for h in hashes)
    assert all(c.chunk_id.startswith("bieu-phi-2026-c") for c in chunks)


def test_chunk_contract_accepts_step1_elements_json():
    elements_doc = {
        "doc_id": "d1",
        "elements": [
            {"element_type": "text", "text": "xin chào", "order": 0},
            {"element_type": "text", "text": "tạm biệt", "order": 1},
        ],
    }
    chunks = chunk_documents(elements_doc, max_chars=10000)
    assert len(chunks) == 1 and chunks[0].type == "text"


def test_hash_embedder_normalized_deterministic():
    embedder = HashEmbedder(dim=1024)
    vecs = embedder.embed_texts(["Phí thường niên thẻ chuẩn", "Phí thường niên thẻ chuẩn"])
    assert len(vecs[0]) == 1024
    assert vecs[0] == vecs[1]
    norm = sum(v * v for v in vecs[0]) ** 0.5
    assert abs(norm - 1.0) < 1e-6
    assert make_embedder("hash").dim == 1024


def test_bm25_vietnamese_ranking_and_persistence(tmp_path):
    index = BM25Index()
    index.add(["c1", "c2"], ["Phí thường niên thẻ chuẩn 300.000đ", "Sơ đồ luồng mở thẻ tín dụng"])
    assert "phí" in tokenize_vi("Phí Thường Niên")
    top = index.search("phí thường niên thẻ chuẩn", top_k=2)
    assert top[0][0] == "c1"
    path = tmp_path / "bm25.json"
    index.save(path)
    reloaded = BM25Index.load(path)
    assert reloaded.search("phí thường niên", top_k=1)[0][0] == "c1"


def test_local_store_dedup_search_and_persistence(tmp_path):
    store = LocalVectorStore()
    embedder = HashEmbedder()
    chunks = chunk_documents(_doc())
    texts = [c.content for c in chunks]
    vectors = embedder.embed_texts(texts)
    payloads = [
        {
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "file_name": c.file_name,
            "page": c.page,
            "section": c.section,
            "type": c.type,
            "image_path": c.image_path,
            "content": c.content,
            "content_hash": c.content_hash,
            "ingested_at": c.ingested_at,
        }
        for c in chunks
    ]
    assert store.upsert([c.chunk_id for c in chunks], vectors, payloads) == 3
    assert store.upsert([c.chunk_id for c in chunks], vectors, payloads) == 0  # re-ingest dedup
    assert store.existing_hashes() == {c.content_hash for c in chunks}
    query = embedder.embed_texts(["phí thường niên thẻ chuẩn 300.000đ"])[0]
    results = store.search_dense(query, top_k=3)
    assert results and results[0][2]["type"] in {"text", "table"}
    filtered = store.search_dense(query, top_k=3, filters={"type": "table"})
    assert all(p["type"] == "table" for _, _, p in filtered)
    store.save(tmp_path)
    reloaded = LocalVectorStore.load(tmp_path)
    assert len(reloaded) == 3
    assert (tmp_path / "payloads.jsonl").exists()


def test_ingest_script_offline(tmp_path):
    import subprocess
    import sys

    input_dir = tmp_path / "parsed"
    (input_dir / "doc1").mkdir(parents=True)
    (input_dir / "doc1" / "document.json").write_text(
        json.dumps(
            {
                "document_id": "doc1",
                "filename": "a.pdf",
                "file_type": "pdf",
                "source_path": "a.pdf",
                "pages": [{"page_number": 1}],
                "elements": [
                    {
                        "element_id": "doc1-e1",
                        "element_type": "paragraph",
                        "text": "hello world",
                        "page_number": 1,
                        "order": 0,
                        "metadata": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "rag"
    proc = subprocess.run(
        [sys.executable, "scripts/ingest.py", "--input", str(input_dir), "--output", str(out), "--embedder", "hash",
         "--backend", "local"],
        capture_output=True,
        text=True,
        cwd=str(tmp_path / ".." / ".." if False else "/home/nghia/ai/tp/TPragsystem"),
    )
    assert proc.returncode == 0, proc.stderr
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["chunks_inserted"] == 1
    # Second run is idempotent.
    proc2 = subprocess.run(
        [sys.executable, "scripts/ingest.py", "--input", str(input_dir), "--output", str(out), "--embedder", "hash",
         "--backend", "local"],
        capture_output=True,
        text=True,
        cwd="/home/nghia/ai/tp/TPragsystem",
    )
    assert proc2.returncode == 0, proc2.stderr
    manifest2 = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest2["chunks_inserted"] == 0
