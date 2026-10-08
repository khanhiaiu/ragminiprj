import pytest

pytest.importorskip("qdrant_client")

from rag.embeddings import HashEmbedder
from rag.indexing import GenerationIndexer
from rag.store import QdrantStore


def test_real_in_memory_qdrant_dense_sparse_hybrid_filters_and_publish():
    store = QdrantStore(in_memory=True, collection="generation_test")
    store.ensure_collection()
    embeddings = HashEmbedder().embed_hybrid(
        ["quyền công dân", "phí thường niên 300000 đồng"]
    )
    payloads = [
        {"doc_id": "law", "document_version": "1", "type": "text", "content_hash": "a"},
        {"doc_id": "fees", "document_version": "1", "type": "table", "content_hash": "b"},
    ]
    indexer = GenerationIndexer(store)
    assert indexer.upsert(["law-1", "fees-1"], embeddings, payloads) == 2
    # Deterministic point IDs make repeat ingest an update, never a duplicate.
    assert indexer.upsert(["law-1", "fees-1"], embeddings, payloads) == 2
    assert store.count() == 2

    query = HashEmbedder().embed_hybrid(["phí thường niên"])[0]
    dense = store.search_dense(query.dense, filters={"type": "table"})
    sparse = store.search_sparse(
        query.sparse_indices, query.sparse_values, filters={"doc_id": "fees"}
    )
    hybrid = store.search_hybrid(query.dense, query.sparse_indices, query.sparse_values)
    assert dense and all(item[2]["type"] == "table" for item in dense)
    assert sparse and all(item[2]["doc_id"] == "fees" for item in sparse)
    assert hybrid and hybrid[0][0] == "fees-1"

    verification = indexer.verify(2, query)
    assert verification.passed
    indexer.publish(verification)
    assert any(alias.alias_name == "docs_current" for alias in store.client.get_aliases().aliases)
