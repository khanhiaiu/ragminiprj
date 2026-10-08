from rag.embeddings import HashEmbedder
from rag.indexing import GenerationIndexer


class FakeStore:
    def __init__(self, count=1):
        self._count = count
        self.published = False
        self.points = {}

    def upsert_hybrid(self, ids, dense, indices, values, payloads):
        for fields in zip(ids, dense, indices, values, payloads):
            self.points[fields[0]] = fields[1:]
        self._count = len(self.points)
        return len(ids)

    def count(self):
        return self._count

    def search_dense(self, *args, **kwargs):
        return [("id", 1.0, {})] if self._count else []

    def search_sparse(self, *args, **kwargs):
        return [("id", 1.0, {})] if self._count else []

    def search_hybrid(self, *args, **kwargs):
        return [("id", 1.0, {})] if self._count else []

    def publish_alias(self):
        self.published = True


def test_repeat_ingest_uses_same_chunk_ids_without_duplicate_points():
    store = FakeStore(0)
    indexer = GenerationIndexer(store)
    embedding = HashEmbedder().embed_hybrid(["hello"])
    indexer.upsert(["stable"], embedding, [{"type": "text"}])
    indexer.upsert(["stable"], embedding, [{"type": "text"}])
    assert store.count() == 1


def test_verification_requires_dense_sparse_and_hybrid_before_publish():
    store = FakeStore(1)
    indexer = GenerationIndexer(store)
    query = HashEmbedder().embed_hybrid(["hello"])[0]
    result = indexer.verify(1, query, filters={"type": "text"})
    assert result.passed
    indexer.publish(result)
    assert store.published


def test_failed_verification_never_publishes():
    store = FakeStore(1)
    indexer = GenerationIndexer(store)
    query = HashEmbedder().embed_hybrid(["hello"])[0]
    result = indexer.verify(2, query)
    assert not result.passed
    try:
        indexer.publish(result)
    except RuntimeError:
        pass
    assert not store.published
