import math

import pytest

from rag.embeddings import BgeM3Embedder, EmbeddingCache, HashEmbedder


class FakeBgeM3:
    def __init__(self, *, bad_dim=False, bad_sparse=False):
        self.calls = []
        self.bad_dim = bad_dim
        self.bad_sparse = bad_sparse

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        dim = 3 if self.bad_dim else 1024
        sparse = [{1: float("nan")}] if self.bad_sparse else [{1: 0.7, 9: 0.2}]
        return {
            "dense_vecs": [[0.0] * dim for _ in texts],
            "lexical_weights": sparse * len(texts),
        }


def test_bge_adapter_returns_dense_and_sparse_with_colbert_disabled():
    model = FakeBgeM3()
    embedder = BgeM3Embedder(model=model, batch_size=4)
    result = embedder.embed_hybrid(["xin chào"])[0]
    assert len(result.dense) == 1024
    assert result.sparse_indices == [1, 9]
    assert result.sparse_values == [0.7, 0.2]
    options = model.calls[0][1]
    assert options["return_dense"] and options["return_sparse"]
    assert options["return_colbert_vecs"] is False
    assert options["max_length"] == 768


def test_embedding_cache_avoids_duplicate_model_work_and_config_invalidates(tmp_path):
    model = FakeBgeM3()
    cache = EmbeddingCache(tmp_path)
    first = BgeM3Embedder(model=model, cache=cache, max_length=768)
    a = first.embed_hybrid(["same text"])[0]
    b = first.embed_hybrid(["same text"])[0]
    assert a == b
    assert len(model.calls) == 1
    changed = BgeM3Embedder(model=model, cache=cache, max_length=512)
    changed.embed_hybrid(["same text"])
    assert len(model.calls) == 2


@pytest.mark.parametrize("model,match", [(FakeBgeM3(bad_dim=True), "dimension"), (FakeBgeM3(bad_sparse=True), "sparse")])
def test_invalid_model_outputs_are_rejected(model, match):
    with pytest.raises(ValueError, match=match):
        BgeM3Embedder(model=model).embed_hybrid(["x"])


def test_streaming_batches_start_at_four():
    model = FakeBgeM3()
    embedder = BgeM3Embedder(model=model, batch_size=4)
    batches = list(embedder.stream((str(i) for i in range(9))))
    assert [len(batch) for batch in batches] == [4, 4, 1]
    assert [len(call[0]) for call in model.calls] == [4, 4, 1]


def test_hash_fake_also_exposes_valid_sparse_vectors():
    result = HashEmbedder().embed_hybrid(["a a b"])[0]
    assert len(result.dense) == 1024
    assert len(result.sparse_indices) == len(result.sparse_values) == 2
    assert all(math.isfinite(value) and value > 0 for value in result.sparse_values)


def test_embedder_rejects_input_that_would_be_silently_truncated():
    class Tokenizer:
        def encode(self, text, *, add_special_tokens=False):
            return list(range(len(text.split()) + (2 if add_special_tokens else 0)))

    model = FakeBgeM3()
    model.tokenizer = Tokenizer()
    embedder = BgeM3Embedder(model=model, max_length=5)
    embedder.embed_hybrid(["a b c"])
    assert len(model.calls) == 1
    with pytest.raises(ValueError, match="refusing silent truncation"):
        embedder.embed_hybrid(["a b c d"])
    assert len(model.calls) == 1
