"""Check that the shared YAML controls actual pipeline and service behavior."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

import project_settings as settings

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "_active_path", None)
    monkeypatch.delenv("RAG_RELEVANCE_THRESHOLD", raising=False)
    path = tmp_path / "config.yaml"
    path.write_text("{}\n")
    monkeypatch.setenv("PROJECT_CONFIG", str(path))
    return path


def write_config(path, data):
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def script(name):
    spec = importlib.util.spec_from_file_location(f"settings_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parser_and_rag_share_config_outside_repository(config_file, tmp_path, monkeypatch):
    from document_parser.config import ParserConfig, ProjectConfig
    from rag.chunking.chunker import ChunkingConfig
    from rag.embeddings import make_embedder
    from rag.llm import make_llm
    from rag.memory import SessionMemory
    from rag import guardrails

    write_config(config_file, {
        "device": "cpu", "paths": {"input": "documents"},
        "chunking": {"target_tokens": 420},
        "embedding": {"batch_size": 2, "device": "cuda:1"},
        "llm": {"model": "test-model", "base_url": "http://model:8080/v1", "timeout_seconds": 23},
        "memory": {"max_turns": 2}, "guardrails": {"max_question_chars": 5},
    })
    monkeypatch.chdir(tmp_path.parent)
    project = ProjectConfig.load()
    assert project.parser_config().ocr_options["device"] == "cpu"
    assert ParserConfig().ocr_options["device"] == "cpu"
    assert project.path("input") == config_file.parent / "documents"
    assert ChunkingConfig().target_tokens == 420
    embedder = make_embedder("bge-m3")  # no model download until an embedding is requested
    assert (embedder.batch_size, embedder.device) == (2, "cuda:1")
    llm = make_llm()
    assert (llm.model, llm.base_url, llm.timeout) == ("test-model", "http://model:8080/v1", 23)
    assert SessionMemory().max_turns == 2
    assert not guardrails.check_input("too long")["ok"]


def test_custom_qdrant_vectors_alias_retrieval_and_generation(config_file):
    from qdrant_client import QdrantClient
    from rag.answer import answer_question
    from rag.embeddings import HashEmbedder
    from rag.indexing import GenerationIndexer
    from rag.llm import FakeLLM
    from rag.retrieve import HybridRetriever
    from rag.store import QdrantStore

    write_config(config_file, {
        "embedding": {"backend": "hash", "dense_dimensions": 16},
        "qdrant": {"alias": "custom_current", "dense_vector_name": "semantic", "sparse_vector_name": "lexical"},
        "retrieval": {"dense_top": 2, "sparse_top": 3, "top_k": 1, "relevance_threshold": 0.5},
        "llm": {"temperature": 0.25, "max_tokens": 32},
    })
    client = QdrantClient(":memory:")
    try:
        store = QdrantStore(client=client, collection="generation_custom")
        store.ensure_collection()
        embedder = HashEmbedder()
        embedding = embedder.embed_hybrid(["annual fee"])
        GenerationIndexer(store).upsert(
            ["fee"], embedding, [{"content": "annual fee", "file_name": "fees.pdf", "page": 1}],
        )
        store.publish_alias()
        assert "custom_current" in {a.alias_name for a in client.get_aliases().aliases}
        retriever = HybridRetriever(store=QdrantStore(client=client, collection="custom_current"), embedder=embedder)
        calls = []

        class RecordingLLM(FakeLLM):
            def complete(self, messages, **kwargs):
                calls.append(kwargs)
                return "annual fee"

        result = answer_question(retriever, "annual fee", RecordingLLM())
        assert not result["fallback"]
        assert len(result["chunks"]) == 1
        assert calls == [{"temperature": 0.25, "max_tokens": 32}]
    finally:
        client.close()


def test_cli_config_env_and_argument_precedence(config_file, monkeypatch, capsys):
    write_config(config_file, {"embedding": {"backend": "hash"}, "qdrant": {"alias": "yaml_alias"}, "retrieval": {"top_k": 2}})
    module = script("query")
    selected = {}

    class Retriever:
        @classmethod
        def load(cls, *args, **kwargs):
            selected.update(kwargs)
            selected["url"] = settings.setting("qdrant.url")
            return cls()

        def retrieve(self, query, **kwargs):
            selected.update(kwargs)
            return {"query": query, "chunks": [], "fallback": True}

    monkeypatch.setattr(module, "HybridRetriever", Retriever)
    monkeypatch.setenv("QDRANT_COLLECTION", "env_alias")
    override = config_file.parent / "override.yaml"
    write_config(override, {"qdrant": {"url": "http://configured:6333"}})
    monkeypatch.setattr(sys, "argv", ["query.py", "--config", str(override), "--query", "fee", "--collection", "cli_alias", "--top-k", "1", "--as-json"])
    assert module.main() == 0
    assert selected["collection"] == "cli_alias"
    assert selected["top_k"] == 1
    assert selected["url"] == "http://configured:6333"
    assert json.loads(capsys.readouterr().out)["fallback"]
    assert settings.setting("qdrant.alias", "QDRANT_COLLECTION") == "env_alias"


def test_api_ui_and_caption_clients_use_same_file(config_file, monkeypatch):
    import api.main as api
    from rag.enrichment.vlm import GeminiVLMClient

    write_config(config_file, {
        "llm": {"base_url": "http://configured-llm/v1"},
        "ui": {"api_url": "http://configured-api:9999"},
        "captioning": {"model": "configured-vision", "timeout_seconds": 17.0, "maximum_retries": 1},
        "retrieval": {"top_k": 3, "rerank": True},
    })
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    assert api._settings()["llm_base_url"] == "http://configured-llm/v1"
    assert settings.setting("ui.api_url", "RAG_API_URL") == "http://configured-api:9999"
    assert api.QueryRequest(question="fee").top_k == 3
    assert api.QueryRequest(question="fee").rerank
    captioner = GeminiVLMClient()
    assert (captioner.model, captioner.timeout_seconds, captioner.max_retries) == ("configured-vision", 17.0, 1)


def test_api_launcher_uses_config_host_port(config_file, monkeypatch):
    import uvicorn

    write_config(config_file, {"api": {"host": "127.0.0.2", "port": 9999}})
    selected = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: selected.update(kwargs))
    monkeypatch.setattr(sys, "argv", ["serve_api.py"])
    assert script("serve_api").main() == 0
    assert selected == {"host": "127.0.0.2", "port": 9999}


@pytest.mark.parametrize("data,reason", [
    ({"retrival": {}}, "Unknown config"),
    ({"retrieval": {"dense_top": 0}}, "finite and positive"),
    ({"retrieval": {"relevance_threshold": float("nan")}}, "finite and non-negative"),
    ({"llm": []}, "mapping"),
    ({"embedding": {"batch_size": "four"}}, "Invalid type"),
    ({"tokenizer": {"revision": "other"}}, "same model and revision"),
])
def test_invalid_shared_settings_fail_early(config_file, data, reason):
    write_config(config_file, data)
    with pytest.raises(ValueError, match=reason):
        settings.load_config()


def test_default_yaml_is_only_runtime_config_source():
    assert settings.DEFAULT_CONFIG == ROOT / "config.yaml"
    assert not (ROOT / "src/document_parser/config.yaml").exists()
    assert not list((ROOT / "config").glob("rag_*.json"))


def test_custom_config_can_disable_a_calibrated_threshold(config_file):
    from rag.retrieve import HybridRetriever
    from rag.embeddings import HashEmbedder

    write_config(config_file, {"retrieval": {"relevance_threshold": None}})
    retriever = HybridRetriever(store=object(), embedder=HashEmbedder())
    assert retriever.threshold is None
