import json
import socket
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from blended_rag.answers import answer, synthesized
from blended_rag.api import create_app
from blended_rag.ingestion import LocalStore
from blended_rag.models import Hit, digest
from blended_rag.providers import GeminiEncoder, OnnxEncoder, ProviderError
from blended_rag.retrieval import FaissRetriever
from blended_rag.evaluation import evaluate
from main import load_env, main


@pytest.fixture
def local(tmp_path, chunks, monkeypatch):
    # Deterministic unit-test vectors; a separate smoke test uses the real bundled model.
    def encode(self, texts):
        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for i, text in enumerate(texts):
            vectors[i, 0] = 1
            vectors[i, 1] = .5 if "account" in text.lower() else 1
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    monkeypatch.setattr(OnnxEncoder, "encode", encode)
    store = LocalStore(tmp_path / "local")
    store.ingest(chunks, "onnx")
    return store


def test_faiss_matches_exact_and_filters_before_topk(local):
    retriever = FaissRetriever(local)
    query = retriever.encoder.encode(["account"])[0]
    expected = sorted(retriever.vectors @ query, reverse=True)
    assert [h.score for h in retriever.search("account", "dense", 3)] == pytest.approx(expected)
    result = retriever.search("account", "dense", 1, {"control_id": "cp-9"})
    assert len(result) == 1 and result[0].chunk.control_id == "cp-9"
    assert retriever.search("account", "dense", 1, {"release": "rev4"}) == []


def test_faiss_artifact_corruption_fails(local):
    metadata = local.read()["faiss"]
    (local.directory / metadata["file"]).write_bytes(b"broken")
    with pytest.raises(ValueError, match="hash mismatch"):
        FaissRetriever(local)


def test_cache_reuse_and_source_change(local, tmp_path, chunks):
    path = tmp_path / "cache.npz"
    local.export_cache(path)
    new = LocalStore(tmp_path / "copy")
    result = new.ingest(chunks, "onnx", path)
    assert result["embedded_chunks"] == 0 and result["cached_chunks"] == len(chunks)
    changed = [replace(chunks[0], source_hash=digest("different source")), *chunks[1:]]
    # A fresh corpus with a mismatching validated source cache must be encoded.
    result = LocalStore(tmp_path / "changed").ingest(changed, "onnx", path)
    assert result["embedded_chunks"] == len(chunks)
    with np.load(path) as data:
        metadata = json.loads(str(data["metadata"]))
        vectors = data["vectors"]
    metadata["model_fingerprint"] = "wrong"
    np.savez_compressed(path, metadata=json.dumps(metadata), vectors=vectors)
    with pytest.raises(ValueError, match="cache model mismatch"):
        LocalStore(tmp_path / "wrong").ingest(chunks, "onnx", path)


def test_local_nlp_and_api_identity(local):
    retriever = FaissRetriever(local)
    result = answer(retriever, "How are accounts managed?")
    assert result["mode"] == "local" and result["vector_engine"] == "faiss-IndexFlatIP"
    assert result["citations"] and '"' in result["answer"]
    assert "Insufficient evidence" in answer(retriever, "galactic banana spacecraft")['answer']
    for citation in result["citations"]:
        assert f'[{citation["citation_id"]}]' in result["answer"]
    with TestClient(create_app(local.directory, mode="local")) as client:
        assert client.get("/health").json()["mode"] == "local"
        assert client.post("/query", json={"question": "account"}).json()["answer_engine"] == "extractive-nlp"


def test_gemini_task_payloads_and_usage():
    captured = []
    encoder = GeminiEncoder(dimensions=3)
    def embed(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[1., 2., 3.])],
                               metadata=SimpleNamespace(billable_character_count=7))
    encoder._client = SimpleNamespace(models=SimpleNamespace(embed_content=embed))
    encoder.encode(["document"])
    encoder.encode_query("question")
    assert [x["config"]["task_type"] for x in captured] == ["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]
    assert all(x["config"]["output_dimensionality"] == 3 for x in captured)
    assert encoder.usage["embedding_requests"] == 2
    assert encoder.usage["embedding_billable_characters"] == 14


def test_gemini_batch_boundaries_and_order():
    encoder = GeminiEncoder(dimensions=3)
    captured = []
    def embed(**kwargs):
        batch = kwargs["contents"]
        captured.append(batch)
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[float(text), 1., 0.]) for text in batch],
                               metadata=SimpleNamespace(billable_character_count=sum(map(len, batch))))
    encoder._client = SimpleNamespace(models=SimpleNamespace(embed_content=embed))
    vectors = encoder.encode([str(i + 1) for i in range(70)])
    assert sorted(map(len, captured)) == [6, 32, 32]
    assert encoder.usage["embedding_requests"] == 3
    assert vectors[:, 0] / vectors[:, 1] == pytest.approx(np.arange(1, 71))


def test_explicit_env_file_json_windows_path(tmp_path, monkeypatch):
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    path = tmp_path / "private.env"
    value = "C:\\private\\service-account.json"
    path.write_text("GOOGLE_APPLICATION_CREDENTIALS=" + json.dumps(value), encoding="utf-8")
    load_env(path)
    import os
    assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == value


def test_cli_explicit_local_identity(local, capsys):
    assert main(["--mode", "local", "--store", str(local.directory), "query", "account"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "local" and payload["answer_engine"] == "extractive-nlp"


def test_evaluation_includes_warmup_usage_and_unpriced_cloud(store, tmp_path):
    class Measured:
        mode = "enterprise"
        model = {"name": "test"}
        encoder = SimpleNamespace(usage={"embedding_requests": 0})
        def search(self, query, method, k, filters):
            if method in {"dense", "hybrid"}:
                self.encoder.usage["embedding_requests"] += 1
            return []
    queries = tmp_path / "q.json"
    queries.write_text(json.dumps({"label_provenance": "test", "queries": [
        {"id": "1", "query": "account", "filters": {}, "relevant": {"rev5:ac-2": 1}}]}))
    output = tmp_path / "eval.json"
    evaluate(Measured(), store, output, queries)
    report = json.loads(output.read_text())
    assert report["usage"]["embedding_requests"] == 3
    assert report["warmup_usage"]["embedding_requests"] == 1
    assert report["timed_usage"]["embedding_requests"] == 2
    assert report["resources"]["external_api_cost_usd"] is None


@pytest.mark.parametrize("payload", [
    {"answer": "Unsupported [99]", "citation_ids": [99], "insufficient": False},
    {"answer": "Missing inline citation", "citation_ids": [1], "insufficient": False},
    {"answer": "Wrong inline [2]", "citation_ids": [1], "insufficient": False},
])
def test_gemini_invalid_citations_fail(chunks, payload):
    encoder = GeminiEncoder()
    encoder._client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kwargs:
        SimpleNamespace(text=json.dumps(payload), usage_metadata=SimpleNamespace(prompt_token_count=12, candidates_token_count=4))))
    retriever = SimpleNamespace(encoder=encoder)
    with pytest.raises(ProviderError):
        synthesized(retriever, "account", [Hit(chunks[0], 1)])
    assert encoder.usage["input_tokens"] == 12


def test_real_bundled_encoder_needs_no_network_or_keys(monkeypatch):
    for key in ["GEMINI_API_KEY", "GOOGLE_API_KEY", "MONGODB_URI"]:
        monkeypatch.delenv(key, raising=False)
    def forbidden(*args, **kwargs):
        raise AssertionError("Local mode attempted network")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    encoder = OnnxEncoder()
    vectors = encoder.encode(["Manage user accounts", "Back up system information"])
    assert vectors.shape == (2, 384)
    assert np.linalg.norm(vectors, axis=1) == pytest.approx([1., 1.], abs=1e-6)
    assert not np.allclose(vectors[0], vectors[1])
