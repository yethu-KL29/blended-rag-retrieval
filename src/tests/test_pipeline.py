import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from blended_rag.answers import answer, compare
from blended_rag.api import create_app
from blended_rag.dataset import load_sources
from blended_rag.evaluation import metrics
from blended_rag.ingestion import Encoder, NeuralEncoder, LocalStore, parse_catalog
from blended_rag.models import Hit, digest, validate_filters
from blended_rag.retrieval import LocalRetriever, rrf


def test_structured_parsing(chunks, source):
    assert {c.control_id for c in chunks} == {"ac-2", "cp-9"}
    assert any(c.section_id == "ac-2_smt.a:0" for c in chunks)
    assert all(c.release == "rev5" and c.source_hash == source["sha256"] for c in chunks)
    assert all(c.content_hash == digest(c.text) for c in chunks)


def test_chunk_limit_and_stability(source, document):
    document["catalog"]["groups"][0]["controls"][0]["parts"][0]["prose"] = "word " * 55
    first = parse_catalog(source, document, max_words=20)
    assert all(len(c.text.split()) <= 20 for c in first)
    assert first == parse_catalog(source, document, max_words=20)


def test_parameters_and_unnamed_nested_sections(source, document):
    control = document["catalog"]["groups"][0]["controls"][0]
    control["params"] = [{"id": "ac-2_prm_1", "label": "account review frequency"}]
    control["parts"] += [{"name": "assessment", "parts": [{"name": "objects", "prose": "First evidence"}]},
                         {"name": "assessment", "parts": [{"name": "objects", "prose": "Second evidence"}]}]
    chunks = parse_catalog(source, document)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert any("account review frequency" in c.text and c.section_id.startswith("parameter:") for c in chunks)


def test_real_withdrawn_control_retains_metadata():
    evidence = []
    for source, document in load_sources():
        evidence.extend(c for c in parse_catalog(source, document) if c.control_id == "ac-2.10")
    result = compare(evidence, "ac-2.10")
    assert all(result["releases"][release]["present"] for release in ["rev4", "rev5"])
    withdrawn = result["releases"]["rev5"]["excerpts"]
    assert any('"withdrawn"' in c["text"] and '"incorporated-into"' in c["text"] and
               '"#ac-2_smt.k"' in c["text"] for c in withdrawn)
    assert all(c["source_url"].startswith("https://") and c["source_hash"] and c["release"] == "rev5"
               for c in withdrawn)


def test_idempotence_does_not_embed(store, chunks, monkeypatch):
    def forbidden(*args):
        raise AssertionError("Repeated embedding expense")
    monkeypatch.setattr(Encoder, "fit", forbidden)
    result = store.ingest(chunks)
    assert result["changed"] is False and result["embedded_chunks"] == 0


def test_release_preservation_and_replacement(store, chunks):
    old = [replace(c, release="rev4", chunk_id=digest(c.chunk_id + "old")) for c in chunks]
    store.ingest(old)
    changed = replace(chunks[0], text="Changed account text.", content_hash=digest("Changed account text."))
    store.ingest([changed])
    actual = store.read()["chunks"]
    assert sum(c["release"] == "rev4" for c in actual) == len(old)
    assert [c["text"] for c in actual if c["release"] == "rev5"] == [changed.text]


def test_failed_preparation_preserves_store(store, chunks, monkeypatch):
    before = store.path.read_bytes()
    def fail(*args):
        raise ValueError("provider failure")
    monkeypatch.setattr(Encoder, "encode", fail)
    changed = replace(chunks[0], text="New text.", content_hash=digest("New text."))
    with pytest.raises(ValueError, match="provider failure"):
        store.ingest([changed])
    assert store.path.read_bytes() == before


def test_model_mismatch_fails_closed(store, chunks):
    state = store.read()
    state["model_name"] = "other-model"
    store.path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="model mismatch"):
        store.ingest(chunks)


def test_fixed_neural_model_reuses_unchanged_vectors(tmp_path, chunks, monkeypatch):
    calls = []
    def encode(self, texts):
        calls.append(len(texts))
        return np.ones((len(texts), self.dimensions)) / np.sqrt(self.dimensions)
    monkeypatch.setattr(NeuralEncoder, "encode", encode)
    neural_store = LocalStore(tmp_path / "neural")
    neural_store.ingest(chunks, "minilm")
    changed = [replace(chunks[0], text="Changed text.", content_hash=digest("Changed text.")), *chunks[1:]]
    result = neural_store.ingest(changed, "minilm")
    assert calls == [len(chunks), 1]
    assert result["embedded_chunks"] == 1


@pytest.mark.parametrize("changed", [False, True])
def test_neural_revision_mismatch_fails_before_cache_reuse(tmp_path, chunks, monkeypatch, changed):
    monkeypatch.setattr(NeuralEncoder, "encode", lambda self, texts: np.ones((len(texts), self.dimensions)))
    neural_store = LocalStore(tmp_path / "neural")
    neural_store.ingest(chunks, "minilm")
    state = neural_store.read()
    state["model_fingerprint"] = "previous-model-revision"
    neural_store.path.write_text(json.dumps(state))
    before = neural_store.path.read_bytes()
    def forbidden(*args):
        raise AssertionError("Mismatched model must fail before embedding")
    monkeypatch.setattr(NeuralEncoder, "encode", forbidden)
    incoming = [replace(chunks[0], text="Changed text.", content_hash=digest("Changed text.")), *chunks[1:]] if changed else chunks
    with pytest.raises(ValueError, match="model mismatch"):
        neural_store.ingest(incoming, "minilm")
    assert neural_store.path.read_bytes() == before


@pytest.mark.parametrize("filters", [{"unknown": "x"}, {"release": "latest"}, {"control_id": ""}, {"release": 5}])
def test_invalid_filters(filters):
    with pytest.raises(ValueError):
        validate_filters(filters)


@pytest.mark.parametrize("method", ["lexical", "dense", "hybrid"])
def test_filtered_retrieval(store, method):
    retriever = LocalRetriever(store)
    hits = retriever.search("account management", method, 5, {"control_id": "ac-2", "release": "rev5"})
    assert hits and all(h.chunk.control_id == "ac-2" for h in hits)
    assert retriever.search("account", method, 5, {"release": "rev4"}) == []


def test_rrf_promotes_agreement(chunks):
    a, b, c = [Hit(chunk, 1) for chunk in chunks]
    result = rrf([[a, b], [c, b]], 3)
    assert result[0].chunk.chunk_id == b.chunk.chunk_id
    assert result[0].score == pytest.approx(2 / 62)


def test_known_metrics():
    actual = metrics(["x", "a", "b"], {"a": 2, "b": 1})
    assert actual["recall@5"] == 1
    assert actual["mrr"] == .5
    assert actual["precision@5"] == .4
    assert actual["ndcg@10"] == pytest.approx((3 / math.log2(3) + 1 / math.log2(4)) / (3 + 1 / math.log2(3)))
    assert all(v == 0 for v in metrics([], {}).values())


def test_no_evidence_and_invalid_query(store):
    retriever = LocalRetriever(store)
    result = answer(retriever, "account", filters={"control_id": "absent"})
    assert result["citations"] == [] and "Insufficient evidence" in result["answer"]
    with pytest.raises(ValueError):
        retriever.search("   ")


def test_comparison_absent_release(chunks):
    result = compare(chunks, "ac-2")
    assert not result["releases"]["rev4"]["present"]
    assert result["releases"]["rev5"]["present"]
    assert "+" in result["textual_diff"]


def test_comparison_pairs_changed_control_evidence(chunks):
    current = chunks[0]
    old_text = "Previous account review requirement."
    previous = replace(current, release="rev4", chunk_id=digest(current.chunk_id + "rev4"),
                       source_url="https://example.org/rev4.json", source_hash=digest("old source"),
                       text=old_text, content_hash=digest(old_text))
    result = compare([previous, current], current.control_id)
    for release, expected in [("rev4", previous), ("rev5", current)]:
        side = result["releases"][release]
        assert side["present"]
        assert side["excerpts"][0]["release"] == release
        assert side["excerpts"][0]["source_url"] == expected.source_url
        assert side["excerpts"][0]["section_id"] == expected.section_id
        assert side["excerpts"][0]["content_hash"] == expected.content_hash
    assert "-" + old_text in result["textual_diff"]
    assert "+" + current.text in result["textual_diff"]


def test_api(store):
    with TestClient(create_app(store.directory)) as client:
        assert client.get("/health").status_code == 200
        result = client.post("/query", json={"question": "account", "filters": {"release": "rev5"}})
        assert result.status_code == 200 and result.json()["citations"]
        assert client.post("/query", json={"question": "account", "filters": {"secret": "x"}}).status_code == 422
        assert client.post("/query", json={"question": " "}).status_code == 422
        assert client.get("/compare/ac-2").status_code == 200


def test_cli_and_missing_atlas_config(store):
    root = Path(__file__).resolve().parents[2]
    command = [sys.executable, str(root / "src/main.py"), "--store", str(store.directory)]
    run = subprocess.run(command + ["query", "account", "--release", "rev5"], capture_output=True, text=True)
    assert run.returncode == 0 and json.loads(run.stdout)["citations"]
    env = {k: v for k, v in os.environ.items() if k != "MONGODB_URI"}
    run = subprocess.run(command + ["--backend", "atlas", "query", "account"], env=env, capture_output=True, text=True)
    assert run.returncode == 2 and "MONGODB_URI" in run.stderr


def test_bundled_corpus_hashes_and_qrels():
    controls = set()
    for source, doc in load_sources():
        chunks = parse_catalog(source, doc)
        assert len(chunks) > 1000
        controls.update(f"{c.release}:{c.control_id}" for c in chunks)
    queries = json.loads((Path(__file__).resolve().parents[1] / "data/queries.json").read_text(encoding="utf-8"))
    assert all(set(q["relevant"]) <= controls for q in queries["queries"])
