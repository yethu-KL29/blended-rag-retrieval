import json
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from blended_rag.api import create_app
from blended_rag.atlas import AtlasRetriever
from blended_rag.evaluation import ablate
from blended_rag.ingestion import LocalStore
from blended_rag.models import Hit, digest, tokens
from blended_rag.retrieval import LocalRetriever, RetrievalConfig, FIELD_BOOSTS, PROFILES, identifiers, rrf
from main import main


@pytest.fixture
def fields_store(tmp_path, chunks):
    records = [replace(chunks[0], chunk_id="title", control_id="cp-9.1", title="unicorn", section_id="appendixz", text="plain body", content_hash=digest("plain body")),
               replace(chunks[1], chunk_id="body", control_id="cp-9", title="ordinary", section_id="other", text="unicorn " * 30, content_hash=digest("unicorn " * 30))]
    store = LocalStore(tmp_path / "fields")
    store.ingest(records)
    return store


@pytest.mark.parametrize("kwargs", [dict(lexical_profile="bad"), dict(query_formulation="bad"), dict(lexical_weight=-1),
                                      dict(dense_weight=float("nan")), dict(lexical_weight=float("inf")),
                                      dict(lexical_weight=0, dense_weight=0), dict(lexical_weight=True)])
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        RetrievalConfig(**kwargs)


def test_immutable_config():
    with pytest.raises(FrozenInstanceError):
        RetrievalConfig().dense_weight = 2


@pytest.mark.parametrize("profile", ["multi-field", "boosted-sum", "best-fields"])
def test_field_only_matches_and_filters(fields_store, profile):
    retriever = LocalRetriever(fields_store, RetrievalConfig(profile))
    for query in ["appendixz", "CP-9.1"]:
        assert retriever.search(query, "lexical")[0].chunk.chunk_id == "title"
        assert not LocalRetriever(fields_store, RetrievalConfig("content-only")).search(query, "lexical")
        assert not retriever.search(query, "lexical", filters={"release": "rev4"})
        assert not retriever.search(query, "hybrid", filters={"control_id": "absent"})
    assert "unicorn" not in retriever.field_stats["section_id"][3]
    assert "appendixz" not in retriever.field_stats["text"][3]


def test_title_boost_reorders_and_max_is_max(fields_store):
    uniform = LocalRetriever(fields_store, RetrievalConfig("multi-field"))
    boosted = LocalRetriever(fields_store, RetrievalConfig("boosted-sum"))
    assert uniform.search("unicorn", "lexical")[0].chunk.chunk_id == "body"
    assert boosted.search("unicorn", "lexical")[0].chunk.chunk_id == "title"
    maximum = LocalRetriever(fields_store, RetrievalConfig("best-fields"))
    query = "unicorn appendixz"
    hits = maximum.search(query, "lexical")
    expected = np.maximum.reduce([weight * maximum._bm25(identifiers(query) if field == "control_id" else tokens(query), [0, 1], maximum.field_stats[field]) for field, weight in FIELD_BOOSTS.items()])
    assert {h.chunk.chunk_id: h.score for h in hits} == pytest.approx({c.chunk_id: expected[i] for i, c in enumerate(maximum.chunks)})


def test_exact_enhancement_case_and_identifier_formulation(fields_store):
    assert identifiers("Use CP-9.1, cp-9, and CP-9.11") == ["cp-9.1", "cp-9", "cp-9.11"]
    for profile in PROFILES:
        retriever = LocalRetriever(fields_store, RetrievalConfig(profile, "identifier"))
        assert {h.chunk.control_id for h in retriever.search("CP-9.1 plain", "lexical")} == {"cp-9.1"}
        assert {h.chunk.control_id for h in retriever.search("CP-9 unicorn", "lexical")} == {"cp-9"}


def test_default_combined_regression(store):
    retriever = LocalRetriever(store)
    scores = np.zeros(len(retriever.chunks))
    for term in set(tokens("account management")):
        for i, counts in enumerate(retriever.counts):
            freq = counts.get(term, 0)
            if freq:
                scores[i] += retriever.idf.get(term, 0) * freq * 2.5 / (freq + 1.5 * (.25 + .75 * retriever.lengths[i] / retriever.average_length))
    expected = sorted((Hit(c, float(scores[i])) for i, c in enumerate(retriever.chunks) if scores[i] > 1e-8), key=lambda h: (-h.score, h.chunk.chunk_id))
    assert retriever.search("account management", "lexical") == expected
    assert retriever.search("account", "hybrid") == rrf([retriever.search("account", "lexical", 50), retriever.search("account", "dense", 50)], 10)


def test_weighted_rrf_and_ties(chunks):
    a, b = [Hit(c, 1) for c in chunks[:2]]
    result = rrf([[a, b], [b, a]], 2, weights=[2, 1])
    assert result[0].chunk == a.chunk and result[0].score == pytest.approx(2 / 61 + 1 / 62)
    assert rrf([[a], [b]], 2, weights=[1, 0])[0].chunk == a.chunk
    assert [h.chunk.chunk_id for h in rrf([[a], [b]], 2)] == sorted([a.chunk.chunk_id, b.chunk.chunk_id])
    for weights in [[0, 0], [1], [1, -1], [1, float("nan")]]:
        with pytest.raises(ValueError):
            rrf([[a], [b]], 2, weights=weights)


@pytest.mark.parametrize("zero_branch", ["dense", "lexical"])
def test_local_zero_branch_skip(store, zero_branch):
    config = RetrievalConfig(dense_weight=0) if zero_branch == "dense" else RetrievalConfig(lexical_weight=0)
    retriever = LocalRetriever(store, config)
    original = retriever.search
    def guarded(query, method="hybrid", k=10, filters=None):
        assert method != zero_branch
        return original(query, method, k, filters)
    retriever.search = guarded
    retriever.search("account", "hybrid")


def atlas_fake(store, profile="boosted-sum", formulation="raw"):
    adapter = AtlasRetriever.__new__(AtlasRetriever)
    adapter.config = RetrievalConfig(profile, formulation)
    adapter.encoder = store.encoder()
    adapter.text_index, adapter.vector_index, adapter.field_index = "original-text", "original-vector", "new-fields"
    adapter._filters = lambda filters: {**filters, "generation": {"$in": ["active"]}, "model_fingerprint": adapter.encoder.fingerprint}
    captured = []
    adapter.collection = SimpleNamespace(aggregate=lambda pipeline, **kwargs: captured.append(pipeline) or [])
    return adapter, captured


def test_additive_index_and_query_boosts(store):
    adapter, captured = atlas_fake(store)
    old = [m.document for m in adapter.index_definitions()]
    definition = adapter.field_index_definition().document
    mappings = definition["definition"]["mappings"]
    assert not mappings["dynamic"]
    assert {m["type"] for m in mappings["fields"]["control_id"]} == {"token", "string"}
    assert set(mappings["fields"]) >= set(FIELD_BOOSTS)
    assert old == [m.document for m in adapter.index_definitions()]
    adapter.search("CP-9.1 unicorn", "lexical", filters={"release": "rev5"})
    search = captured[0][0]["$search"]
    assert search["index"] == "new-fields"
    clauses = search["compound"]["should"]
    boosts = {next(iter(clause.values()))["path"]: next(iter(clause.values()))["score"]["boost"]["value"] for clause in clauses}
    assert boosts == FIELD_BOOSTS
    assert clauses[0]["equals"]["value"] == "cp-9.1"


def test_best_fields_bounded_union_identical_filters_and_dense(store):
    adapter, captured = atlas_fake(store, "best-fields", "identifier")
    adapter.search("CP-9.1 account", "hybrid", filters={"release": "rev5", "source_id": "test"})
    lexical = [p for p in captured if "$search" in p[0]]
    assert len(lexical) == 4
    filters = [p[0]["$search"]["compound"]["filter"] for p in lexical]
    assert all(f == filters[0] for f in filters)
    assert {"in": {"path": "control_id", "value": ["cp-9.1"]}} in filters[0]
    assert all({"$limit": 100} in p for p in lexical)
    assert all(p.index({"$sort": {"score": -1, "chunk_id": 1}}) < p.index({"$limit": 100}) for p in lexical)
    dense = captured[-1][0]["$vectorSearch"]["filter"]
    assert dense["release"] == "rev5" and dense["source_id"] == "test" and dense["generation"] == {"$in": ["active"]}
    # Identifier constraint applies only to lexical formulation; metadata restrictions match both branches.


def test_atlas_zero_dense_and_no_active_sources(store):
    adapter, captured = atlas_fake(store)
    adapter.config = RetrievalConfig("boosted-sum", dense_weight=0)
    adapter.encoder.encode = lambda *args: pytest.fail("dense call with zero weight")
    adapter.search("account", "hybrid")
    assert all("$search" in p[0] for p in captured)
    captured.clear()
    adapter._filters = lambda _: {"generation": {"$in": []}}
    assert adapter.search("account", "hybrid") == [] and not captured


def test_cli_api_config_and_invalid_weights(store, capsys):
    args = ["--store", str(store.directory), "--lexical-profile", "boosted-sum", "--dense-weight", "0"]
    assert main([*args, "query", "account"]) == 0
    assert json.loads(capsys.readouterr().out)["retrieval_config"]["dense_weight"] == 0
    assert main(["--store", str(store.directory), "--dense-weight", "nan", "query", "account"]) == 2
    config = RetrievalConfig("best-fields", "identifier", 2, 0)
    with TestClient(create_app(store.directory, config=config)) as client:
        assert client.get("/health").json()["retrieval_config"] == config.metadata()
        assert client.post("/query", json={"question": "account"}).json()["retrieval_config"] == config.metadata()


def test_checkpointed_ablation_and_failures(store, tmp_path):
    queries = tmp_path / "queries.json"
    queries.write_text(json.dumps({"label_provenance": "frozen test labels", "queries": [{"id": "q", "query": "account", "filters": {}, "relevant": {"rev5:ac-2": 1}}]}))
    retriever = LocalRetriever(store)
    config = retriever.config
    original = retriever.search
    def search(*args, **kwargs):
        if retriever.config.lexical_profile == "best-fields":
            raise RuntimeError("private failure details")
        return original(*args, **kwargs)
    retriever.search = search
    target = tmp_path / "ablate.json"
    result = ablate(retriever, store, target, queries)
    report = json.loads(target.read_text())
    assert result["status"] == "failed" and retriever.config == config
    assert all(row["status"] in {"complete", "failed"} for row in report["configurations"])
    assert sum("dense" in row.get("summary", {}) for row in report["configurations"]) == 1
    assert report["configurations"][0]["name"] == "common-dense"
    assert "private failure" not in target.read_text()
    assert json.loads((tmp_path / "ablate-level1-combined.json").read_text())["query_file_sha256"] == digest(queries.read_bytes())

def test_atlas_field_setup_only_additive(store):
    adapter, _ = atlas_fake(store)
    existing = [{"name": "original-text", "latestDefinition": {"preserve": True}, "queryable": True},
                {"name": "original-vector", "latestDefinition": {"preserve": True}, "queryable": True}]
    before = json.loads(json.dumps(existing))
    created = []
    def create(model):
        spec = model.document
        created.append(spec)
        existing.append({"name": spec["name"], "latestDefinition": spec["definition"], "queryable": True})
    adapter.collection = SimpleNamespace(list_search_indexes=lambda: existing, create_search_index=create)
    result = adapter.setup_field_index(timeout=1)
    assert result["document_embedding_requests"] == 0 and result["document_rewrites"] == 0
    assert existing[:2] == before and len(created) == 1
    assert adapter.setup_field_index(timeout=1)["queryable"] and len(created) == 1


def test_atlas_best_fields_max_and_deterministic_order(store, chunks):
    adapter, _ = atlas_fake(store, "best-fields")
    results = iter([[Hit(chunks[0], 2), Hit(chunks[1], 1)],
                    [Hit(chunks[0], 1), Hit(chunks[1], 2)], [Hit(chunks[0], .5)]])
    adapter._hits = lambda pipeline: next(results)
    hits = adapter.search("account", "lexical")
    assert [h.score for h in hits] == [2, 2]
    assert [h.chunk.chunk_id for h in hits] == sorted([chunks[0].chunk_id, chunks[1].chunk_id])


def test_identifier_punctuation_and_no_partial_ids():
    assert identifiers("CP-9.1. AC-2, cp-9.11; CP-9.1suffix invalid") == ["cp-9.1", "ac-2", "cp-9.11"]


def test_failed_ablation_cli_returns_nonzero_and_preserves_details(store, tmp_path, monkeypatch, capsys):
    import main as cli

    target = tmp_path / "failed-ablation.json"
    details = {"status": "failed", "configurations": [{"name": "test-config", "status": "failed", "error_type": "ProviderError"}]}

    def failed_ablation(retriever, active_store, output):
        assert active_store.directory == store.directory
        assert str(output) == str(target)
        target.write_text(json.dumps(details), encoding="utf-8")
        return {"output": str(target), **details}

    monkeypatch.setattr(cli, "ablate", failed_ablation)
    assert cli.main(["--store", str(store.directory), "ablate", "--output", str(target)]) == 2
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "failed" and printed["configurations"] == details["configurations"]
    assert json.loads(target.read_text(encoding="utf-8")) == details
