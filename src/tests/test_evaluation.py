import json

import pytest

from blended_rag.evaluation import evaluate


def test_lazy_model_warmup_precedes_timing_and_is_not_scored(store, tmp_path, monkeypatch):
    events = []

    class LazyRetriever:
        model = {"name": "test-lazy-model"}

        def search(self, query, method, k, filters):
            events.append(("search", method, k))
            return []

    clock = iter(range(6))

    def timer():
        events.append(("timer",))
        return next(clock)

    monkeypatch.setattr("blended_rag.evaluation.time.perf_counter", timer)
    queries = tmp_path / "queries.json"
    queries.write_text(json.dumps({"label_provenance": {"status": "test"}, "queries": [
        {"id": "one", "query": "account", "filters": {}, "relevant": {"rev5:ac-2": 2}}
    ]}), encoding="utf-8")
    output = tmp_path / "evaluation.json"
    evaluate(LazyRetriever(), store, output, queries)
    assert events[:2] == [("search", "dense", 1), ("timer",)]
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["query_count"] == 1
    assert len(report["per_query"]) == 3
    assert all(row["latency_ms"] == 1000 for row in report["per_query"])
    assert "untimed dense warmup" in report["measurement"]


def test_empty_benchmark_is_rejected(store, tmp_path):
    queries = tmp_path / "empty.json"
    queries.write_text(json.dumps({"queries": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="at least one query"):
        evaluate(None, store, tmp_path / "report.json", queries)


def test_explicit_lexical_only_evaluation_never_warms_dense(store, tmp_path):
    calls = []

    class LexicalOnlyRetriever:
        model = {"name": "lexical-only"}

        def search(self, query, method, k, filters):
            assert method == "lexical", "Explicit lexical-only evaluation called dense retrieval"
            calls.append((method, k))
            return []

    queries = tmp_path / "queries.json"
    queries.write_text(json.dumps({"label_provenance": "test", "queries": [
        {"id": "one", "query": "account", "filters": {}, "relevant": {"rev5:ac-2": 1}}
    ]}), encoding="utf-8")
    output = tmp_path / "lexical.json"
    evaluate(LexicalOnlyRetriever(), store, output, queries, methods=["lexical"])
    assert calls == [("lexical", 100), ("lexical", 100)]
    report = json.loads(output.read_text(encoding="utf-8"))
    assert set(report["summary"]) == {"lexical"}
    assert "per measured method" in report["measurement"]
