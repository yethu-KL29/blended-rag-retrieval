from __future__ import annotations

import json
import math
import platform
import time
from pathlib import Path

import numpy as np

from .dataset import DATA, manifest
from .models import digest
from .providers import provenance, usage_snapshot, usage_delta


def metrics(ranking: list[str], relevant: dict[str, int]) -> dict:
    # qrels identify controls within one release, not every split excerpt as a separate relevant document.
    ranking = list(dict.fromkeys(ranking))
    positive = {key for key, grade in relevant.items() if grade > 0}
    def recall(k):
        return len(set(ranking[:k]) & positive) / len(positive) if positive else 0.0
    dcg = sum((2 ** relevant.get(key, 0) - 1) / math.log2(i + 2) for i, key in enumerate(ranking[:10]))
    ideal = sum((2 ** grade - 1) / math.log2(i + 2) for i, grade in enumerate(sorted(relevant.values(), reverse=True)[:10]))
    return {"recall@5": recall(5), "recall@10": recall(10),
            "mrr": next((1 / i for i, key in enumerate(ranking, 1) if key in positive), 0.0),
            "ndcg@10": dcg / ideal if ideal else 0.0,
            "precision@5": len(set(ranking[:5]) & positive) / 5,
            "precision@10": len(set(ranking[:10]) & positive) / 10}


def evaluate(retriever, store, output="artifacts/evaluation.json", queries=DATA / "queries.json"):
    raw = Path(queries).read_bytes()
    benchmark = json.loads(raw)
    if not benchmark["queries"]:
        raise ValueError("Benchmark must contain at least one query")
    first = benchmark["queries"][0]
    before = usage_snapshot(retriever)
    # MiniLM loads lazily on its first dense request; keep that cold start outside timing.
    retriever.search(first["query"], "dense", 1, first["filters"])
    after_warmup = usage_snapshot(retriever)
    rows, summaries = [], {}
    for method in ["lexical", "dense", "hybrid"]:
        method_rows = []
        for item in benchmark["queries"]:
            start = time.perf_counter()
            # Retrieve a fixed pool, deduplicate controls, then score top controls.
            hits = retriever.search(item["query"], method, 100, item["filters"])
            latency = (time.perf_counter() - start) * 1000
            ranking = list(dict.fromkeys(f"{h.chunk.release}:{h.chunk.control_id}" for h in hits))
            row = {"query_id": item["id"], "method": method, "latency_ms": latency,
                   "answerable": bool(item["relevant"]),
                   "ranking": ranking, "metrics": metrics(ranking, item["relevant"]),
                   "evidence": [{"chunk_id": h.chunk.chunk_id, "control_id": h.chunk.control_id,
                                 "release": h.chunk.release, "score": h.score, "source_hash": h.chunk.source_hash} for h in hits]}
            rows.append(row)
            method_rows.append(row)
        times = [r["latency_ms"] for r in method_rows]
        positive_rows = [r for r in method_rows if r["answerable"]]
        negative_rows = [r for r in method_rows if not r["answerable"]]
        summaries[method] = {key: float(np.mean([r["metrics"][key] for r in positive_rows])) if positive_rows else 0.0
                             for key in method_rows[0]["metrics"]}
        summaries[method].update({"answerable_queries": len(positive_rows), "negative_queries": len(negative_rows),
                                  "negative_queries_returning_hits": sum(bool(r["ranking"]) for r in negative_rows)})
        summaries[method].update({"p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95)),
                                  "sequential_queries_per_second": len(times) / (sum(times) / 1000)})
    state = store.read()
    usage = usage_delta(before, usage_snapshot(retriever))
    enterprise = getattr(retriever, "mode", None) == "enterprise"
    report = {"backend": type(retriever).__name__, "model": retriever.model,
              **provenance(retriever), "usage": usage,
              "warmup_usage": usage_delta(before, after_warmup),
              "timed_usage": usage_delta(after_warmup, usage_snapshot(retriever)),
              "answer_validation": "Retrieval-only evaluation; recall is not answer accuracy. Qrels are small and unvalidated.",
              "label_provenance": benchmark["label_provenance"], "query_file_sha256": digest(raw),
              "query_count": len(benchmark["queries"]), "corpus": manifest(),
              "python": platform.python_version(), "platform": platform.platform(),
              "measurement": "One untimed dense warmup, then one single-process timed pass; model load/ingestion excluded. No concurrency load test.",
              "retrieval_unit": "release:control_id; ranked chunk pool of 100 collapsed to unique controls",
              "resources": {"chunks": len(state["chunks"]), "store_bytes": store.path.stat().st_size,
                            "encoder_bytes": (store.directory / state["model_file"]).stat().st_size,
                            "embedding_api_calls": usage.get("embedding_requests", 0), "llm_calls": usage.get("generation_requests", 0),
                            "external_api_cost_usd": None if enterprise else 0,
                            "cost_note": "Local CPU/storage costs unpriced. Atlas storage/index size and charges unmeasured."},
              "summary": summaries, "per_query": rows}
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return {"output": str(target), "summary": summaries, "label_provenance": benchmark["label_provenance"],
            **provenance(retriever), "usage": usage, "warmup_usage": report["warmup_usage"]}
