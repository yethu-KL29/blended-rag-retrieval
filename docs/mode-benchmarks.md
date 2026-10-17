# Independent local and enterprise measurements

Measured 2026-10-17. Both modes use the same 20,114 source chunks and frozen 16 queries (14 answerable, 2 negatives). Qrels are incomplete, source-derived and independently unvalidated. These are retrieval measurements, not answer accuracy or proof of general superiority.

## Local: bundled ONNX MiniLM + BM25 + FAISS + RRF

| Method | Recall@5 | Recall@10 | MRR | NDCG@10 | p50 / p95 ms |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.7857 | 0.8214 | 0.7695 | 0.7727 | 93.1 / 210.3 |
| Dense | 0.9286 | 1.0000 | 0.8464 | 0.8780 | 117.3 / 171.9 |
| Hybrid (RRF) | 0.8929 | 1.0000 | 0.9018 | 0.9055 | 237.8 / 382.0 |

## Enterprise: Gemini embeddings + Atlas Search/Vector Search + RRF

| Method | Recall@5 | Recall@10 | MRR | NDCG@10 | p50 / p95 ms |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.7500 | 0.8214 | 0.7556 | 0.7654 | 523.7 / 856.3 |
| Dense | 0.9286 | 1.0000 | 0.8690 | 0.9014 | 1511.1 / 3061.2 |
| Hybrid (RRF) | 0.8929 | 1.0000 | 0.7946 | 0.8433 | 1933.3 / 2652.4 |

Local hybrid improves ranking metrics over local dense, while enterprise hybrid has lower MRR/NDCG than Gemini dense on this small benchmark. The unmodified RRF pipeline is therefore not universally better. Both dense/hybrid modes return hits for one negative question; local answer abstention remains an uncalibrated heuristic. Each report includes precision, negative-return counts and complete per-query rankings.

Latency uses one untimed dense warmup and one timed sequential pass, excluding model loading, ingestion and answer generation. The machine and cloud network were not isolated; these are observations, not controlled capacity estimates.

## Ingestion, usage and costs

- Genuine local ONNX encoding: 20,114 chunks in 1082.18 seconds, 18.59 chunks/s; zero embedding/LLM API calls. The resulting 29,656,389-byte compressed cache is bundled with source/model identity checks. Grader ingestion uses cached document vectors; questions still execute the real model and FAISS search.
- Gemini preparation: 20,114 chunks in 404.32 seconds, 629 online batched requests, 4,523,265 submitted characters and 3,989,423 provider-reported billable characters. The 404.32 seconds covers local preparation/embedding, **not** the subsequent Atlas upload or index build. Atlas independently contains and indexes all 20,114 active chunks from both sources.
- Enterprise benchmark: 33 embedding requests including 1 untimed warmup; 1,359 submitted/1,205 billable characters; zero answer-generation calls. Timed retrieval accounts for 32 embedding requests. An initial attempt failed with AutoReconnect; all enterprise metrics above come from the complete successful retry. Partial failed-run usage is not included in that benchmark report.
- Verified enterprise answer example: `gemini-3.1-flash-lite`, 322 prompt tokens, 48 output tokens, 0 thought tokens and 2 validated citations. At the published global standard rates of $0.25 per million input tokens and $1.50 per million output tokens, this single generation has an illustrative list-price cost of $0.0001525. This excludes its query embedding, Atlas, retries and other smoke tests; it is not an invoice total. [Google pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing).
- Local external API charges are $0; CPU/storage remain unpriced. Actual cloud invoice charges, Atlas storage/index costs and embedding dollar conversion are not independently reconciled, so enterprise total USD remains `null` rather than zero. Counts and token usage are measured; do not treat a partial estimate as total deployment cost.

## Reproduction and evidence

Use [the exact local and enterprise commands](architecture-modes.md) or `run.yaml` for grading. Current compact comparison: [mode-benchmark-summary.json](mode-benchmark-summary.json). Complete actual evidence/rankings: `measurements/local-onnx-faiss.json.gz` and `measurements/enterprise-gemini-atlas.json.gz`. Ingestion records and the cited answer example are stored separately alongside them. Historical LSA/full-precision MiniLM reports retain their original meaning.

## Verification boundaries

All literal `run.yaml` application commands completed in a fresh base-dependency environment with cloud variables cleared and external network denied. Local ingestion, query, comparison, evaluation and idempotence use real data. Windows async API tests need loopback sockets for their event-loop wakeup; external connections and external DNS stay blocked. The clean environment contains neither PyTorch nor the Google generation SDK. Final test/API/idempotence results are recorded in `verification-modes.json`.

Live Atlas tests use only `aintropy_blended_nist_gemini768_v1`, its `_sources` collection and dedicated Gemini text/vector indexes in the authorized existing POC database. Both indexes reached READY/queryable, and `$searchMeta` counted 20,114 active indexed chunks. Enterprise synthesis rejected one response with invalid citation references; the subsequent cited answer example succeeded. Citation validation does not prove factual completeness or semantic entailment.

Docker build, Linux execution, production concurrent load, independent qrel review, reranking, and invoice-level cost reconciliation remain untested. The optional legacy disposable-collection integration test stays skipped; dedicated Gemini live checks are separate and do not touch existing POC data.
