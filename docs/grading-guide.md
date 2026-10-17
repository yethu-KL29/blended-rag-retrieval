# Reproduction and evidence guide

This guide maps the requested deliverables to executable commands and supporting files. The assignment's official template and rubric remain authoritative. `run.yaml` retains its original setup/run/test structure; its commands run from the repository root.

[Observed submission verification](submission-verification.json) records the exact successful command replay, environment restrictions, test results and unfinished optional SPLADE validation. It is separate from the historical benchmark reports.

## Execution contract

Use Python 3.11–3.13 and an isolated virtual environment. Install and execute the literal commands in [run.yaml](../run.yaml). Setup requires access to a Python package source. The default application runtime uses bundled public documents, bundled quantized ONNX MiniLM, real FAISS search, persisted BM25 postings and extractive cited answers. It requires no cloud credentials, Torch or model download.

There are three explicit configurations:

| Configuration | Retrieval | Setup | Evidence |
|---|---|---|---|
| Default local | BM25 + neural MiniLM/FAISS + two-way RRF | `requirements.txt`; local `ingest` | [Existing local measurements](mode-benchmarks.md) |
| Optional local research (off by default) | BM25 + MiniLM/FAISS + SPLADE + three-way RRF | `requirements-sparse.txt`; `sparse-setup`; `sparse-index` | [Verified partial results](splade-partial-summary.json) cover 12,672 chunks; full-corpus SPLADE remains unfinished |
| Explicit cloud fallback | Atlas BM25 + Gemini dense + two-way RRF | `requirements-enterprise.txt`; private configuration | [Separate live Atlas/Gemini measurements](mode-benchmarks.md) |

There is no automatic substitution between these modes. `hybrid` always means two-way lexical/dense fusion; `blended` explicitly requests three-way local fusion with positive sparse weight. Missing optional model/index files fail with setup guidance. Current verification status belongs in the associated measurement report; an implemented command alone is not evidence of its execution.

CPU constraints stopped SPLADE document inference at 12,672/20,114 chunks. The saved actual vectors were assembled and searched in a separate partial store without further document encoding. Its complete five-method comparison applies only to that subset; it cannot be presented as a full-corpus result or a controlled comparison against the historical 20,114-chunk baselines. The default BM25/MiniLM system still uses all 20,114 chunks and does not load SPLADE. The [standalone guide](END_TO_END_IMPLEMENTATION.md) explains every stage and records the exact partial scope.

## Requirement-to-evidence map

| Deliverable | Implementation or documentation | How to inspect or verify |
|---|---|---|
| Meaningful real corpus | [Pinned source manifest](../src/data/manifest.json), NIST Rev4/Rev5 catalogs | Local `ingest`; source hash/chunk stability tests |
| Parse, chunk, embed, persist | [Pipeline write-up](README.md), [mode architecture](architecture-modes.md) | Ingestion reports identify cached versus actually embedded chunks; repeat ingestion exercises idempotence |
| Exact lexical retrieval | [Field strategies](field-aware-retrieval.md), `blended_rag/postings.py` | `--method lexical`; tests compare persisted BM25 to a reference formula across profiles |
| Neural dense retrieval | Bundled quantized ONNX MiniLM + FAISS | `--method dense`; real encoder no-network test and FAISS/exact-score comparison |
| Learned sparse retrieval | `blended_rag/sparse.py`, [optional SPLADE setup](local-splade.md) | Explicit `sparse-index`, `--method sparse`, `sparse-inspect`; cached-model smoke is separately gated |
| Blended retrieval | Weighted RRF and field/query ablations | `--method hybrid`; optional `--sparse-weight 1 … --method blended`; [field results](field-experiment-results.md) |
| Cited answers and version changes | Question-focused extractive NLP locally; bounded Gemini synthesis in cloud; exact textual version comparison | Default query/compare commands; citation, no-evidence and API tests |
| Retrieval evaluation | `blended_rag/evaluation.py`, frozen `queries.json` | Local `evaluate` writes complete per-query rankings and aggregate Recall@K, MRR, NDCG and latency |
| Costs and resource limits | [Measurement caveats](mode-benchmarks.md), report resources/usage fields | Local API cost is zero; CPU/storage costs are unpriced; cloud prices/charges remain unmeasured where unknown |
| Working interface/deployment | CLI, FastAPI, Dockerfile | Default commands and API tests; `serve` exposes `/docs`; Docker/Linux execution is not yet independently verified |
| Reproducibility | Exact `run.yaml`, model/source hashes, immutable revisions, explicit modes | Inspect report provenance; run default commands without cloud configuration |

## Tests that protect behavior

Run `python -m pytest src/tests -q`. The suite checks outcomes and failure paths rather than scoring the project by test count.

| Test file | Main contracts |
|---|---|
| `test_pipeline.py` | Structural parsing, chunk/source stability, updates and idempotence, failed-write safety, filters, exact metric examples, citations/comparison, CLI/API |
| `test_modes.py` | Real bundled encoder without network, FAISS correctness, cache invalidation, explicit modes, cloud usage and citation validation |
| `test_fields.py` | Independently scored fields, boosting/max-field behavior, identifier punctuation, weighted fusion, zero-weight branches, additive Atlas index configuration, checkpointed ablations |
| `test_evaluation.py` | Warmup outside timing, empty benchmark rejection, lexical-only isolation |
| `test_sparse.py` | Persisted BM25/reference agreement, sparse dot products, source/model mismatch, artifact tampering, checkpoint resume, atomic publication, five-method evaluation/failure reports |
| `test_sparse_cli.py` | Reject unsupported Atlas sparse methods before constructing a cloud backend |
| `test_atlas.py` | Atlas index/pipeline contracts, filters and generation isolation; live integration is explicitly gated |
| `test_review_fixes.py` | Unknown billing metadata and concurrent API usage attribution |

Default sparse unit tests use a labeled fake encoder to test arithmetic and recovery deterministically. They do not establish SPLADE model quality. The separate real-model smoke uses cached weights only:

```sh
RUN_CACHED_SPLADE_SMOKE=1 python -m pytest src/tests/test_sparse.py -k cached_real -q
```

On PowerShell, set `$env:RUN_CACHED_SPLADE_SMOKE="1"` before running the Python command. Missing live Atlas credentials and an unset real-model smoke flag cause explicit skips; skips are not cloud/model passes. Full source-corpus retrieval benchmarks are separate from these small deterministic tests.

## Reading the results

The frozen evaluation contains 16 questions: 14 answerable seeds and two negatives. Relevance labels are source-derived, incomplete and independently unvalidated. Evaluation collapses the top 100 chunk results into unique `release:control_id` values before scoring. Recall@10 is averaged over answerable questions; negative-return counts are reported separately. Recall measures recovery of labeled controls, not answer correctness.

Reports include the query hash, corpus/model identity, individual rankings and source hashes. Local, historical LSA/full-precision MiniLM, and Atlas/Gemini results remain separate. The sparse comparison uses the same queries for all five methods and records failures rather than silently selecting completed configurations. Its first-call latency shares a process; warmed p95 excludes ingestion and answer generation. A single pass over 16 questions is not a production latency study or evidence of statistical superiority.

The project investigates field/query configurations and lexical/dense/learned sparse representations inspired by [IBM Blended-RAG](https://github.com/ibm-self-serve-assets/Blended-RAG). Its weighted RRF implementation is not an exact reproduction of IBM's experiment. SPLADE and reranking are not claimed mandatory unless the official assignment says so; reranking has not been benchmarked here.

See the [architecture diagrams](system-architecture.md) for indexing, query execution and deployment boundaries.
