---
title: 'Version-aware compliance Blended RAG'
type: 'feature'
ticket: ''
created: '2026-10-17'
status: 'built'
baseline_revision: '78b1c1ea63a8532ab6994d7864809edba1dc738d'
route: 'full'
route_source: 'auto'
risk: 'medium'
review: 'quick'
review_source: 'pinned'
lenses_ran: ['quick']
review_loop_iteration: 0
context: []
---

<frozen-after-approval reason="human-owned intent — approved in conversation">

## Intent

Build a reproducible enterprise policy and compliance retrieval application for the AIntropy Part 2 Task 7 assessment. Adapt architectural patterns from the existing KMS enterprise reference and IBM Blended RAG, with MongoDB Atlas as the primary search and storage platform. Compare lexical, dense, and fused retrieval on the same evidence and benchmark rather than assuming improvement.

The user approved the plan and explicitly requested implementation. This is a standalone assessment; workflow setup is supporting infrastructure. Existing untracked artifacts/project-context is user-provided context and must be preserved.

## Boundaries & Constraints

Always preserve source URLs, section/control identifiers, source releases, and content hashes. Use public real documents with recorded provenance. Keep historical releases independent. Provide offline tests and a credential-free execution path with clearly labeled capabilities. Separate measured results from estimates and disclose unvalidated relevance labels. Keep code under src, concise write-up under docs, and exact reproducible commands in run.yaml. Adapt reference patterns without copying corporate secrets, private documents, or application-specific configuration.

Never push, provision paid infrastructure, claim live Atlas validation without execution, equate BM25 with learned sparse embeddings, or invent policy changes. Service Bus is optional deployment architecture, not a prerequisite. No unrelated Contact Center analytics, agents, or user role workflows.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Behavior | Error Handling |
|---|---|---|---|
| Query | Valid question and release filter | Ranked cited evidence from that release | Reject invalid filters |
| Version comparison | Two releases and control identifier | Paired source text and explicit textual differences | Identify absent controls |
| Reingestion | Same source release and hash | No duplicate chunks or repeated embedding expense | Fail closed on model mismatch |
| Changed document | Same release with changed source | Replace only that source's chunks after preparation | Preserve other releases |
| No evidence | Empty results | Explicit insufficient evidence | No invented answer |
| Missing service | Atlas requested without configuration | Actionable configuration error | Never silently substitute offline results |

</frozen-after-approval>

## Code Map

- `README.md`, `run.yaml`: official template contracts; no implementation currently exists.
- `artifacts/project-context`: user task brief and enterprise reference path; preserve.
- Reference `kmsai-chat-service/src/retrievers/hybrid_retriever.py`: Atlas text/vector pipelines, filters, native rank fusion; replace ChatState/provider dependencies with common retriever interface.
- Reference `kmsai-ingestion-service/src/services/rag/ingestion_service.py`: batched embedding and MongoDB persistence patterns; adapt deletion to preserve source versions.
- Reference `kmsai-ingestion-service/src/core/queue_manager.py`: lifecycle event publishing; optional deployment concept only.
- IBM reference: Elasticsearch MiniLM dense indexing and ELSER mappings, not a MongoDB-compatible drop-in.

## Tasks & Acceptance

**Execution:**
- [x] `src/blended_rag/models.py`, `ingestion.py`, `dataset.py`: validated shared records, deterministic structure-aware parsing/chunking, pinned public corpus download and incremental storage preparation.
- [x] `src/blended_rag/retrieval.py`, `atlas.py`: common retriever interface, local lexical/dense evaluation mode, Atlas lexical/vector/fusion and reproducible index setup with filtering.
- [x] `src/blended_rag/evaluation.py`: metrics and reproducible per-query evidence, latency percentiles, corpus and resource measurements; clearly mark benchmark label provenance.
- [x] `src/main.py`, `src/blended_rag/api.py`, `answers.py`: CLI/API, grounded evidence response and version comparison, safe configuration handling.
- [x] `src/tests/`: meaningful deterministic tests for filters, chunk preservation, idempotence, fusion, known metrics, empty answers and CLI/API; gated Atlas integration test.
- [x] `requirements.txt`, `Dockerfile`, `.env.example`, `run.yaml`, `README.md`, `docs/README.md`: exact setup and run commands, compact architecture/tradeoffs, measured report and limits.

**Acceptance Criteria:**
- Given a fresh checkout, when the documented offline commands execute, then ingestion, query, comparison, evaluation and tests complete without Atlas credentials.
- Given a configured Atlas collection, when index setup, ingestion and query execute, then lexical and vector searches use configured indexes and matching metadata filters; integration results remain unclaimed until run.
- Given one frozen corpus and query file, when all retrieval methods are evaluated, then Recall@5/10, MRR, NDCG@10, precision and p50/p95 are computed from actual retrieved rankings.
- Given public versioned controls, when queried or compared, then each returned excerpt identifies the exact source release and identifier.

## Implementation Notes

2026-10-17: Installed local BMad runtime after user requested continuation. Prior conversation approved the product scope. Live corpus network availability and Atlas credentials will determine which integration measurements can be obtained, without changing supported interfaces.

Reference audit: artifact-only delete-before-embed risks losing historical and current evidence. Prepare and validate embedding counts, finite values and dimensions before persistence; deterministic upserts must survive partial retries. Reject unknown filters instead of silently dropping them. Initialize optional providers lazily. Do not reuse call_id grouping. Keep retrieval errors distinct from empty results. Existing reranking threshold 25 is unreachable after a final limit of 10; any optional reranker must operate on the candidate pool before final truncation. Parse OSCAL structurally rather than decoding JSON into generic plain text.

Implementation clarification: the offline default may use TF-IDF/SVD LSA, explicitly labeled as a statistical baseline. Include an optional lazy-loaded pretrained sentence-transformers MiniLM encoder for neural semantic retrieval, with isolated fingerprints/dimensions and documented optional installation/model download. Fixed neural embeddings permit incremental updates without fitting the whole corpus. Pinned source LICENSE.md confirms US public domain plus worldwide CC0 dedication.

## Plan Change Log

## Review Triage Log

2026-10-17 quick review: 0 high, 1 medium, 0 low, 0 false, 0 maybe-false. Medium / patch — ingestion.py omits controls without prose/parameters; verified bundled Rev5 ac-2.10 has status withdrawn and incorporated-into #ac-2_smt.k, but parsing emits no evidence and compare infers absence. Preserve status/link evidence in existing Chunk representation and add a regression test.

Root verification: neural cache checks model name but not the fixed model revision fingerprint; a changed pinned revision could reuse stale vectors. Patch the existing mismatch check and test. Atlas source generation uses hashes but not chunk identity/title; parser-only structural changes under a fixed neural model can incorrectly skip updates. Hash complete prepared chunk records and test. Add explicit paired-release comparison coverage for the matrix row.

## Design Notes

Offline retrieval provides reproducible development checks; Atlas remains the production backend. Offline dense representations must be labeled by their actual model, never described as Atlas measurements. Public NIST OSCAL controls and assessment procedures provide hierarchical identifiers and version history. Benchmark seeds may be source-derived but require independent human review before quality conclusions. Client RRF is a compatibility path where native fusion is unavailable. Prepare embeddings before replacing source chunks to reduce failure exposure.

## Verification

- `python -m pytest src/tests -q`: all deterministic checks pass; live Atlas checks skip explicitly without credentials.
- Execute every command in `run.yaml` verbatim, then rerun ingestion to verify idempotence.
- Inspect generated evaluation JSON for actual dataset/model/label provenance and finite metrics.

## Final Verification

2026-10-17: All eight literal run.yaml commands passed inside a clean .venv. Final suite: 31 passed, 1 skipped (live Atlas), 2 dependency/degenerate-fixture warnings. Matrix rows are covered by passing filter, paired-comparison, idempotence/model-mismatch, source-replacement/preparation-failure, empty-evidence and missing-Atlas-config tests. Independent quick review finding and root correctness findings were patched and verified; no findings deferred. Final corpus: 20,114 chunks across 922 Rev4 and 1,196 Rev5 control records, including withdrawn records. BM25/LSA/hybrid Recall@10 .8214/.8214/.8571; NDCG@10 .7727/.7054/.7849. Full local evidence is in artifacts/evaluation.json and artifacts/verification-clean.json; commit-safe measured summary is docs/benchmark-summary.json. Atlas, pretrained MiniLM, and Docker execution remain unvalidated; answers remain extractive and qrels source-derived seeds.
