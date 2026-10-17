---
title: 'Enterprise-first RAG with an offline grading mode'
type: 'feature'
ticket: ''
created: '2026-10-17'
status: 'built'
baseline_revision: '15e7223fde423a54b961a293d1a470ac41fcc239'
route: 'full'
route_source: 'auto'
risk: 'medium'
review: 'quick'
review_source: 'pinned'
lenses_ran: ['quick']
context: []
---

<frozen-after-approval reason="human-owned intent — explicit user architecture decision">

## Intent

Make the existing application explicitly enterprise-first in architecture, with a fully functional local execution mode for grading. Primary enterprise mode uses MongoDB Atlas lexical/vector search, Gemini embeddings, RRF and grounded Gemini answers. Default grading mode uses real local BM25, MiniLM, FAISS and RRF with question-focused extractive NLP answers, without cloud keys, runtime downloads or manual configuration. Preserve existing LSA/full-precision MiniLM baselines, metrics, historical reports and the official template. This is one cohesive mode contract; the user explicitly requested both implementations.

## Boundaries & Constraints

Always expose execution mode, embedding model, vector engine and answer engine in CLI/API results and benchmark reports. Enterprise errors must never silently become local results. Use separate stores/model fingerprints and dedicated new Atlas namespaces in the existing authorized POC database; never modify old POC collections/indexes. Read credentials privately and never log or commit them. Report local and enterprise measurements separately, including warmup/API usage and unpriced cloud costs. Preserve small/unvalidated qrel caveats; retrieval recall is not answer accuracy. No push or paid infrastructure provisioning.

Default run.yaml must work from a checkout after dependency installation, with network and provider credentials absent during application commands. Bundled, licensed model assets and a source/model-validated cache of genuinely computed corpus vectors are permitted; query embeddings and FAISS retrieval always run locally. Corpus changes/cache mismatches must trigger real encoding. Explain cached versus newly embedded counts. Keep local answers extractive, quoted and cited; do not claim neural generation or calibrated abstention. Enterprise synthesis must use bounded evidence, validate citation IDs, and report actual model/token usage.

## I/O & Edge-Case Matrix

| Scenario | Expected behavior |
|---|---|
| Explicit local query; no keys/network | MiniLM query embedding, filtered FAISS/BM25/RRF, cited NLP answer |
| Explicit enterprise query | Atlas retrieval with Gemini query task and Gemini cited answer |
| Provider/network failure | Clear enterprise error; no mode substitution or credential leak |
| Restrictive metadata filter | Filter before vector top-k; zero matches return insufficient evidence |
| Duplicate/changed ingestion | Reuse unchanged fixed-model vectors; preserve other releases and failed prior state |
| Model/cache mismatch | Reject incompatible state/cache; never mix embeddings |
| Unrelated question | Local relevance heuristic can abstain; document that it is not calibrated |
| No evidence / invalid citation | No unsupported synthesized answer; explicit insufficiency/error |

</frozen-after-approval>

## Code Map

- `src/blended_rag/ingestion.py`: existing parser/LocalStore and LSA/full-precision MiniLM; generalize fixed-model reuse, retain historical compatibility.
- `retrieval.py`, `atlas.py`: existing BM25/RRF and Atlas generation pointers/prefilters; preserve these methods and add actual filtered FAISS search.
- `answers.py`, `api.py`, `src/main.py`: replace raw concatenation with local sentence selection and explicit enterprise synthesis/mode routing.
- `evaluation.py`: preserve formulas; replace hardcoded zero cloud usage with true provider counters and explicit provenance.
- `src/assets/minilm/`: root has already downloaded pinned ONNX quint8 model, tokenizer/configs, SHA manifest and Apache-2.0 license. Do not replace raw assets.
- Existing tests and historical `docs/*summary.json`/measurements remain regression baselines.

## Tasks & Acceptance

- [x] Add bundled ONNX MiniLM encoder (attention-mask mean pooling, L2 normalization, length 256, CPU threads bounded), Gemini provider and optional Google dependencies. Gemini uses RETRIEVAL_DOCUMENT versus RETRIEVAL_QUERY, configurable model/dimensions (default gemini-embedding-001/768), bounded retries/timeouts and real billable-character/request counters. Use tested google-genai 1.75.0 with Vertex/ADC or explicit API key.
- [x] Generalize fixed-model cache identity and incremental reuse. Provide a documented cache-export command/helper so root can build a bundled compressed corpus cache after real ONNX ingestion. Cache keys bind ordered chunk/input hashes and model fingerprint; validate all counts/dimensions/finiteness. Root runs model computation and live experiments after implementation.
- [x] Add FAISS IndexFlatIP on normalized float32, durable index metadata and correct filtered subset mapping. Preserve existing LocalRetriever legacy baselines; expose modern local mode separately if helpful.
- [x] Add explicit `--mode local|enterprise` and separate defaults (`artifacts/runtime-local`, `artifacts/runtime-enterprise`); default grading commands explicitly choose local. Preserve legacy backend/encoder routes. Allow explicit `--env-file` for private enterprise configuration, never auto-select enterprise from ambient keys.
- [x] Implement question-focused extractive NLP sentence selection with exact source text/citations, deduplication and conservative insufficiency heuristic. Add Gemini structured cited synthesis with configurable chat model, bounded context and citation validation. Allow explicit extractive answers in enterprise mode; no silent fallback.
- [x] Extend reports with mode/provider/vector/answer provenance and actual usage deltas; enterprise unknown USD costs are null/unpriced, not zero. Record warmup calls even though untimed. Keep evaluation retrieval-only and answer validation separate.
- [x] Update run.yaml, dependency sets, Docker, .env.example, README and concise docs. Record this architecture decision explicitly. Include exact local/baseline/enterprise commands, optional cloud setup, caches/asset attribution and separate results/untested status. Local base dependencies need no PyTorch: faiss-cpu 1.12.0, onnxruntime 1.23.2, tokenizers 0.22.2 have verified Windows cp313 and Linux wheels.
- [x] Add meaningful tests for FAISS-vs-exact cosine and filter-before-top-k; model/artifact/cache mismatch; provider document/query task payloads; NLP citations/abstention; Gemini citation validation; CLI/API mode identity; usage attribution; and no-network/no-key local execution. Retain existing tests.

**Acceptance Criteria:**
- Given a checkout with dependencies installed and no network/cloud credentials, when every run.yaml application command executes, then real MiniLM/FAISS retrieval, local answers, comparison and evaluation succeed and identify local mode.
- Given authorized POC cloud credentials, when enterprise ingestion/index/query/evaluation execute, then new namespaces contain Gemini vectors and actual Atlas results are reported separately; failures remain explicitly untested/failed.
- Given the frozen corpus and qrels, when local and enterprise evaluate, then their source/query hashes match while provider/mode/usage remain distinct.

## Implementation Notes

Root preflight now confirms Atlas 8.0.34 access to the authorized POC database and a valid Gemini 768-dimensional vector. Model assets are pinned to upstream 1110a243fdf4706b3f48f1d95db1a4f5529b4d41; ONNX files total 23.5 MB. Intended live namespaces: aintropy_blended_nist_gemini768_v1 and its _sources collection, with aintropy_nist_gemini_text_v1 / aintropy_nist_gemini_vector_v1 indexes. Root owns private configuration, cache generation and live measurements; implementation agent owns source/tests/docs. Do not claim these new modes tested until their runs finish.

## Plan Change Log

## Review Triage Log

- medium / patch: ingestion returns raw Gemini usage, bypassing unknown billing metadata handling. Confirmed return at LocalStore.ingest and flag handling in usage_snapshot; fix to use shared snapshot semantics.
- medium / patch: API shares cumulative counters across synchronous concurrent requests; snapshot/delta covers no lock. Confirmed shared lifespan retriever and answer flow. Serialize full API query operation and test attribution.

## Verification

- Run deterministic tests, then every literal run.yaml command in a clean environment with cloud variables cleared and network denied during runtime.
- Independently run enterprise setup/ingestion/index readiness/query/compare/idempotence/evaluation and capture actual API usage. Use only the dedicated namespaces above.
- Inspect compact model/cache manifests, separate raw rankings and architecture documentation; no secrets or 100%-accuracy claims.

## Completed verification evidence

- All7 literal grading run/test commands passed in a fresh base-only venv; no PyTorch/Google generation SDK, cloud variables cleared and external network denied. Windows asyncio requires internal loopback wakeup sockets; corrected an overly strict verification harness without changing product code. Final suite48 passed,1 optional legacy live skip.
- Local ONNX genuinely encoded20114 chunks; source/model-bound29656389-byte compressed cache exported. Fresh grader ingestion reports0 newly embedded/20114 cached chunks; query embeddings and FAISS are real.
- Gemini genuinely encoded20114 chunks in629 online requests. Dedicated Atlas collection and both READY/queryable indexes contain20114 active documents across2 source pointers; searchMeta independently confirmed visibility.
- Full retrieval benchmark completed for both modes on identical corpus/query hashes. Complete successful enterprise retry is distinguished from initial AutoReconnect failure; no partial fabricated report.
- Live Gemini grounded CLI response and enterprise API health/query/compare passed; unknown/invalid citation references are rejected. Enterprise reingestion embedded0/uploaded0.
- Independent quick review findings were verified and patched; no deferred findings. Unknown usage and concurrent attribution regression tests pass.
- docs/mode-benchmarks.md, docs/mode-benchmark-summary.json, docs/verification-modes.json and compressed complete rankings preserve separate local/enterprise evidence and explicit untested boundaries.

- Exact Git index checkout independently validates all pinned model hashes, complete20114-chunk cache identity and real offline query encoding; `.gitattributes` preserves hashed bytes on Windows.
