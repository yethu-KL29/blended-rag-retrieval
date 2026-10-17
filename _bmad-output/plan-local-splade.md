---
title: 'Local three-retriever Blended RAG with SPLADE'
type: 'feature'
ticket: ''
created: '2026-10-17'
status: 'in-progress'
baseline_revision: '203eab47fa56b3fb5b29231b7f14b84533e015bd'
route: 'full'
route_source: 'auto'
risk: 'medium'
review: 'quick'
review_source: 'pinned'
lenses_ran: ['quick']
context: []
---

<frozen-after-approval reason="user explicitly requested local BM25/dense/SPLADE and MongoDB hybrid fallback">

## Intent

Make local BM25 + MiniLM/FAISS + optional SPLADE the main research implementation. MongoDB/Gemini remains the existing hybrid fallback. Dense neural retrieval already works; preserve it. Add genuine indexed lexical and learned sparse scoring, three-way fusion and reproducible measurements, without claiming exact IBM replication.

## Boundaries & Constraints

Default grader commands remain offline, free of cloud keys, Torch and model downloads; SPLADE is explicit optional setup. No mocked optional retrieval or silent substitution. Preserve corpus, dense caches/FAISS, qrels, previous reports and all MongoDB data/indexes. Do not train models or add Chroma when FAISS already works. Keep qrels unvalidated and separate local/MongoDB results. No push. Coding agent performs no downloads, full-corpus runs or remote operations; root owns these and final measurements.

## I/O & Edge-Case Matrix

| Scenario | Expected behavior |
|---|---|
| Indexed BM25 across field profiles | Same scoring/rankings as brute-force reference; prefiltered top-k |
| Restart or changed source/model | Reload validated indexes; reject corruption and stale/model-mixed data |
| SPLADE weighted terms | True sparse dot product agrees with tiny reference; learned query expansions inspectable |
| Requested sparse without assets/index | Actionable error, no implicit download or alternate results |
| Failed/resumed sparse indexing | Completed batches reused only for identical corpus/model; prior ready index survives |
| Two/three-way fusion or zero branch | Explicit method and weights, no calls to disabled branch |
| Default grader / Mongo hybrid | Existing behavior and provenance remain correct; no sparse imports/calls |

</frozen-after-approval>

## Code Map

- `retrieval.py`: LocalRetriever currently scans Counter entries; field_stats/combined formula and test diagnostics must survive. FaissRetriever dense path already genuine. RetrievalConfig immutable; default weights1/1 and methods lexical/dense/hybrid.
- `ingestion.py`, `providers.py`: dense store/source hash handling is trusted; do not replace schema/assets or existing encoders. Optional sparse model belongs in a separate module with lazy imports.
- `evaluation.py`: preserve metrics and field ablations. Extend explicit methods to sparse/blended; add sparse comparison command using same qrels/protocol with completed reports/checkpoints.
- `main.py`, `api.py`: explicit optional sparse setup/query/evaluation and static config, existing API lock/provenance. Atlas must reject sparse/blended explicitly, retaining its hybrid pipeline.
- `src/tests`:75 pass/1 optional live skip at baseline. Agent owns source/tests/docs, root owns local model prefetch, full corpus indexing, no-network/live proofs and measured artifacts.

## Tasks & Acceptance

- [ ] Add persisted BM25 inverted postings (SciPy CSC/CSR or equivalent) for combined and separate fields; no full per-term document scans at query time. Bind format/tokenizer/k1/b, ordered complete chunks and field definitions; validate shape/finiteness/hash, atomic writes. Preserve reference statistics/API as needed for regression; compare scores against brute-force for all profiles and filters.
- [ ] Add lazy `SparseEncoder` provider for `naver/splade-cocondenser-ensembledistil`, pinned revision `49cf4c7b0db5b870a401ddf5e2669993ef3699c7`, CPU max256, bounded threads/batches. Use actual `encode_document` and `encode_query`, raw nonnegative SPLADE weights and dot product, never dense normalization or query lookup. Model has30522 vocabulary dimensions; document input is title + control ID + text, disclosed separately from field-aware BM25.
- [ ] Add separate persistent sparse store/index and explicit `sparse-index` command with resumable small batch checkpoints, progress/usage/timing, integrity checks and atomic ready manifest. Validate counts, finite nonnegative weights, vocab shape, full source/model identity and file digests. Do not materialize corpus-by-vocabulary dense arrays; only N-document scores may be dense. Setup may explicitly download pinned weights; queries are local-files-only. Include model/revision/license/resource provenance and optional dependencies separate from base.
- [ ] Add `sparse` direct retrieval and `blended` weighted three-way RRF, with `sparse_weight` default0 so old defaults remain unchanged; explicit blended requires positive sparse weight and ready sparse index. Sparse-only can run directly; hybrid stays two-way. Filter before top-k, deterministic ties, zero-weight branch skipping, exact source citations/NLP answers. Wire CLI/API and reject sparse selection on Atlas without affecting hybrid.
- [ ] Add a sparse evaluation command covering BM25-only, neural dense-only, SPLADE-only, two-way hybrid and three-way blended on the same frozen queries. Preserve field-aware options and report input representations, index build time/bytes/nnz, model disk size, cold/warm latency, Recall@10/MRR/NDCG/p95 and external API cost0. Store each completed configuration separately, flag failures, return nonzero if incomplete. Do not promote sparse to default from small seeds alone.
- [ ] Test matrix, format/source/model mismatch, batch resume/failure atomicity, sparse dot products, FAISS/default regression, CLI/API methods, Atlas rejection and no-Torch/no-key default operation. Use fake encoder only in deterministic tests, clearly distinguished from root actual-model evidence. Provide a gated real-model smoke path using cached weights.
- [ ] Update README, architecture, setup/dependencies and exact commands: local primary, Mongo hybrid fallback, optional model download then offline operation, no automatic switching, scope/limits and CC-BY-NC-SA-4.0 model attribution. Keep official template/run.yaml functional without requiring SPLADE. Previous enterprise-first decision is superseded, historical measurements remain labeled.

**Acceptance:** Given existing corpus/dense store and pinned optional weights, root indexes all20114 chunks once, queries with real SPLADE and compares all five methods. Given no cloud settings/Torch, every default grader command succeeds using neural MiniLM/FAISS and indexed BM25. Given Mongo fallback, hybrid still works and sparse requests fail clearly.

## Implementation Notes

Root prefetches the pinned model into the local Hugging Face cache; do not bundle440MB weights into Git. Root will measure CPU throughput and long full-corpus indexing, so expose a practical batch/progress/resume interface early. Current optional environment has sentence-transformers5.1.2, transformers4.57.6 and Torch2.14.1+cpu. Default base environment has none of these. User authorized this concrete extension and architecture change; no repeated approval needed.

## Plan Change Log

## Review Triage Log

| Finding | Verdict | Verified evidence and route |
|---|---|---|
| Sparse checkpoint entries can be reordered | medium | Resume indexes batches by list position while `_read_matrix` only validates self-declared names/hashes. Expected generation/batch filenames already exist and can be checked without schema changes. Patch: reject swapped or excess entries before reuse. |
| Sparse evaluation can mislabel legacy cloud encoders | medium | CLI explicit store without mode selects LocalRetriever; its encoder can be Gemini while sparse evaluation hardcodes MiniLM/FAISS and zero API cost. Patch: reject cloud encoder before calls and derive supported legacy-local representation metadata from actual provider. |

## Verification

- Agent runs focused tests then full regression; root independently reviews and checks all matrix rows.
- Root verifies default run.yaml in the existing fresh base-only environment with external networking/cloud variables denied.
- Root runs real pinned model smoke, full20114-chunk sparse indexing, restart/idempotence and all five retrieval methods; save actual complete rankings and costs separately. Test cached model queries without network.
- Independent quick review, verified patches, final tests, clean local conventional commit; no push.

## Grader evidence extension

User requested meaningful tests, documentation, architecture diagrams and explanatory code comments. Added `docs/grading-guide.md` mapping requirements to commands/files/test contracts, and `docs/system-architecture.md` covering independent indexes, retrieval and deployment. README/docs link both. Existing original submission contract remains intact. No grader-directed evaluation manipulation instructions are introduced.

## Verified handoff at user-requested conclusion

Implementation and independent review fixes are complete. Root regression:89 passed/2 gated skips. All9 literal run.yaml run/test commands passed in isolated base-only environment with external network/DNS denied and cloud variables cleared; final tests were replayed after review fixes. Live Atlas/Gemini hybrid returned3 citations with extractive answers. See docs/submission-verification.json.

User requested conclusion before the long CPU job finished. Full SPLADE indexing stopped at 7168/20114 persisted chunks, with no ready full-corpus index or five-method benchmark published. Checkpoints remain resumable. Plan is intentionally not marked built: full-corpus acceptance and final measurements remain incomplete.
