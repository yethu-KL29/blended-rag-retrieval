---
title: 'Run neural retrieval and isolated Atlas experiment'
type: 'bugfix'
ticket: ''
created: '2026-10-17'
status: 'built'
baseline_revision: '15bfe850126ec80414bdefe5a246506f1de32abb'
route: 'oneshot'
route_source: 'auto'
risk: 'medium'
review: 'quick'
review_source: 'pinned'
lenses_ran: ['quick']
context: []
---

<frozen-after-approval reason="human-owned intent — user requested live experiment">

## Intent

Keep the working baseline and run pretrained MiniLM retrieval on the same corpus and frozen query set, then test MongoDB Atlas lexical/vector/hybrid retrieval in the KMS RAG POC database with dedicated new collections and indexes. Reuse connection configuration privately; never print or commit credentials, modify POC documents/indexes, bypass network controls, or claim successful Atlas execution when access fails. Record any network blocker and actual measurements transparently.

Correct the reachable lazy model loading issue so the evaluator's claim that model loading is excluded from measured retrieval latency is true for MiniLM. Perform an untimed dense warmup through the common retriever interface before timed queries, record it in measurement metadata, and cover ordering and excluded warmup results with a deterministic test. Report seed-benchmark limitations unchanged.

</frozen-after-approval>

## Implementation Notes

Initial preflight: POC URI resolves to three nodes, but all TCP connections time out; verified again with explicit CA bundle and again after dependency installation. No authentication or TLS handshake reached and no Atlas writes performed. User has been asked to check VPN/IP network access. MiniLM dependencies and pinned model are downloaded; corpus encoding is running. Code scope is fewer than 100 changed lines: warmup and one regression test; documentation/report updates accompany the actual experiment.

Completed local run: all 20,114 chunks encoded with pinned MiniLM in 573.16 seconds; unchanged reingestion embeds zero chunks. Same corpus/query hashes verified across LSA and MiniLM reports. Dense MiniLM and its hybrid Recall@10 = 1.0 on 14 answerable seeds; one out-of-domain negative returns hits for both. LSA comparison rerun with the same warmup and four-thread settings. Full compressed evidence and comparison summary saved under docs; credentials and raw connection configuration remain outside the repository. Tests: 33 passed, 1 skipped (live Atlas). Review scope is quick; broader architectural lenses are skipped for this bounded timing correction and experiment report. The live Atlas experiment remains blocked on network reachability; no remote collections/indexes were created.

## Verification

- Run targeted evaluation regression and the deterministic suite after code changes.
- Run MiniLM ingestion/evaluation/idempotence on the frozen corpus and preserve actual rankings.
- Retry Atlas only with configured POC credentials and dedicated names; if still unreachable, report the blocker and continue independent local verification.

## Review Triage Log

Quick independent review: no findings. The completed artifact is the local neural experiment, corrected latency measurement and honest deployment recipe. The requested live Atlas experiment remains incomplete because network access is unavailable; dedicated collections/indexes have not been created. No pre-existing POC data or configuration files were changed. Full tests passed (33 passed, 1 skipped); no code findings deferred.
