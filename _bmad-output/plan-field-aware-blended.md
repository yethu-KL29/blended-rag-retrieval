---
title: 'Field-aware blended retrieval experiments'
type: 'feature'
ticket: ''
created: '2026-10-17'
status: 'built'
baseline_revision: '775bbfefd0efad2f3b80930613b3da1cdd5e4f45'
route: 'full'
route_source: 'auto'
risk: 'medium'
review: 'quick'
review_source: 'pinned'
lenses_ran: ['quick']
context: []
---

<frozen-after-approval reason="user explicitly requested field-aware indexing/search and Level 1/2 experiments">

## Intent

Current local BM25 and Atlas lexical search concatenate control_id, title and text, without separate field statistics, weights or section coverage. Extend both modes with explicit field-aware lexical profiles, query formulations and weighted fusion, then measure Level 2 against the preserved Level 1 baseline. Enterprise remains Atlas/Gemini; local remains credential-free MiniLM/FAISS/NLP.

## Boundaries & Constraints

Always preserve default combined scoring, original metrics, qrels, embeddings, caches, collections and original Atlas indexes. Search all useful fields available in the current chunk schema: control_id, title, section_id and text (the description/body). Release/source remain filters; hashes, URLs and vectors are provenance/storage, not relevance-bearing prose. Do not invent missing human-readable section names or claim every metadata field should be tokenized. Add a new dedicated field-search index on the already authorized collection without re-embedding or rewriting documents.

Report configurations, field names/boosts, query formulation, fusion weights, usage, setup/index time and actual ranking metrics. Keep local/enterprise experiments separate. No best_fields/Elasticsearch equivalence claims without explaining client candidate limits. Qrels remain small and independently unvalidated; no automatic winner/default promotion or superiority claims. SPLADE and reranking remain optional future experiments, not implemented claims. No push, infrastructure provisioning or POC namespace changes.

## I/O & Edge-Case Matrix

| Scenario | Expected behavior |
|---|---|
| Query matches title or section but not body | Multi-field profiles retrieve it; content-only does not |
| Exact ID CP-9.1 versus CP-9 | Explicit identifier formulation preserves enhancement identity and normalizes case |
| Strong title match versus weak body repetition | Boosted title/control-ID score can change order using independently normalized fields |
| Field best-fields combination | Max weighted field score; finite Atlas candidate union explicitly documented |
| Restrictive metadata/no matches | Same filters on every lexical subquery and dense branch; no leakage |
| Invalid/zero/nonfinite fusion settings | Reject; a zero-weight branch makes no provider/search calls |
| Existing default commands | Same combined profile and original results; offline grader commands still work |

</frozen-after-approval>

## Code Map

- `retrieval.py`: preserve combined Counters/default RRF; FaissRetriever overrides dense only. Add separate field scoring/config, retain stores/encoders.
- `atlas.py`: explicit mappings; lexical searches search_text. New field index must leave original text/vector definitions intact.
- `models.py`: text is description/body; section_id is technical, no family title. Preserve dotted exact enhancement IDs.
- `main.py`, `api.py`, `providers.py`: CLI/static API config and provenance; shared API retriever already serializes queries.
- `evaluation.py`: reuse frozen qrels/metrics; add checkpointed ablations and method subsets.
- Agent owns source/tests/docs, never assets/old reports/stores/remote operations. Root owns private config, live operations and new measured reports.

## Tasks & Acceptance

- [x] Implement validated immutable retrieval configuration and CLI global flags for lexical profile, raw/identifier formulation, lexical/dense RRF weights. Expose the same static config in API serving, answers/health/evaluation.
- [x] Add content-only, preserved combined, uniform multi-field sum, boosted sum (control_id8/title3/section1.5/text1), and boosted best_fields(max) profiles. Local per-field BM25 has separate lengths/IDF; ID field is exact-token when appropriate. Use deterministic regex identifier extraction, never labels.
- [x] Add weighted RRF with safe validation, deterministic ties and zero-branch skipping; defaults reproduce unweighted baseline. Preserve embeddings.
- [x] Add Atlas field index `aintropy_nist_gemini_fields_v2` (legacy derives its own name) and readiness setup without altering existing definitions. Use boosted compound clauses for sum. Best-fields merges max scores across bounded field candidates with identical filters; disclose limits.
- [x] Add `ablate` CLI, using unchanged corpus/qrels and common warmup/candidate protocol. Predeclare profiles plus identifier/fusion-weight variations, evaluate lexical/hybrid and one common dense baseline. Checkpoint each configuration with explicit completion/failure, usage and setup timing separate from warm latency.
- [x] Test every matrix row, Atlas index/query payloads and filters without keys, default regression, invalid parameters, exact enhancement IDs, weighted RRF and CLI/API wiring. Document commands, scoring distinctions and root-produced results.

**Acceptance:** Given existing stores and credentials, new profiles run on all20114 chunks locally and through the new Atlas index without new embeddings. Given no credentials/network, run.yaml still succeeds. Given the frozen benchmark, independent local/enterprise ablations include Recall@10, MRR, NDCG@10, p95, setup/usage and limitations without relabeling old metrics or claiming validated qrels.

## Implementation Notes

User authorization covers this concrete extension; no repeated approval checkpoint. New index is additive and within the dedicated authorized collection. Coding agent performs no remote operations; root runs index setup and real experiments. Default combined profile remains unchanged regardless of results.

## Plan Change Log

## Review Triage Log

- medium / patch: Explicit lexical-only evaluation unconditionally warms dense when warm_methods=False. Confirmed evaluate entry path and CLI methods forwarding. Warm explicit subsets only; retain default historical warmup.
- medium / patch: Failed ablate results are printed then main returns0. Confirmed exception checkpoint handling and unconditional CLI success. Return nonzero for failed status while preserving details.

## Verification

- Run focused field/config tests and full regression suite.
- Root verifies local profile queries/ablations without external network and literal run.yaml compatibility.
- Root creates only the new dedicated Atlas field index, checks readiness and exact field-only queries, then runs separate enterprise ablations. Capture existing index definitions before/after, index-build duration and zero embedding recomputation.
- Independent quick review, verified fixes, final full tests and local conventional commit; no push.

## Completed evidence

- Independent quick review found two medium patch issues; explicit method warmups and failed-ablation CLI status were fixed, with regressions. Final isolated base-dependency suite75 passed/1 optional legacy live skip. Every frozen matrix row is covered by passing field/config/Atlas/default tests.
- All9 current run.yaml application/test commands passed with external networking denied and cloud settings cleared; fresh base venv has neither Torch nor google-genai. Root added a compact field query/evaluation to the grading recipe; full9-configuration ablations remain optional and were independently run in both modes.
- Both actual ablations completed all9 configurations on identical frozen corpus/qrels. Local combined full rankings exactly match the preserved original baseline. New field index indexes20114 documents, all4 field-only checks preserve filters, and original Atlas definitions are unchanged. Index build23.63sec, zero document rewrites/embedding requests.
- Enterprise field-aware API health/query returned200 with exact CP-9.1 citations and zero provider requests for lexical/NLP mode. Full enterprise ablation used153 query embedding requests,6219 submitted/5517 billable characters, zero generation calls.
- Raw rankings, profiles, scores, timing, setup/usage, source-ID audit and proof are separate local/enterprise artifacts under docs. Fixed boosts do not consistently improve quality; no default promotion, no validated-label or learned-sparse claims.
