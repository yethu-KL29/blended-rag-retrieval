# Field-aware retrieval: measured results

Both modes ran all nine predeclared configurations on the same 20,114 chunks and frozen 16 queries. Fourteen queries have positive labels. These judgments are incomplete and independently unvalidated; the source audit confirms that all 17 labeled control IDs exist, not that relevance is fully validated.

## Local

| Configuration | Method | Recall@10 | MRR | NDCG@10 | p95 ms |
|---|---|---:|---:|---:|---:|
| common-dense | dense | 1.0000 | 0.8464 | 0.8780 | 159.4 |
| combined | lexical | 0.8214 | 0.7695 | 0.7727 | 162.5 |
| combined | hybrid | 1.0000 | 0.9018 | 0.9055 | 204.2 |
| content-only | lexical | 0.8214 | 0.5106 | 0.5836 | 124.0 |
| content-only | hybrid | 0.9643 | 0.7411 | 0.7885 | 254.8 |
| multi-field | lexical | 0.8214 | 0.7235 | 0.7394 | 267.4 |
| multi-field | hybrid | 1.0000 | 0.8554 | 0.8717 | 375.1 |
| boosted-sum | lexical | 0.6786 | 0.6908 | 0.6705 | 186.5 |
| boosted-sum | hybrid | 0.9286 | 0.7644 | 0.7911 | 308.1 |
| best-fields | lexical | 0.7500 | 0.6726 | 0.6852 | 175.7 |
| best-fields | hybrid | 0.9286 | 0.7290 | 0.7806 | 382.9 |
| boosted-identifier | lexical | 0.6786 | 0.6908 | 0.6705 | 265.5 |
| boosted-identifier | hybrid | 0.9286 | 0.7644 | 0.7911 | 375.4 |
| boosted-lexical2 | lexical | 0.6786 | 0.6908 | 0.6705 | 191.6 |
| boosted-lexical2 | hybrid | 0.8929 | 0.7254 | 0.7517 | 288.5 |
| boosted-dense2 | lexical | 0.6786 | 0.6908 | 0.6705 | 214.3 |
| boosted-dense2 | hybrid | 1.0000 | 0.8095 | 0.8404 | 310.9 |

## Enterprise

| Configuration | Method | Recall@10 | MRR | NDCG@10 | p95 ms |
|---|---|---:|---:|---:|---:|
| common-dense | dense | 1.0000 | 0.8690 | 0.9014 | 3222.3 |
| combined | lexical | 0.8214 | 0.7556 | 0.7654 | 594.8 |
| combined | hybrid | 1.0000 | 0.7946 | 0.8433 | 6705.1 |
| content-only | lexical | 0.8214 | 0.4809 | 0.5618 | 1575.7 |
| content-only | hybrid | 1.0000 | 0.7937 | 0.8410 | 6797.7 |
| multi-field | lexical | 0.7857 | 0.7210 | 0.7256 | 1412.9 |
| multi-field | hybrid | 1.0000 | 0.8304 | 0.8566 | 6813.5 |
| boosted-sum | lexical | 0.6786 | 0.6616 | 0.6575 | 1384.6 |
| boosted-sum | hybrid | 1.0000 | 0.8214 | 0.8378 | 4454.2 |
| best-fields | lexical | 0.6429 | 0.5923 | 0.5940 | 1534.1 |
| best-fields | hybrid | 0.9286 | 0.7429 | 0.7876 | 4493.8 |
| boosted-identifier | lexical | 0.6786 | 0.6616 | 0.6575 | 783.2 |
| boosted-identifier | hybrid | 1.0000 | 0.8214 | 0.8378 | 3912.9 |
| boosted-lexical2 | lexical | 0.6786 | 0.6616 | 0.6575 | 931.7 |
| boosted-lexical2 | hybrid | 0.7143 | 0.7121 | 0.6894 | 3924.1 |
| boosted-dense2 | lexical | 0.6786 | 0.6616 | 0.6575 | 909.5 |
| boosted-dense2 | hybrid | 1.0000 | 0.8065 | 0.8440 | 3865.4 |

## Interpretation and cost

Field-aware search provides explicit title/section/control-ID coverage and scoring control, but the fixed boosts do not improve quality consistently. The original combined local hybrid retains stronger ranking metrics than the new local presets. On Atlas, uniform multi-field hybrid slightly improves MRR/NDCG over combined hybrid on these seeds; Gemini dense remains a strong reference. These observations do not justify promoting a preset without independently judged held-out queries. The combined default remains unchanged.

The additive Atlas field index became queryable in 23.63 seconds. It indexes all 20,114 active chunks and leaves the original text/vector definitions unchanged. Document rewrites and document embedding requests were zero. Local and enterprise retriever setup times include JSON store/model/index loading; they are not isolated lexical indexing costs.

Local external API calls/charges are zero. The enterprise comparison made 153 query embedding requests with 6,219 submitted and 5,517 provider-reported billable characters, including warmups; answer-generation calls were zero. Atlas index/storage charges and invoice-level total USD remain unpriced. No document re-embedding was required for Level 2; Gemini query embeddings still ran during retrieval. Latencies are one sequential pass with an untimed warmup per measured method, excluding ingestion and answer generation; the environment and network were not isolated.

## Reproduce and inspect

[Commands, profiles, weights and scoring semantics](field-aware-retrieval.md); [configuration-level metrics and usage](field-ablation-summary.json); [offline grader/live Atlas verification](field-verification.json); [source-ID audit](measurements/field-label-integrity.json). Complete rankings/evidence/configuration reports are bundled separately as `measurements/field-local.json.gz` and `measurements/field-enterprise.json.gz`.

`run.yaml` executes the preserved baseline plus a field-aware identifier query and a compact multi-field evaluation. The full nine-configuration comparison is available through `python src/main.py --mode local ablate --output artifacts/evaluation-field-local.json` and its explicitly configured enterprise equivalent. Full ablations are optional for grading runtime; both were independently executed here.

Neither implementation claims ELSER/SPLADE, learned sparse retrieval, a cross-encoder reranker or Elasticsearch best_fields equivalence. Atlas uses native text scores plus exact-ID equals scores; local uses independent BM25 field statistics. Max-field scores use bounded field candidates. Technical `section_id` values are searchable; human-readable family/section names are not invented. See the [IBM reference](https://github.com/ibm-self-serve-assets/Blended-RAG) for the original field/query/index experiments, and [MongoDB compound scoring](https://www.mongodb.com/docs/search/query/operators-collectors/compound/) for native additive behavior.
