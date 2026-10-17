# Field-aware retrieval experiments

The default remains `combined`, raw queries, and lexical/dense RRF weights 1/1. These preserve Level 1 scores and rankings. Existing stores, embeddings, caches, qrels and original Atlas index definitions remain unchanged. The extension runs with the existing local MiniLM/FAISS store or the separate enterprise Atlas/Gemini store.

Global flags precede the command:

```powershell
.venv/Scripts/python.exe src/main.py --mode local --lexical-profile boosted-sum query "backup restoration" --method lexical
.venv/Scripts/python.exe src/main.py --mode local --lexical-profile best-fields --query-formulation identifier query "CP-9.1" --method hybrid
.venv/Scripts/python.exe src/main.py --mode local --lexical-profile boosted-sum --lexical-weight 2 --dense-weight 1 evaluate --methods lexical hybrid --output artifacts/local-boosted.json
.venv/Scripts/python.exe src/main.py --mode local ablate --output artifacts/local-field-ablation.json
.venv/Scripts/python.exe src/main.py --mode enterprise --env-file .env indexes --fields-only
.venv/Scripts/python.exe src/main.py --mode enterprise --env-file .env ablate --output artifacts/enterprise-field-ablation.json
.venv/Scripts/python.exe src/main.py --mode local --lexical-profile boosted-sum serve
```

Index creation and enterprise evaluation require the existing authorized private configuration. `indexes --fields-only` creates only `aintropy_nist_gemini_fields_v2` in the enterprise collection; legacy Atlas derives `<original_text_index>_fields_v2`. It checks definitions and waits for queryable readiness, reporting index setup seconds and zero document rewrites/embedding requests. `indexes` preserves its original behavior. The field index maps `control_id` as both a lowercase token and keyword string, `title`, `section_id`, and `text` as searchable strings, and existing filter/generation fields as tokens. No index is created implicitly by queries or ablations.

| Profile | Fields and weights | Combination |
| --- | --- | --- |
| `combined` | control_id + title + text in original combined representation | Original BM25/Atlas search_text |
| `content-only` | text 1 | Body/description only |
| `multi-field` | control_id 1, title 1, section_id 1, text 1 | Sum |
| `boosted-sum` | control_id 8, title 3, section_id 1.5, text 1 | Sum |
| `best-fields` | control_id 8, title 3, section_id 1.5, text 1 | Maximum weighted field score |

Local fields have separate document lengths and IDF, with the same BM25 constants as the original combined profile. Control IDs are single exact lowercase tokens, including dotted enhancement IDs. Atlas sums boosted native text clauses and exact token `equals` clauses. The Atlas ID clause uses a boosted constant equals score, whereas local ID matches use field BM25; numeric scores and field normalization across engines are not equivalent. Atlas `best-fields` searches each field independently, takes at most 100 candidates per field, then merges by maximum weighted score with deterministic chunk-ID ties. This bounded candidate union is not an exhaustive Elasticsearch `best_fields` implementation and can omit documents outside every field's candidate window.

`section_id` is the technical section identifier already present in each chunk. There is no invented human-readable section name. Release/source remain metadata filters, and hashes, URLs and vectors remain provenance/storage fields.

`--query-formulation identifier` extracts control IDs with a deterministic regex, normalizes case and restricts the lexical branch to exact detected IDs. Thus CP-9.1 cannot silently become CP-9; the dense branch continues using the original natural-language query. When no ID is detected, it behaves like raw formulation. Raw field profiles also score exact ID tokens, while default combined tokenization remains unchanged. Content-only still requires body terms to match. Release, source and explicit control filters apply to every lexical subquery and the dense branch.

Hybrid retrieval uses weighted RRF: `sum(weight / (60 + rank))`, with the original candidate pool `min(100, max(50, k * 5))`. Ties use chunk ID. Weights must be finite, nonnegative, with at least one positive value. A zero-weight hybrid branch performs no search or provider request. Direct lexical/dense calls remain available for isolated measurements regardless of hybrid weights.

CLI serving passes an immutable `RetrievalConfig` into the API. Python callers can use `create_app(..., config=RetrievalConfig(...))`. Query, health, answer and evaluation outputs disclose the static configuration and scoring distinctions. Request bodies do not mutate shared retrieval configuration.

`ablate` predeclares the preserved Level 1 combined baseline; Level 2 content-only, uniform multi-field, boosted sum, and boosted max; boosted sum with identifier formulation; and boosted sum with lexical/dense weights 2/1 and 1/2. It evaluates lexical and hybrid for each configuration plus one common dense baseline. Every measured method receives one untimed first-query warmup at pool 100, followed by one sequential timed pass over the unchanged benchmark. Ranked chunks are collapsed to unique release/control IDs using the original metrics. Each individual report includes Recall@10, MRR, NDCG@10, p95, detailed rankings, corpus/qrels hashes, usage, and warmup usage. There is no answer generation in this retrieval benchmark.

The main checkpoint lists every configuration as pending, running, complete or failed, writes after each configuration, and saves individual reports beside it. Failure records expose exception types without private provider messages, and later configurations continue. Reports include construction/setup time separately from warm latency; index build timing comes from the separate index-setup result. Existing reports are not relabeled or promoted. Use separate output paths for local and enterprise measured reports. Rerunning an ablation deliberately starts a fresh run; checkpointing preserves progress and failures but does not implement resume.

Root-produced measured artifacts are the evidence for actual results; this implementation document makes no measured ranking claims. The benchmark has small, independently unvalidated qrels. Results do not establish answer correctness or universal superiority, and no configuration becomes the new default. SPLADE and reranking remain future experiments.

Actual independent local and enterprise measurements, including regressions, are in [field-experiment-results.md](field-experiment-results.md). The grading recipe exercises a compact multi-field evaluation; the full ablation command remains available for reproduction.
