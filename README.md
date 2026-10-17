# Version-aware compliance Blended RAG

Retrieve cited NIST SP 800-53 controls from independent source snapshots, compare their exact text, and benchmark lexical, dense and reciprocal-rank fusion retrieval. The primary research implementation is local indexed BM25 + bundled ONNX MiniLM/FAISS + explicitly optional SPLADE. MongoDB Atlas/Gemini remains the cloud hybrid fallback. Default grading uses BM25/MiniLM hybrid and quoted question-focused NLP answers with no credentials, runtime downloads or PyTorch.

Review the [requirement-to-evidence and test guide](docs/grading-guide.md) and [indexing, retrieval and deployment diagrams](docs/system-architecture.md). These identify what the default commands execute and which optional checks require additional setup.

For a complete source-grounded explanation, read [the end-to-end implementation guide](docs/END_TO_END_IMPLEMENTATION.md). It covers every pipeline stage, what was fitted versus pretrained, disk indexes and real query execution, answers, evaluation formulas, operations, exact commands and known limits.

See [the execution-mode architecture decision and exact local, baseline, cache-export and enterprise commands](docs/architecture-modes.md). Use `--mode local` for grading and `--mode enterprise --env-file .env` for optional private cloud configuration. Stores, model fingerprints, usage and reports remain distinct. [Current local and live enterprise results](docs/mode-benchmarks.md) are reported separately from preserved historical baselines.

[Field-aware experiment results](docs/field-experiment-results.md) compare content-only, separate-field, boosted and max-field retrieval in both modes; the combined default remains unchanged.

Use Python 3.11–3.13 in an isolated virtual environment. Exact grader commands are in [run.yaml](run.yaml), run from this directory:

```sh
python -m pip install -r requirements.txt
python src/main.py --mode local ingest
python src/main.py --mode local query "How are user accounts managed?" --release rev5 --method hybrid -k 3
python src/main.py --mode local compare ac-1 --old rev4 --new rev5
python src/main.py --mode local evaluate --output artifacts/evaluation-local.json
python -m pytest src/tests -q
python src/main.py --mode local serve
```

The API exposes `POST /query`, `GET /compare/{control_id}`, `GET /health` and `/docs`. Query body: `{"question":"account management","method":"hybrid","k":5,"filters":{"release":"rev5"}}`. Filters also accept `source_id` and `control_id`; unknown fields are rejected. Responses include provenance, source URLs, IDs, releases and hashes. Local answers quote source sentences using an uncalibrated relevance heuristic. Enterprise defaults to bounded Gemini synthesis with citation-ID validation; semantic entailment remains unvalidated.

Bundled gzip catalogs are public NIST Rev4 (catalog version `2015-01-22`) and Rev5 (`5.2.0`, including assessment procedures), pinned to one upstream commit and checked against SHA256 on ingestion. `python src/main.py download` explicitly redownloads those immutable URLs and verifies hashes. No credentials or corpus download are required for default execution.

Default dense vectors are bundled **ONNX quantized MiniLM, 384 dimensions** with FAISS. Historical **TF-IDF + 64-dimensional SVD (LSA)** remains available through `--store artifacts/runtime-lsa ingest --encoder lsa`. The optional historical full-precision MiniLM baseline uses:

```sh
python -m pip install -r requirements-neural.txt
python src/main.py --store artifacts/runtime-minilm ingest --encoder minilm
python src/main.py --store artifacts/runtime-minilm query "Remove access for departing employees" --release rev5
python src/main.py --store artifacts/runtime-minilm evaluate --output artifacts/evaluation-minilm.json
```

This optional full-precision MiniLM baseline downloads its pinned model on first use; the default ONNX model is bundled. The full-precision baseline has been executed locally on all 20,114 chunks; the small benchmark's labels remain unvalidated. See `docs/experiment-summary.json` for the comparison and `docs/measurements/` for complete compressed rankings. Use separate stores/Atlas collections for different embedding models. Unchanged ingestion embeds zero chunks; changed LSA corpora refit all vectors, while MiniLM reuses unchanged chunk vectors.

Enterprise setup uses the dedicated Gemini collection and indexes listed in [architecture-modes.md](docs/architecture-modes.md). Install the optional dependencies and explicitly load private configuration:

```sh
python -m pip install -r requirements-enterprise.txt
python src/main.py --mode enterprise --env-file .env ingest
python src/main.py --mode enterprise --env-file .env indexes
python src/main.py --mode enterprise --env-file .env query "How are user accounts managed?" --release rev5
python src/main.py --mode enterprise --env-file .env evaluate --output artifacts/evaluation-enterprise.json
python src/main.py --mode enterprise --env-file .env serve
```

The cloud fallback uses `gemini-embedding-001` (768 dimensions), Atlas BM25/vector retrieval and client RRF. The answer model is configurable through `GEMINI_CHAT_MODEL`; the historical live POC used `gemini-3.1-flash-lite`. Local answers use extractive NLP and require no generative model. Enterprise failures remain enterprise errors; no automatic local substitution occurs.

**SPLADE is optional and disabled by default** (`sparse_weight=0`; default `hybrid` searches only BM25 + MiniLM/FAISS). Real SPLADE vectors for **12,672 of 20,114 chunks** were saved before the CPU-resource-limited full indexing run was stopped. Full-corpus SPLADE indexing and benchmarking remain unfinished; partial-corpus evidence is identified separately in the [end-to-end guide](docs/END_TO_END_IMPLEMENTATION.md). The default full-corpus application and `run.yaml` do not depend on SPLADE.

Optional SPLADE has a separate explicit setup, index and retrieval path. Install its dependencies only in an optional environment (CPU Torch can be installed from the PyTorch CPU package index):

```sh
python -m pip install -r requirements-sparse.txt
python src/main.py --mode local sparse-setup
python src/main.py --mode local sparse-index --batch-size 8 --checkpoint-size 128 --threads 4
python src/main.py --mode local query "Remove access for departing employees" --method sparse --release rev5
python src/main.py --mode local --sparse-weight 1 query "Remove access for departing employees" --method blended --release rev5
python src/main.py --mode local sparse-inspect "repair damaged information"
python src/main.py --mode local sparse-evaluate --output artifacts/sparse-evaluation-local.json
```

Only `sparse-setup` downloads weights. Indexing and queries require the pinned local cache; missing dependencies, weights or ready index produce errors. Repeating `sparse-index` resumes completed checkpoints for identical source/model inputs; a ready index is published atomically. Inference batches and persisted checkpoints have separate sizes. Sparse setup is CPU intensive, uses approximately 440 MB of model weights plus runtime memory, and indexing the full corpus may take substantial time. The default `sparse_weight=0` keeps existing hybrid behavior; `blended` requires positive sparse weight. API `method` accepts `sparse`/`blended`; server weights come from CLI options before `serve`. Atlas rejects both methods.

SPLADE uses [`naver/splade-cocondenser-ensembledistil`](https://huggingface.co/naver/splade-cocondenser-ensembledistil) at revision `49cf4c7b0db5b870a401ddf5e2669993ef3699c7`, attributed to NAVER under **CC-BY-NC-SA-4.0**. Its noncommercial/share-alike terms apply to the optional model. It encodes title + control ID + text, truncated at 256 model tokens, into 30,522 raw nonnegative learned vocabulary weights. Ranking uses sparse dot products without dense normalization. This differs from field-aware BM25. See [local sparse architecture and limits](docs/local-splade.md). Historical measurements remain historical; small unvalidated qrels do not justify a default change or an exact IBM replication claim.

See [docs/README.md](docs/README.md) for architecture, measurements and limits. Docker: `docker build -t compliance-rag .` then `docker run --rm -p 8000:8000 compliance-rag`.

---

Original assessment submission contract follows.

## Submission instructions

1. Install the **AIntropy Grader Bot** GitHub App (https://github.com/apps/aintropy-grader-bot) and grant it access to your fork, if you haven't already — we can't evaluate your submission without this.
2. Put your code under `src/`.
3. Put your write-up under `docs/` — markdown, slides, a notebook, or another format of your choosing are all fine, as long as it's committed in this repo.
4. Fill in `run.yaml` with the **exact** commands needed to set up and run your code. We run these commands exactly as written to evaluate your submission — this is the one part of the submission we read literally, not loosely: if `run.yaml` doesn't work, we can't see your code run.
5. Push to your own fork's default branch any time during your test's time limit. No pull request is needed or expected.

Once you start the test you have until your test's time limit to complete the assignment — see your invitation email for the exact duration. We ignore any commits pushed to your repo after that.

Questions? careers@aintropy.ai

Field-aware lexical profiles, weighted fusion and separate local/enterprise ablations: [commands and scoring details](docs/field-aware-retrieval.md).
