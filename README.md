# Version-aware compliance Blended RAG

Retrieve cited NIST SP 800-53 controls from independent source snapshots, compare their exact text, and benchmark lexical, dense and reciprocal-rank fusion retrieval. Primary enterprise mode uses Atlas Search/Vector Search, Gemini embeddings and cited Gemini synthesis. Default grading mode uses bundled ONNX MiniLM, real FAISS, BM25/RRF and quoted question-focused NLP answers with no credentials, runtime downloads or PyTorch.

See [the execution-mode architecture decision and exact local, baseline, cache-export and enterprise commands](docs/architecture-modes.md). Use `--mode local` for grading and `--mode enterprise --env-file .env` for optional private cloud configuration. Stores, model fingerprints, usage and reports remain distinct. [Current local and live enterprise results](docs/mode-benchmarks.md) are reported separately from preserved historical baselines.

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

MiniLM downloads its pinned model on first use. It has now been executed locally on all 20,114 chunks; the small benchmark's labels remain unvalidated. See `docs/experiment-summary.json` for the comparison and `docs/measurements/` for complete compressed rankings. Use separate stores/Atlas collections for different embedding models. Unchanged ingestion embeds zero chunks; changed LSA corpora refit all vectors, while MiniLM reuses unchanged chunk vectors.

Enterprise setup uses the dedicated Gemini collection and indexes listed in [architecture-modes.md](docs/architecture-modes.md). Install the optional dependencies and explicitly load private configuration:

```sh
python -m pip install -r requirements-enterprise.txt
python src/main.py --mode enterprise --env-file .env ingest
python src/main.py --mode enterprise --env-file .env indexes
python src/main.py --mode enterprise --env-file .env query "How are user accounts managed?" --release rev5
python src/main.py --mode enterprise --env-file .env evaluate --output artifacts/evaluation-enterprise.json
python src/main.py --mode enterprise --env-file .env serve
```

The primary solution uses `gemini-embedding-001` (768 dimensions), Atlas BM25/vector retrieval and client RRF. The answer model is configurable through `GEMINI_CHAT_MODEL`; the live POC uses `gemini-3.1-flash-lite`. Local answers use extractive NLP and require no generative model. Enterprise failures remain enterprise errors; no automatic local substitution occurs.

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
