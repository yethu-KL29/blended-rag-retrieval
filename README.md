# Version-aware compliance Blended RAG

Retrieve cited NIST SP 800-53 controls from independent source snapshots, compare their exact text, and benchmark lexical, dense and reciprocal-rank fusion retrieval. MongoDB Atlas Search and Vector Search are the production adapter; bundled offline mode provides a reproducible development baseline.

Use Python 3.11–3.13 in an isolated virtual environment. Exact grader commands are in [run.yaml](run.yaml), run from this directory:

```sh
python -m pip install -r requirements.txt
python src/main.py ingest
python src/main.py query "How are user accounts managed?" --release rev5 --method hybrid -k 3
python src/main.py compare ac-1 --old rev4 --new rev5
python src/main.py evaluate --output artifacts/evaluation.json
python -m pytest src/tests -q
python src/main.py serve
```

The API exposes `POST /query`, `GET /compare/{control_id}`, `GET /health` and `/docs`. Query body: `{"question":"account management","method":"hybrid","k":5,"filters":{"release":"rev5"}}`. Filters also accept `source_id` and `control_id`; unknown fields are rejected. Responses contain exact excerpts, source URLs, control/section IDs, releases and hashes. Answers are extractive evidence; no LLM synthesis or automatic entailment verification is performed.

Bundled gzip catalogs are public NIST Rev4 (catalog version `2015-01-22`) and Rev5 (`5.2.0`, including assessment procedures), pinned to one upstream commit and checked against SHA256 on ingestion. `python src/main.py download` explicitly redownloads those immutable URLs and verifies hashes. No credentials or corpus download are required for default execution.

Default dense vectors are **TF-IDF + 64-dimensional SVD (LSA)**, a fitted local baseline. Optional neural semantic retrieval uses pinned MiniLM:

```sh
python -m pip install -r requirements-neural.txt
python src/main.py --store artifacts/runtime-minilm ingest --encoder minilm
python src/main.py --store artifacts/runtime-minilm query "Remove access for departing employees" --release rev5
python src/main.py --store artifacts/runtime-minilm evaluate --output artifacts/evaluation-minilm.json
```

MiniLM downloads its pinned model on first use. Its execution and quality are unvalidated in the included report. Use separate stores/Atlas collections for different embedding models. Unchanged ingestion embeds zero chunks; changed LSA corpora refit all vectors, while MiniLM reuses unchanged chunk vectors.

For Atlas, export the variables shown in [.env.example](.env.example) in your shell, run offline ingestion to prepare the encoder, then run:

```sh
python src/main.py --backend atlas indexes
python src/main.py --backend atlas ingest
python src/main.py --backend atlas query "ac-2 account management" --release rev5
python src/main.py --backend atlas evaluate --output artifacts/evaluation-atlas.json
```

Index setup waits for queryable readiness. Search ingestion remains eventually consistent; wait for evidence before benchmarking. Atlas failures never silently use offline results. Set `RUN_ATLAS_TESTS=1` and a dedicated collection name containing `test` to enable the gated integration test. No live Atlas validation is claimed. `.env` is an example only and is not automatically loaded.

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
