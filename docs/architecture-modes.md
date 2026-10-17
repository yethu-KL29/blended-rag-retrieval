# Decision: enterprise architecture with offline grading

Primary deployment uses Gemini document/query embeddings, MongoDB Atlas lexical/vector search, reciprocal rank fusion, and bounded Gemini synthesis. Default grading uses bundled ONNX MiniLM, BM25, FAISS IndexFlatIP and RRF, followed by question-focused extractive sentence selection. Mode is explicit: ambient credentials never select enterprise, and enterprise failures never return local results.

Stores default to `artifacts/runtime-local` and `artifacts/runtime-enterprise`. Legacy LSA/full-precision MiniLM remain separate routes. Results identify mode, embedding model, vector engine and answer engine. Evaluation measures retrieval, not answer accuracy. Source-derived qrels are small and unvalidated. Historical reports retain their original meaning and are not measurements of the new modes.

## Local reproduction

```sh
python -m pip install -r requirements.txt
python src/main.py --mode local ingest
python src/main.py --mode local query "How are user accounts managed?" --release rev5
python src/main.py --mode local compare ac-1
python src/main.py --mode local evaluate --output artifacts/evaluation-local.json
python src/main.py --mode local serve
```

Application commands work without network, cloud keys, runtime downloads or PyTorch. Installation may need network. Local answers quote exact source sentences and supply citations. Their lexical relevance heuristic can abstain but is not calibrated and does not establish entailment.

MiniLM weights/tokenizer are bundled in `src/assets/minilm`. The SHA256 manifest pins upstream revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; the Apache-2.0 license is included. Git attributes preserve exact hashed asset bytes on Windows and Linux checkouts. This quantized ONNX representation differs from the historical full-precision baseline. Encoding uses attention-mask mean pooling, L2 normalization, length 256 and bounded CPU threads.

After genuine ONNX ingestion, export corpus vectors:

```sh
python src/main.py --mode local export-cache --output src/assets/corpus-minilm.npz
```

Ingestion uses this optional compressed cache only when the model fingerprint and every ordered complete chunk/source record match. Source/corpus mismatch triggers encoding; incompatible model or malformed vectors are rejected. Query embeddings and FAISS always run locally. Ingestion reports distinguish embedded and cached chunks. Incremental fixed-model ingestion reuses unchanged text/title/control vectors, replaces the incoming source/release, and preserves other releases. Failed preparation leaves the prior state pointer intact. Load only trusted locally generated joblib artifacts.

## Enterprise reproduction

Install `requirements-enterprise.txt` (google-genai 1.75.0). Copy `.env.example` to a private ignored file and provide the authorized POC URI/database plus either a Gemini API key or explicit Vertex/ADC settings. Never commit credentials. The dedicated collection is `aintropy_blended_nist_gemini768_v1` with its `_sources` collection; indexes are `aintropy_nist_gemini_text_v1` and `aintropy_nist_gemini_vector_v1`. Do not use old POC namespaces. No paid provisioning is performed.

This POC hard-enforces those three namespace names in enterprise mode. Set `GOOGLE_GENAI_USE_VERTEXAI=true`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` and `GOOGLE_APPLICATION_CREDENTIALS` for Vertex service-account/ADC use. An explicit env-file supports unquoted values, single-quoted literals, or JSON double-quoted values. For Windows paths use JSON escapes, for example `GOOGLE_APPLICATION_CREDENTIALS="C:\\private\\service-account.json"`. Files are never automatically loaded. Keep their names covered by ignore rules (`.env` is ignored).

```sh
python src/main.py --mode enterprise --env-file .env ingest
python src/main.py --mode enterprise --env-file .env indexes
python src/main.py --mode enterprise --env-file .env query "How are user accounts managed?" --release rev5
python src/main.py --mode enterprise --env-file .env --answer-engine extractive-nlp query "How are user accounts managed?"
python src/main.py --mode enterprise --env-file .env compare ac-1
python src/main.py --mode enterprise --env-file .env evaluate --output artifacts/evaluation-enterprise.json
python src/main.py --mode enterprise --env-file .env ingest
```

Defaults are `gemini-embedding-001` / 768 dimensions and `gemini-2.5-flash`; environment settings configure models/dimensions. Corpus embeddings use RETRIEVAL_DOCUMENT; queries use RETRIEVAL_QUERY. Changing model/dimensions requires a new store and reviewed fresh namespaces/indexes. Atlas generation pointers switch after successful writes; metadata filters select active generations before ranking. Wait for eventual index visibility before measuring.

Synthesis sees at most ten 2400-character excerpts, requests structured output and rejects unknown/inconsistent citation IDs. Empty evidence yields explicit insufficiency. Citation validation does not prove semantic entailment. Provider failures are sanitized without fallback. Counters record request attempts, submitted embedding characters, provider-reported billable characters when available, and actual generation tokens. Unknown billing metadata and Atlas/model USD prices remain unpriced. Reports include untimed warmup usage separately from timed retrieval usage. Generation is outside retrieval evaluation.

The API serializes complete query operations within each worker, including usage snapshots, retrieval and synthesis, so shared provider counters are attributed to exactly one request. The default server uses a single worker; concurrent queries queue behind its request lock. Each separately configured worker owns its own retriever, lock and usage counters.

## Preserved baselines

```sh
python src/main.py --store artifacts/runtime-lsa ingest --encoder lsa
python src/main.py --store artifacts/runtime-lsa evaluate --output artifacts/evaluation-lsa.json
python -m pip install -r requirements-neural.txt
python src/main.py --store artifacts/runtime-minilm ingest --encoder minilm
python src/main.py --store artifacts/runtime-minilm evaluate --output artifacts/evaluation-minilm.json
```

The full-precision baseline may download its explicitly selected pinned model. The official submission contract and historical reports remain intact; execution instructions now select the correct mode. [Independent measurements and verification](mode-benchmarks.md) cover both full-corpus modes, live Atlas/Gemini answers and the credential-free grading commands. Deterministic implementation tests include a genuine no-network ONNX encoder smoke test. Docker/Linux execution and production load remain untested.
