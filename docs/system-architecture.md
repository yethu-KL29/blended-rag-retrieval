# Retrieval architecture

Local retrieval is primary. MongoDB Atlas/Gemini is an explicitly configured hybrid fallback. All diagrams describe the implemented boundaries; an optional component requires its documented setup before use.

## Indexing and persistence

```mermaid
flowchart TD
  C[Bundled pinned NIST catalogs] --> H[Verify SHA256 and parse controls]
  H --> P[Section-preserving chunks with release, IDs and source hashes]
  P --> L[Local versioned chunk store]
  L --> BM[Field tokenization and BM25 weighted postings]
  L --> MC[Validate bundled MiniLM corpus cache]
  MC -->|identity matches| V[384-dimensional normalized vectors]
  MC -->|cache absent or source changed| ON[Bundled ONNX MiniLM encoding]
  ON --> V
  V --> F[Persist FAISS IndexFlatIP]
  L -->|explicit optional indexing| SP[Pinned SPLADE document encoder]
  SP --> CK[Hashed CSR checkpoints]
  CK --> SR[Assemble sparse matrix and atomically publish ready manifest]
  P -->|explicit cloud ingest| GE[Gemini document embeddings]
  GE --> AT[Dedicated Atlas collection]
  AT --> AI[Atlas lexical and vector indexes]
```

BM25 stores separate CSC matrices for combined text and individual fields so a term query follows its postings. SPLADE stores weighted learned-vocabulary vectors as a CSR matrix and ranks with sparse dot products. These are different representations. Source/model identities prevent reusing an index for incompatible chunks or encoders. Neither FAISS nor ordinary Atlas text indexes score SPLADE vectors in this implementation.

## Local query execution

```mermaid
flowchart TD
  Q[Question and validated metadata filters] --> CFG[Explicit method, field strategy and weights]
  CFG --> B[BM25 term-posting scores]
  CFG --> D[ONNX MiniLM query vector and FAISS scores]
  CFG -->|optional cached model and ready index| S[SPLADE query vector and sparse dot products]
  B --> BF[Filter before branch top-k]
  D --> DF[Filter before branch top-k]
  S --> SF[Filter before branch top-k]
  BF --> R[Direct ranking or weighted reciprocal rank fusion]
  DF --> R
  SF --> R
  R --> E[Ranked evidence with stable IDs and provenance]
  E --> A[Select exact source sentences and append citations]
  E --> EV[Separate retrieval metrics against frozen qrels]
```

`lexical`, `dense`, and `sparse` return their direct rankings. `hybrid` uses BM25+dense; `blended` additionally uses SPLADE with positive sparse weight. Zero-weight branches are skipped. Ranking ties are deterministic. Retrieval evaluation does not invoke answer generation. Version comparison follows a separate exact-text comparison path over the selected releases.

## CLI/API and cloud boundary

```mermaid
flowchart LR
  U[CLI or API caller] --> APP[main.py or FastAPI]
  APP --> MODE{Explicit mode}
  MODE -->|local default| LOC[Local stores and CPU encoders]
  LOC --> NLP[Extractive cited answer]
  MODE -->|enterprise and private configuration| CL[Atlas BM25 and Gemini dense]
  CL --> RRF[Two-way client RRF]
  RRF --> GEN[Bounded Gemini synthesis and citation-ID validation]
  APP --> CMP[Version comparison]
  APP --> MEAS[Separate per-mode evaluation reports]
```

The single-worker API serializes query operations to keep shared provider usage counters attributable. No automatic cloud-to-local failover occurs. Atlas rejects sparse/three-way methods. Model setup/download is explicit; normal cached SPLADE inference does not download weights. Default local runtime has no network dependency after package installation.

The Dockerfile serves the default local application. Docker/Linux runtime and production concurrent capacity remain unverified. An ingestion queue such as Service Bus is a possible production extension, not part of the demonstrated query path. See [execution/configuration](architecture-modes.md), [sparse limits](local-splade.md), and [requirement evidence](grading-guide.md).
