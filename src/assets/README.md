# Offline assets

`minilm/manifest.json` pins the real quantized ONNX model/tokenizer/configs and their SHA256 values. `minilm/LICENSE.txt` and `NOTICE.txt` preserve Apache-2.0 attribution.

`corpus-minilm.npz` contains genuinely computed float32 vectors for all 20,114 bundled NIST chunks, plus complete source/chunk/model identity metadata. See `corpus-minilm-manifest.json` and `docs/measurements/local-onnx-ingestion.json`. This is an embedding cache, without queries, expected rankings, answers or relevance labels. Every query is encoded by the bundled MiniLM model, and BM25/FAISS/RRF retrieve from the documents.

Rebuild: run `python src/main.py --mode local --store artifacts/runtime-local-rebuild ingest --cache missing-cache.npz`, then `python src/main.py --mode local --store artifacts/runtime-local-rebuild export-cache`. Update this asset manifest after rebuilding. Source changes invalidate cache identity and cause real encoding.
