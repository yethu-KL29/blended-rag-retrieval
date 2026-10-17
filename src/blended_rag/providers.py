"""Pinned offline encoder and explicit Gemini provider. No implicit downloads."""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from pathlib import Path

import numpy as np

from .models import digest

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "minilm"


class ProviderError(RuntimeError):
    """Sanitized cloud error, safe for CLI/API responses."""


class OnnxEncoder:
    name = "all-MiniLM-L6-v2-onnx-quint8"
    dimensions = 384
    fixed_model = True

    def __init__(self):
        manifest = json.loads((ASSETS / "manifest.json").read_text())
        for item in manifest["files"]:
            path = ASSETS / item["file"]
            if digest(path.read_bytes()) != item["sha256"]:
                raise ValueError("Bundled MiniLM asset hash mismatch")
        self.fingerprint = digest(json.dumps(manifest, sort_keys=True) + "|pool=attention-mean|normalize=l2|max_length=256")

    def fit(self, chunks):
        return self

    def __getstate__(self):
        return {"fingerprint": self.fingerprint}

    def __setstate__(self, state):
        self.__init__()
        if state["fingerprint"] != self.fingerprint:
            raise ValueError("Bundled MiniLM model mismatch")

    def encode(self, texts):
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        if not hasattr(self, "_session"):
            import onnxruntime as ort
            from tokenizers import Tokenizer
            options = ort.SessionOptions()
            options.intra_op_num_threads = min(4, os.cpu_count() or 1)
            options.inter_op_num_threads = 1
            self._session = ort.InferenceSession(str(ASSETS / "model.onnx"), sess_options=options,
                                                providers=["CPUExecutionProvider"])
            self._tokenizer = Tokenizer.from_file(str(ASSETS / "tokenizer.json"))
            self._tokenizer.enable_truncation(max_length=256)
            self._tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        vectors = []
        for offset in range(0, len(texts), 32):
            batch = self._tokenizer.encode_batch(texts[offset:offset + 32])
            feed = {"input_ids": np.asarray([x.ids for x in batch], dtype=np.int64),
                    "attention_mask": np.asarray([x.attention_mask for x in batch], dtype=np.int64),
                    "token_type_ids": np.asarray([x.type_ids for x in batch], dtype=np.int64)}
            hidden = self._session.run(None, {x.name: feed[x.name] for x in self._session.get_inputs()})[0]
            mask = feed["attention_mask"][..., None]
            pooled = (hidden * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)
            pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
            vectors.extend(pooled)
        result = np.asarray(vectors, dtype=np.float32)
        if result.shape != (len(texts), self.dimensions) or not np.isfinite(result).all():
            raise ValueError("Invalid MiniLM vectors")
        return result


class GeminiEncoder:
    fixed_model = True

    def __init__(self, model=None, dimensions=None):
        self.name = model or os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
        self.dimensions = int(dimensions or os.getenv("GEMINI_EMBEDDING_DIMENSIONS", "768"))
        self.fingerprint = digest(self.name + f"|dimensions={self.dimensions}|document/query|normalize=l2")
        self.usage = {"embedding_requests": 0, "embedding_characters": 0, "embedding_billable_characters": 0,
                      "generation_requests": 0, "input_tokens": 0, "output_tokens": 0,
                      "thought_tokens": 0, "total_tokens": 0}

    def fit(self, chunks):
        return self

    def __getstate__(self):
        return {"model": self.name, "dimensions": self.dimensions}

    def __setstate__(self, state):
        self.__init__(state["model"], state["dimensions"])

    def client(self):
        if not hasattr(self, "_client"):
            try:
                from google import genai
                from google.genai import types
                options = types.HttpOptions(timeout=60000, retry_options=types.HttpRetryOptions(attempts=1))
                if os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in {"1", "true"}:
                    self._client = genai.Client(vertexai=True, project=os.getenv("GOOGLE_CLOUD_PROJECT"),
                                                location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"), http_options=options)
                else:
                    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                    if not key:
                        raise ProviderError("Enterprise Gemini requires an API key or explicit Vertex/ADC configuration")
                    self._client = genai.Client(api_key=key, http_options=options)
            except ProviderError:
                raise
            except Exception:
                raise ProviderError("Gemini client setup failed; install requirements-enterprise.txt and check private configuration") from None
        return self._client

    def encode(self, texts, task="RETRIEVAL_DOCUMENT"):
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        client = self.client()
        lock = Lock()

        def embed_batch(batch):
            for attempt in range(3):
                with lock:
                    self.usage["embedding_requests"] += 1
                    self.usage["embedding_characters"] += sum(map(len, batch))
                try:
                    response = client.models.embed_content(model=self.name, contents=batch,
                        config={"task_type": task, "output_dimensionality": self.dimensions})
                    metadata = getattr(response, "metadata", None)
                    # Vertex reports billable characters; absent provider billing metadata stays unknown.
                    billed = getattr(metadata, "billable_character_count", None)
                    with lock:
                        if billed is not None:
                            self.usage["embedding_billable_characters"] += billed
                        else:
                            self.billing_metadata_unavailable = True
                    if len(response.embeddings) != len(batch):
                        raise ProviderError("Gemini returned incorrect embedding count")
                    return [entry.values for entry in response.embeddings]
                except Exception:
                    if attempt == 2:
                        raise ProviderError("Enterprise Gemini embedding failed; no local substitution was performed") from None
                    time.sleep(.5 * (2 ** attempt))
        batches = [texts[i:i + 32] for i in range(0, len(texts), 32)]
        # map preserves order even when bounded concurrent requests finish out of order.
        with ThreadPoolExecutor(max_workers=4) as executor:
            vectors = [vector for batch in executor.map(embed_batch, batches) for vector in batch]
        result = np.asarray(vectors, dtype=np.float32)
        if result.shape != (len(texts), self.dimensions) or not np.isfinite(result).all():
            raise ProviderError("Gemini returned invalid embedding dimensions or values")
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ProviderError("Gemini returned zero embedding")
        return result / norms

    def encode_query(self, text):
        return self.encode([text], task="RETRIEVAL_QUERY")[0]


def usage_snapshot(retriever):
    encoder = getattr(retriever, "encoder", None)
    usage = dict(getattr(encoder, "usage", {}))
    if getattr(encoder, "billing_metadata_unavailable", False):
        usage["embedding_billable_characters"] = None
    return usage


def usage_delta(before, after):
    return {key: (None if after.get(key) is None or before.get(key, 0) is None
                  else after.get(key, 0) - before.get(key, 0)) for key in after}


def provenance(retriever):
    from .retrieval import RetrievalConfig
    atlas = getattr(retriever, "vector_engine", "") == "mongodb-atlas-vector-search"
    return {"lexical_scoring": ("Atlas native text field statistics; exact control_id uses boosted equals constant scores; best-fields uses bounded candidate union"
                                if atlas else "Indexed BM25 postings: combined statistics preserved; other profiles use independent per-field lengths and IDF, exact-token control_id"),
            "field_search_index": getattr(retriever, "field_index", None),
            "retrieval_config": getattr(retriever, "config", RetrievalConfig()).metadata(),
            "sparse_model": getattr(getattr(retriever, "sparse_encoder", None), "identity", None),
            "sparse_index": getattr(retriever, "sparse_metadata", None),
            "mode": getattr(retriever, "mode", "legacy"), "embedding_model": retriever.model,
            "embedding_provider": "gemini" if isinstance(getattr(retriever, "encoder", None), GeminiEncoder) else "local",
            "vector_engine": getattr(retriever, "vector_engine", "numpy-exact-cosine"),
            "answer_engine": getattr(retriever, "answer_engine", "extractive-nlp"),
            "billing_note": "Submitted characters counted; billable characters available only when provider reports metadata; USD unpriced"}
