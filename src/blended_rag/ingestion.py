from __future__ import annotations

import json
import os
import re
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .models import Chunk, digest


def parse_catalog(source: dict, document: dict, max_words: int = 180) -> list[Chunk]:
    if max_words < 20:
        raise ValueError("max_words must be at least 20")
    chunks = []

    def parts(items, parent=""):
        for i, part in enumerate(items):
            section = part.get("id") or f"{parent}/{part.get('name', 'prose')}[{i}]"
            if part.get("prose"):
                yield section, part["prose"]
            yield from parts(part.get("parts", []), section)

    def walk(node):
        for control in node.get("controls", []):
            cid = control["id"]
            definitions = []
            for param in control.get("params", []):
                description = param.get("label", "")
                selection = param.get("select", {})
                if selection:
                    description += " Selection " + selection.get("how-many", "unspecified") + ": " + "; ".join(selection.get("choice", []))
                definitions.append(("parameter:" + param["id"], "Organization-defined parameter " + param["id"] + ": " + (description or "unspecified value")))
            sections = list(parts(control.get("parts", [])))
            if not sections:
                metadata = {key: control[key] for key in ("id", "class", "title", "props", "links") if key in control}
                sections.append((cid + "_metadata", "Control metadata: " + json.dumps(metadata, sort_keys=True)))
            for section, prose in [*sections, *definitions]:
                # OSCAL prose contains Markdown. Keep it, including parameter references.
                words = re.sub(r"\s+", " ", prose).strip().split()
                for offset in range(0, len(words), max_words):
                    text = " ".join(words[offset:offset + max_words])
                    sid = f"{section}:{offset // max_words}"
                    key = f"{source['source_id']}:{source['release']}:{cid}:{sid}"
                    chunks.append(Chunk(digest(key), source["source_id"], source["release"],
                                        cid, sid, control["title"], text, source["url"],
                                        source["sha256"], digest(text)))
            walk(control)
        for group in node.get("groups", []):
            walk(group)

    walk(document["catalog"])
    return chunks


class Encoder:
    """Local latent semantic analysis: TF-IDF + fitted SVD, NOT a neural embedding."""
    name = "tfidf-lsa-64-v1"

    def fit(self, chunks: list[Chunk]):
        self.vectorizer = TfidfVectorizer(lowercase=True, sublinear_tf=True,
                                          token_pattern=r"(?u)\b[\w-]+\b", max_features=30000)
        matrix = self.vectorizer.fit_transform([c.title + " " + c.control_id + " " + c.text for c in chunks])
        self.svd = TruncatedSVD(n_components=min(64, max(1, matrix.shape[0] - 1), max(1, matrix.shape[1] - 1)), random_state=42)
        self.svd.fit(matrix)
        self.dimensions = self.svd.n_components
        self.fingerprint = digest(self.name + "|" + "|".join(c.chunk_id + digest(c.title) + c.content_hash for c in chunks))
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = normalize(self.svd.transform(self.vectorizer.transform(texts)))
        if vectors.shape != (len(texts), self.dimensions) or not np.isfinite(vectors).all():
            raise ValueError("Embedding count, dimensions or finite-value validation failed")
        return vectors


class NeuralEncoder:
    """Optional pinned MiniLM; installed/downloaded only when explicitly selected."""
    name = "sentence-transformers/all-MiniLM-L6-v2"
    revision = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    dimensions = 384
    fingerprint = digest(name + "@" + revision + "|normalize=true")

    def fit(self, chunks):
        return self

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        pass

    def encode(self, texts):
        if not hasattr(self, "_model"):
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ValueError("Neural mode requires: pip install -r requirements-neural.txt") from None
            self._model = SentenceTransformer(self.name, revision=self.revision, device="cpu")
        vectors = self._model.encode(texts, batch_size=64, normalize_embeddings=True,
                                     show_progress_bar=False, convert_to_numpy=True)
        if vectors.shape != (len(texts), self.dimensions) or not np.isfinite(vectors).all():
            raise ValueError("Embedding count, dimensions or finite-value validation failed")
        return vectors


class LocalStore:
    def __init__(self, directory: str | Path = "artifacts/runtime"):
        self.directory = Path(directory)
        self.path = self.directory / "store.json"

    def read(self) -> dict:
        if not self.path.exists():
            raise ValueError("No local index; run: python src/main.py ingest")
        return json.loads(self.path.read_text(encoding="utf-8"))

    def ingest(self, chunks: list[Chunk], encoder_kind="lsa") -> dict:
        start = time.perf_counter()
        if not chunks or len({c.chunk_id for c in chunks}) != len(chunks):
            raise ValueError("Ingestion needs unique nonempty chunks")
        self.directory.mkdir(parents=True, exist_ok=True)
        previous = self.read() if self.path.exists() else None
        encoder_class = {"lsa": Encoder, "minilm": NeuralEncoder}.get(encoder_kind)
        if encoder_class is None:
            raise ValueError("encoder must be lsa or minilm")
        if previous and (previous["model_name"] != encoder_class.name or
                         (encoder_kind == "minilm" and previous["model_fingerprint"] != encoder_class.fingerprint)):
            raise ValueError("Embedding model mismatch; use a new --store directory")
        old = {c["chunk_id"]: c for c in previous["chunks"]} if previous else {}
        incoming_sources = {(c.source_id, c.release) for c in chunks}
        incoming = {c.chunk_id: asdict(c) for c in chunks}
        retained = {k: c for k, c in old.items() if (c["source_id"], c["release"]) not in incoming_sources}
        merged = {**retained, **incoming}
        if previous and merged == old:
            return {"chunks": len(old), "changed": False, "embedded_chunks": 0,
                    "elapsed_seconds": time.perf_counter() - start}
        ordered = [Chunk(**c) for _, c in sorted(merged.items())]
        # Fitted LSA is corpus-dependent. Every changed corpus gets its own model generation.
        encoder = encoder_class().fit(ordered)
        if encoder_kind == "minilm" and previous:
            cached = {c["chunk_id"]: (c, v) for c, v in zip(previous["chunks"], previous["vectors"], strict=True)}
            vectors = [None] * len(ordered)
            pending = []
            for i, c in enumerate(ordered):
                prior = cached.get(c.chunk_id)
                if prior and all(prior[0][key] == getattr(c, key) for key in ["content_hash", "title", "control_id"]):
                    vectors[i] = prior[1]
                else:
                    pending.append(i)
            prepared = encoder.encode([ordered[i].title + " " + ordered[i].control_id + " " + ordered[i].text for i in pending]) if pending else []
            for i, vector in zip(pending, prepared, strict=True):
                vectors[i] = vector
            embedded_count = len(pending)
            vectors = np.array(vectors)
        else:
            vectors = encoder.encode([c.title + " " + c.control_id + " " + c.text for c in ordered])
            embedded_count = len(ordered)
        if vectors.shape != (len(ordered), encoder.dimensions) or not np.isfinite(vectors).all():
            raise ValueError("Embedding count, dimensions or finite-value validation failed")
        model_file = f"encoder-{encoder.fingerprint}.joblib"
        model_path = self.directory / model_file
        joblib.dump(encoder, model_path)
        state = {"model_name": encoder.name, "model_fingerprint": encoder.fingerprint,
                 "model_file": model_file, "dimensions": encoder.dimensions,
                 "chunks": [asdict(c) for c in ordered], "vectors": vectors.tolist()}
        # Atomic pointer/state replacement; old store remains usable on preparation failures.
        handle, temporary = tempfile.mkstemp(dir=self.directory, suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as out:
                json.dump(state, out, separators=(",", ":"))
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        elapsed = time.perf_counter() - start
        return {"chunks": len(ordered), "changed": True, "embedded_chunks": embedded_count,
                "elapsed_seconds": elapsed, "chunks_per_second": len(ordered) / elapsed,
                "store_bytes": self.path.stat().st_size, "model_bytes": model_path.stat().st_size}

    def encoder(self):
        state = self.read()
        # Only load locally generated trusted artifacts, never a user upload.
        encoder = joblib.load(self.directory / state["model_file"])
        if encoder.fingerprint != state["model_fingerprint"] or encoder.name != state["model_name"]:
            raise ValueError("Embedding model mismatch")
        return encoder
