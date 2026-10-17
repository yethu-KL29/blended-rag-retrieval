from __future__ import annotations

import math
from collections import Counter
from typing import Protocol

import numpy as np

from .ingestion import LocalStore
from .models import Chunk, Hit, digest, tokens, validate_filters, validate_query


class Retriever(Protocol):
    def search(self, query: str, method: str = "hybrid", k: int = 10, filters: dict | None = None) -> list[Hit]: ...


def rrf(rankings: list[list[Hit]], k: int, constant: int = 60) -> list[Hit]:
    scores, chunks = {}, {}
    for ranking in rankings:
        seen = set()
        for rank, hit in enumerate(ranking, 1):
            key = hit.chunk.chunk_id
            if key in seen:
                continue
            seen.add(key)
            scores[key] = scores.get(key, 0.0) + 1 / (constant + rank)
            chunks[key] = hit.chunk
    return [Hit(chunks[key], score) for key, score in
            sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]]


class LocalRetriever:
    mode = "legacy"
    vector_engine = "numpy-exact-cosine"
    answer_engine = "extractive-nlp"
    def __init__(self, store: LocalStore):
        state = store.read()
        self.chunks = [Chunk(**c) for c in state["chunks"]]
        self.vectors = np.array(state["vectors"])
        self.encoder = store.encoder()
        self.model = {"name": state["model_name"], "fingerprint": state["model_fingerprint"]}
        self.counts = [Counter(tokens(c.title + " " + c.control_id + " " + c.text)) for c in self.chunks]
        self.lengths = np.array([sum(c.values()) for c in self.counts])
        self.average_length = float(self.lengths.mean())
        df = Counter(term for counts in self.counts for term in counts)
        self.idf = {term: math.log(1 + (len(self.chunks) - count + .5) / (count + .5)) for term, count in df.items()}

    def search(self, query, method="hybrid", k=10, filters=None):
        validate_query(query, k)
        filters = validate_filters(filters)
        if method not in {"lexical", "dense", "hybrid"}:
            raise ValueError("method must be lexical, dense or hybrid")
        if method == "hybrid":
            pool = max(50, k * 5)
            return rrf([self.search(query, "lexical", min(pool, 100), filters),
                        self.search(query, "dense", min(pool, 100), filters)], k)
        selected = [i for i, c in enumerate(self.chunks) if all(getattr(c, key) == val for key, val in filters.items())]
        if method == "dense":
            vector = self.encoder.encode([query])[0]
            scores = self.vectors @ vector
        else:
            scores = np.zeros(len(self.chunks))
            for term in set(tokens(query)):
                for i in selected:
                    freq = self.counts[i].get(term, 0)
                    if freq:
                        scores[i] += self.idf.get(term, 0) * (freq * 2.5) / (freq + 1.5 * (.25 + .75 * self.lengths[i] / self.average_length))
        indices = sorted((i for i in selected if scores[i] > 1e-8), key=lambda i: (-scores[i], self.chunks[i].chunk_id))[:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in indices]


class FaissRetriever(LocalRetriever):
    """Modern local mode: genuine FAISS; metadata filtering precedes top-k."""
    mode = "local"
    vector_engine = "faiss-IndexFlatIP"

    def __init__(self, store):
        super().__init__(store)
        import faiss
        state = store.read()
        meta = state.get("faiss", {})
        if (meta.get("model_fingerprint") != self.encoder.fingerprint or
            meta.get("dimensions") != self.encoder.dimensions or
            meta.get("chunk_ids") != [c.chunk_id for c in self.chunks]):
            raise ValueError("FAISS artifact/model mismatch; ingest local mode in its own store")
        path = store.directory / meta["file"]
        if digest(path.read_bytes()) != meta["sha256"]:
            raise ValueError("FAISS artifact hash mismatch")
        self.index = faiss.read_index(str(path))
        if self.index.d != self.encoder.dimensions or self.index.ntotal != len(self.chunks):
            raise ValueError("FAISS artifact count/dimensions mismatch")

    def search(self, query, method="hybrid", k=10, filters=None):
        if method != "dense":
            return super().search(query, method, k, filters)
        import faiss
        validate_query(query, k)
        filters = validate_filters(filters)
        selected = [i for i, c in enumerate(self.chunks) if all(getattr(c, key) == val for key, val in filters.items())]
        if not selected:
            return []
        index = self.index
        if filters:
            index = faiss.IndexFlatIP(self.encoder.dimensions)
            vectors = np.ascontiguousarray(self.vectors[selected], dtype=np.float32)
            faiss.normalize_L2(vectors)
            index.add(vectors)
        query_vector = np.ascontiguousarray(self.encoder.encode([query]), dtype=np.float32)
        faiss.normalize_L2(query_vector)
        scores, positions = index.search(query_vector, min(k, len(selected)))
        hits = [Hit(self.chunks[selected[int(pos)]], float(score))
                for score, pos in zip(scores[0], positions[0]) if pos >= 0 and score > 1e-8]
        return sorted(hits, key=lambda h: (-h.score, h.chunk.chunk_id))
