from __future__ import annotations

import math
from collections import Counter
from typing import Protocol

import numpy as np

from .ingestion import LocalStore
from .models import Chunk, Hit, tokens, validate_filters, validate_query


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
