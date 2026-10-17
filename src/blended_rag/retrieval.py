from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from collections import Counter
from typing import Protocol

import numpy as np

from .ingestion import LocalStore
from .models import Chunk, Hit, digest, tokens, validate_filters, validate_query


PROFILES = ("combined", "content-only", "multi-field", "boosted-sum", "best-fields")
FIELD_BOOSTS = {"control_id": 8.0, "title": 3.0, "section_id": 1.5, "text": 1.0}


def identifiers(query):
    """Control IDs, including dotted enhancements, independent of benchmark labels."""
    return list(dict.fromkeys(re.findall(r"(?<![a-z0-9-])[a-z]{2,3}-[0-9]+(?:\.[0-9]+)*(?![a-z0-9-]|\.[a-z0-9])", query.lower())))


@dataclass(frozen=True)
class RetrievalConfig:
    lexical_profile: str = "combined"
    query_formulation: str = "raw"
    lexical_weight: float = 1.0
    dense_weight: float = 1.0
    sparse_weight: float = 0.0

    def __post_init__(self):
        if self.lexical_profile not in PROFILES:
            raise ValueError("Unknown lexical profile")
        if self.query_formulation not in {"raw", "identifier"}:
            raise ValueError("query formulation must be raw or identifier")
        for value in (self.lexical_weight, self.dense_weight, self.sparse_weight):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("Fusion weights must be finite nonnegative numbers")
        if self.lexical_weight + self.dense_weight + self.sparse_weight <= 0:
            raise ValueError("At least one fusion weight must be positive")

    def metadata(self):
        profile = self.lexical_profile
        fields = {"search_text": 1.0} if profile == "combined" else {"text": 1.0} if profile == "content-only" else (
            dict.fromkeys(FIELD_BOOSTS, 1.0) if profile == "multi-field" else dict(FIELD_BOOSTS))
        return {**asdict(self), "fields": fields, "field_combination": "max" if profile == "best-fields" else "sum",
                "rrf_constant": 60, "hybrid_branch_candidate_limit": 100,
                "atlas_best_fields_candidate_limit_per_field": 100,
                "identifier_semantics": "identifier formulation restricts detected IDs to exact lowercase control_id tokens"}


class Retriever(Protocol):
    def search(self, query: str, method: str = "hybrid", k: int = 10, filters: dict | None = None) -> list[Hit]: ...


def rrf(rankings: list[list[Hit]], k: int, constant: int = 60, weights=None) -> list[Hit]:
    weights = [1.0] * len(rankings) if weights is None else list(weights)
    if len(weights) != len(rankings) or not weights or any(
            isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w < 0 for w in weights) or not any(weights):
        raise ValueError("Invalid RRF weights")
    if not isinstance(constant, int) or constant < 0:
        raise ValueError("RRF constant must be a nonnegative integer")
    scores, chunks = {}, {}
    for ranking, weight in zip(rankings, weights):
        if weight == 0:
            continue
        seen = set()
        for rank, hit in enumerate(ranking, 1):
            key = hit.chunk.chunk_id
            if key in seen:
                continue
            seen.add(key)
            scores[key] = scores.get(key, 0.0) + weight / (constant + rank)
            chunks[key] = hit.chunk
    return [Hit(chunks[key], score) for key, score in
            sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]]


class LocalRetriever:
    mode = "legacy"
    vector_engine = "numpy-exact-cosine"
    answer_engine = "extractive-nlp"
    def __init__(self, store: LocalStore, config=None):
        self.directory = store.directory
        self.config = config or RetrievalConfig()
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
        self.field_stats = {}
        for field in FIELD_BOOSTS:
            counts = [Counter([c.control_id.lower()] if field == "control_id" else tokens(getattr(c, field))) for c in self.chunks]
            lengths = np.array([sum(c.values()) for c in counts])
            df = Counter(term for c in counts for term in c)
            idf = {term: math.log(1 + (len(counts) - n + .5) / (n + .5)) for term, n in df.items()}
            self.field_stats[field] = (counts, lengths, max(float(lengths.mean()), 1e-12), idf)
        from .postings import BM25Postings
        self.postings = BM25Postings(store.directory, self.chunks, {
            "combined": (self.counts, self.lengths, max(self.average_length, 1e-12), self.idf), **self.field_stats})

    def _bm25(self, terms, selected, stats):
        counts, lengths, average, idf = stats
        scores = np.zeros(len(self.chunks))
        for term in set(terms):
            for i in selected:
                freq = counts[i].get(term, 0)
                if freq:
                    scores[i] += idf.get(term, 0) * (freq * 2.5) / (freq + 1.5 * (.25 + .75 * lengths[i] / average))
        return scores

    def search(self, query, method="hybrid", k=10, filters=None):
        validate_query(query, k)
        filters = validate_filters(filters)
        if method not in {"lexical", "dense", "hybrid", "sparse", "blended"}:
            raise ValueError("method must be lexical, dense, hybrid, sparse or blended")
        if method in {"hybrid", "blended"}:
            if method == "blended" and self.config.sparse_weight <= 0:
                raise ValueError("blended requires positive sparse_weight and a ready sparse-index")
            if method == "hybrid" and self.config.lexical_weight + self.config.dense_weight <= 0:
                raise ValueError("hybrid requires positive lexical or dense weight")
            pool = max(50, k * 5)
            branches = [("lexical", self.config.lexical_weight), ("dense", self.config.dense_weight)]
            # Hybrid retains its original two branches; sparse participates only by explicit selection.
            if method == "blended":
                branches.append(("sparse", self.config.sparse_weight))
            return rrf([self.search(query, branch, min(pool, 100), filters) if weight else []
                        for branch, weight in branches], k, weights=[weight for _, weight in branches])
        selected = [i for i, c in enumerate(self.chunks) if all(getattr(c, key) == val for key, val in filters.items())]
        if method == "sparse":
            from .sparse import SparseEncoder, SparseStore
            if not hasattr(self, "sparse_encoder"):
                self.sparse_encoder = SparseEncoder()
            if not hasattr(self, "sparse_index"):
                self.sparse_index, self.sparse_metadata = SparseStore(self.directory / "sparse").load(self.chunks, self.sparse_encoder.identity)
            vector = self.sparse_encoder.encode_query(query)
            scores = (self.sparse_index @ vector.T).toarray().ravel()
        elif method == "dense":
            vector = self.encoder.encode([query])[0]
            scores = self.vectors @ vector
        else:
            ids = identifiers(query)
            if self.config.query_formulation == "identifier" and ids:
                selected = [i for i in selected if self.chunks[i].control_id.lower() in ids]
            profile = self.config.lexical_profile
            if profile == "combined":
                scores = self.postings.score("combined", tokens(query))
            else:
                fields = self.config.metadata()["fields"]
                parts = [weight * self.postings.score(field, ids if field == "control_id" else tokens(query))
                         for field, weight in fields.items()]
                scores = np.maximum.reduce(parts) if profile == "best-fields" else np.sum(parts, axis=0)
        indices = sorted((i for i in selected if scores[i] > 1e-8), key=lambda i: (-scores[i], self.chunks[i].chunk_id))[:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in indices]


class FaissRetriever(LocalRetriever):
    """Modern local mode: genuine FAISS; metadata filtering precedes top-k."""
    mode = "local"
    vector_engine = "faiss-IndexFlatIP"

    def __init__(self, store, config=None):
        super().__init__(store, config)
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
