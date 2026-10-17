"""Atlas Search and Vector Search with client RRF compatibility fusion."""
from __future__ import annotations

import os
import json
import time
from dataclasses import asdict

from pymongo import MongoClient, ReplaceOne
from pymongo.operations import SearchIndexModel

from .ingestion import LocalStore
from .models import Chunk, Hit, digest, validate_filters, validate_query
from .retrieval import rrf


class AtlasRetriever:
    mode = "legacy"
    vector_engine = "mongodb-atlas-vector-search"
    answer_engine = "extractive-nlp"
    def __init__(self, store: LocalStore):
        uri = os.getenv("MONGODB_URI")
        if not uri:
            raise ValueError("Atlas requires MONGODB_URI; configure .env.example variables in your environment")
        self.client = MongoClient(uri, serverSelectionTimeoutMS=10000, connectTimeoutMS=10000,
                                  socketTimeoutMS=60000, retryWrites=True, retryReads=True)
        self.client.admin.command("ping")
        db = self.client[os.getenv("MONGODB_DATABASE", "compliance_rag")]
        name = os.getenv("MONGODB_COLLECTION", "chunks")
        self.collection = db[name]
        self.sources = db[name + "_sources"]
        self.text_index = os.getenv("ATLAS_TEXT_INDEX", "compliance_text")
        self.vector_index = os.getenv("ATLAS_VECTOR_INDEX", "compliance_vector")
        self.encoder = store.encoder()
        self.model = {"name": self.encoder.name, "fingerprint": self.encoder.fingerprint}

    def index_definitions(self):
        fields = {key: {"type": "token"} for key in
                  ["release", "source_id", "control_id", "generation", "model_fingerprint"]}
        fields.update({key: {"type": "string", "analyzer": "lucene.standard"} for key in ["title", "text", "search_text"]})
        return [SearchIndexModel(name=self.text_index, definition={"mappings": {"dynamic": False, "fields": fields}}),
                SearchIndexModel(name=self.vector_index, type="vectorSearch", definition={"fields": [
                    {"type": "vector", "path": "embedding", "numDimensions": self.encoder.dimensions, "similarity": "cosine"},
                    *[{"type": "filter", "path": key} for key in
                      ["release", "source_id", "control_id", "generation", "model_fingerprint"]]]})]

    def setup_indexes(self, timeout=300):
        self.collection.create_index([("source_id", 1), ("release", 1), ("generation", 1)],
                                     name="source_release_generation")
        existing = {i["name"]: i for i in self.collection.list_search_indexes()}
        for index in self.index_definitions():
            spec = index.document
            if spec["name"] not in existing:
                self.collection.create_search_index(index)
            elif existing[spec["name"]].get("latestDefinition") != spec["definition"]:
                raise ValueError("Existing Atlas index definition differs; use a new index name or update explicitly")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            indexes = {i["name"]: i for i in self.collection.list_search_indexes()}
            if all(indexes.get(name, {}).get("queryable") for name in [self.text_index, self.vector_index]):
                return {"indexes": [self.text_index, self.vector_index], "queryable": True}
            time.sleep(2)
        raise TimeoutError("Atlas indexes did not become queryable before timeout; inspect Atlas index status")

    def ingest(self, store: LocalStore):
        state = store.read()
        if state["model_fingerprint"] != self.encoder.fingerprint:
            raise ValueError("Embedding model mismatch")
        current = list(self.sources.find({}, {"_id": 0}))
        if any(s["model_fingerprint"] != self.encoder.fingerprint for s in current):
            raise ValueError("Atlas embedding model mismatch; use a new collection and index for a changed LSA corpus")
        grouped = {}
        for chunk, vector in zip(state["chunks"], state["vectors"], strict=True):
            grouped.setdefault((chunk["source_id"], chunk["release"]), []).append((chunk, vector))
        count = 0
        for (source_id, release), items in grouped.items():
            key = f"{source_id}:{release}"
            prepared = [{**c, "embedding": v} for c, v in sorted(items, key=lambda item: item[0]["chunk_id"])]
            generation = digest(json.dumps({"source_id": source_id, "release": release,
                                           "model_fingerprint": self.encoder.fingerprint, "chunks": prepared},
                                          sort_keys=True, separators=(",", ":"), allow_nan=False))
            previous = self.sources.find_one({"_id": key})
            if previous and previous["generation"] == generation:
                continue
            writes = [ReplaceOne({"_id": c["chunk_id"] + ":" + generation},
                                 {**c, "search_text": c["control_id"] + " " + c["title"] + " " + c["text"], "embedding": v, "generation": generation,
                                  "model_fingerprint": self.encoder.fingerprint}, upsert=True) for c, v in items]
            for offset in range(0, len(writes), 200):
                self.collection.bulk_write(writes[offset:offset + 200], ordered=False)
            # Source pointer switches only after all prepared chunks have persisted.
            self.sources.replace_one({"_id": key}, {"_id": key, "source_id": source_id, "release": release,
                                     "generation": generation, "model_fingerprint": self.encoder.fingerprint}, upsert=True)
            count += len(items)
        return {"uploaded_chunks": count, "sources": len(grouped),
                "note": "Atlas Search indexing is eventually consistent; wait for evidence before benchmarking"}

    def _filters(self, filters):
        pointers = list(self.sources.find(filters, {"generation": 1, "model_fingerprint": 1})) if "control_id" not in filters else list(
            self.sources.find({k: v for k, v in filters.items() if k != "control_id"}, {"generation": 1, "model_fingerprint": 1}))
        if any(p["model_fingerprint"] != self.encoder.fingerprint for p in pointers):
            raise ValueError("Atlas embedding model mismatch")
        return {**filters, "model_fingerprint": self.encoder.fingerprint,
                "generation": {"$in": [p["generation"] for p in pointers]}}

    def search(self, query, method="hybrid", k=10, filters=None):
        validate_query(query, k)
        filters = validate_filters(filters)
        if method not in {"lexical", "dense", "hybrid"}:
            raise ValueError("method must be lexical, dense or hybrid")
        if method == "hybrid":
            pool = min(100, max(50, k * 5))
            return rrf([self.search(query, "lexical", pool, filters), self.search(query, "dense", pool, filters)], k)
        active = self._filters(filters)
        if not active["generation"]["$in"]:
            return []
        if method == "lexical":
            constraints = [{"in": {"path": key, "value": value["$in"]}} if isinstance(value, dict)
                           else {"equals": {"path": key, "value": value}} for key, value in active.items()]
            pipeline = [{"$search": {"index": self.text_index, "compound": {
                "must": [{"text": {"query": query, "path": "search_text"}}], "filter": constraints}}},
                {"$limit": k}, {"$set": {"score": {"$meta": "searchScore"}}}]
        else:
            vector = (self.encoder.encode_query(query) if hasattr(self.encoder, "encode_query") else self.encoder.encode([query])[0]).tolist()
            pipeline = [{"$vectorSearch": {"index": self.vector_index, "path": "embedding", "queryVector": vector,
                         "numCandidates": max(100, k * 20), "limit": k, "filter": active}},
                        {"$set": {"score": {"$meta": "vectorSearchScore"}}}]
        pipeline.append({"$project": {"embedding": 0}})
        fields = Chunk.__dataclass_fields__
        return [Hit(Chunk(**{key: row[key] for key in fields}), float(row["score"]))
                for row in self.collection.aggregate(pipeline, maxTimeMS=60000)]

    def close(self):
        self.client.close()
        provider_client = getattr(self.encoder, "_client", None)
        if provider_client is not None:
            provider_client.close()
