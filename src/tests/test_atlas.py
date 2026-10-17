import os
import time
import copy

import pytest

from blended_rag.atlas import AtlasRetriever


def test_atlas_index_contract(store, monkeypatch):
    # Inspect definitions without constructing a network client.
    adapter = AtlasRetriever.__new__(AtlasRetriever)
    adapter.encoder = store.encoder()
    adapter.text_index = "text"
    adapter.vector_index = "vector"
    definitions = [d.document for d in adapter.index_definitions()]
    fields = definitions[1]["definition"]["fields"]
    assert fields[0]["numDimensions"] == store.read()["dimensions"]
    assert {f["path"] for f in fields[1:]} >= {"release", "control_id", "source_id", "generation"}


def test_atlas_pipeline_filters_match(store):
    adapter = AtlasRetriever.__new__(AtlasRetriever)
    adapter.encoder = store.encoder()
    adapter.text_index = "text"
    adapter.vector_index = "vector"
    active = {"release": "rev5", "control_id": "ac-2", "generation": {"$in": ["active-generation"]},
              "model_fingerprint": adapter.encoder.fingerprint}
    adapter._filters = lambda filters: active
    captured = []
    class Collection:
        def aggregate(self, pipeline, **kwargs):
            captured.append(pipeline)
            return []
    adapter.collection = Collection()
    for method in ["lexical", "dense"]:
        adapter.search("account", method, 5, {"release": "rev5", "control_id": "ac-2"})
    lexical = captured[0][0]["$search"]["compound"]["filter"]
    lexical_values = {next(iter(item.values()))["path"]: next(iter(item.values()))["value"] for item in lexical}
    assert lexical_values["release"] == "rev5" and lexical_values["control_id"] == "ac-2"
    assert lexical_values["generation"] == ["active-generation"]
    assert captured[1][0]["$vectorSearch"]["filter"] == active


def test_atlas_creates_collection_before_search_indexes(store):
    adapter = AtlasRetriever.__new__(AtlasRetriever)
    adapter.encoder = store.encoder()
    adapter.text_index, adapter.vector_index = "text", "vector"
    events, indexes = [], []
    class Collection:
        def create_index(self, keys, name):
            events.append("ordinary-index")
            assert keys == [("source_id", 1), ("release", 1), ("generation", 1)]
        def list_search_indexes(self):
            assert events[0] == "ordinary-index"
            events.append("list-search-indexes")
            return indexes
        def create_search_index(self, model):
            events.append("create-search-index")
            indexes.append({"name": model.document["name"], "queryable": True})
    adapter.collection = Collection()
    assert adapter.setup_indexes(timeout=1)["queryable"]
    assert events[0] == "ordinary-index" and events.count("create-search-index") == 2


@pytest.mark.parametrize("field,value", [("title", "Revised control title"), ("chunk_id", "new-parser-identity")])
def test_atlas_generation_tracks_complete_chunk_records(store, field, value):
    adapter = AtlasRetriever.__new__(AtlasRetriever)
    adapter.encoder = store.encoder()
    pointers, writes = {}, []
    class Sources:
        def find(self, *args):
            return list(pointers.values())
        def find_one(self, query):
            return pointers.get(query["_id"])
        def replace_one(self, query, document, **kwargs):
            pointers[query["_id"]] = document
    class Collection:
        def bulk_write(self, batch, **kwargs):
            writes.extend(batch)
    class PreparedStore:
        state = copy.deepcopy(store.read())
        def read(self):
            return self.state
    prepared = PreparedStore()
    adapter.sources, adapter.collection = Sources(), Collection()
    count = len(prepared.state["chunks"])
    assert adapter.ingest(prepared)["uploaded_chunks"] == count
    original = next(iter(pointers.values()))["generation"]
    assert adapter.ingest(prepared)["uploaded_chunks"] == 0
    prepared.state["chunks"][0][field] = value
    assert adapter.ingest(prepared)["uploaded_chunks"] == count
    assert next(iter(pointers.values()))["generation"] != original
    assert len(writes) == count * 2


@pytest.mark.skipif(os.getenv("RUN_ATLAS_TESTS") != "1" or not os.getenv("MONGODB_URI"),
                    reason="Requires explicit RUN_ATLAS_TESTS=1 and Atlas credentials; live integration unvalidated")
def test_live_atlas(store):
    assert "test" in os.getenv("MONGODB_COLLECTION", ""), "Use a dedicated collection with 'test' in its name"
    atlas = AtlasRetriever(store)
    try:
        atlas.setup_indexes()
        atlas.ingest(store)
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if all(atlas.search("account management", method, 5, {"release": "rev5", "control_id": "ac-2"})
                   for method in ["lexical", "dense", "hybrid"]):
                break
            time.sleep(2)
        else:
            pytest.fail("Atlas results not visible after eventual-consistency wait")
        assert atlas.search("account", "hybrid", 5, {"release": "rev4"}) == []
        assert atlas.ingest(store)["uploaded_chunks"] == 0
    finally:
        atlas.close()
