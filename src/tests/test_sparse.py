"""Deterministic fake sparse vectors; these tests are not actual-model evidence."""
import json
from pathlib import Path
import numpy as np
import pytest
from scipy import sparse

from blended_rag.retrieval import LocalRetriever, RetrievalConfig, PROFILES


class FakeEncoder:
    dimensions = 30522
    identity = {"model": "test-only", "revision": "fake", "dimensions": 30522}
    def encode_document(self, texts):
        return sparse.csr_matrix(([2.] * len(texts), (range(len(texts)), [7] * len(texts))), shape=(len(texts), self.dimensions))
    def encode_query(self, text):
        return sparse.csr_matrix(([3.], ([0], [7])), shape=(1, self.dimensions))


def test_postings_reference_and_reload(store):
    r = LocalRetriever(store)
    for profile in PROFILES:
        r.config = RetrievalConfig(profile)
        first = r.search("accounts ac-2", "lexical", filters={"release": "rev5"})
        assert first
        assert r.postings.path.exists()
        terms = ["accounts", "ac-2"]
        stats = {"combined": (r.counts, r.lengths, max(r.average_length, 1e-12), r.idf), **r.field_stats}
        for field, values in stats.items():
            assert np.allclose(r.postings.score(field, terms), r._bm25(terms, list(range(len(r.chunks))), values))
        selected = [i for i, c in enumerate(r.chunks) if c.control_id == "ac-2"]
        fields = r.config.metadata()["fields"]
        parts = [weight * r._bm25(["ac-2"] if field == "control_id" else terms, selected,
                    stats["combined" if field == "search_text" else field]) for field, weight in fields.items()]
        reference = np.maximum.reduce(parts) if profile == "best-fields" else np.sum(parts, axis=0)
        expected = sorted((i for i in selected if reference[i] > 1e-8), key=lambda i: (-reference[i], r.chunks[i].chunk_id))
        actual = r.search("accounts ac-2", "lexical", filters={"control_id": "ac-2"})
        assert [h.chunk.chunk_id for h in actual] == [r.chunks[i].chunk_id for i in expected]
        assert np.allclose([h.score for h in actual], [reference[i] for i in expected])
        again = LocalRetriever(store, r.config)
        assert [(h.chunk.chunk_id, h.score) for h in first] == [(h.chunk.chunk_id, h.score) for h in again.search("accounts ac-2", "lexical", filters={"release": "rev5"})]
    r.postings.path.write_text("corrupt")
    with pytest.raises(ValueError, match="BM25"):
        LocalRetriever(store)


def test_sparse_dot_product_and_missing(store):
    from blended_rag.sparse import SparseStore
    r = LocalRetriever(store)
    with pytest.raises(ValueError, match="sparse-index"):
        r.search("accounts", "sparse")
    s = SparseStore(store.directory / "sparse")
    report = s.build(r.chunks, FakeEncoder(), batch_size=1)
    assert report["nnz"] == len(r.chunks)
    index, meta = s.load(r.chunks, FakeEncoder().identity)
    assert np.allclose((index @ FakeEncoder().encode_query("x").T).toarray(), 6)
    r.sparse_encoder = FakeEncoder()
    assert r.search("accounts", "sparse")[0].score == 6
    r.config = RetrievalConfig(lexical_weight=0, dense_weight=0, sparse_weight=1)
    r.encoder.encode = lambda *_: pytest.fail("disabled dense branch called")
    assert r.search("accounts", "blended")
    assert s.build(r.chunks, FakeEncoder(), batch_size=1)["changed"] is False


def test_sparse_failure_resume_atomicity(store):
    from blended_rag.sparse import SparseStore
    chunks = LocalRetriever(store).chunks
    s = SparseStore(store.directory / "sparse")
    class Failing(FakeEncoder):
        calls = 0
        def encode_document(self, texts):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("interrupted")
            return super().encode_document(texts)
    with pytest.raises(RuntimeError):
        s.build(chunks, Failing(), batch_size=1)
    assert not s.path.exists()
    result = s.build(chunks, FakeEncoder(), batch_size=1)
    assert result["reused_batches"] == 1
    manifest = json.loads(s.path.read_text())
    (s.directory / manifest["file"]).write_bytes(b"bad")
    with pytest.raises(ValueError):
        s.load(chunks, FakeEncoder().identity)


@pytest.mark.parametrize("corruption", ["swapped", "extra"])
def test_sparse_resume_rejects_wrong_batch_positions(store, corruption):
    from blended_rag.sparse import SparseStore
    chunks = LocalRetriever(store).chunks
    s = SparseStore(store.directory / "sparse")
    class Interrupted(FakeEncoder):
        calls = 0
        def encode_document(self, texts):
            self.calls += 1
            if self.calls == 3:
                raise RuntimeError("interrupted")
            return super().encode_document(texts)
    with pytest.raises(RuntimeError):
        s.build(chunks, Interrupted(), batch_size=1)
    checkpoint = next(s.directory.glob("checkpoint-*.json"))
    state = json.loads(checkpoint.read_text())
    if corruption == "swapped":
        state["batches"].reverse()
    else:
        state["batches"].extend([state["batches"][-1]] * len(chunks))
    checkpoint.write_text(json.dumps(state))
    encoder = FakeEncoder()
    encoder.encode_document = lambda *_: pytest.fail("invalid checkpoint reached inference")
    with pytest.raises(ValueError, match="checkpoint.*batch"):
        s.build(chunks, encoder, batch_size=1)
    assert not s.path.exists()


def test_sparse_evaluation_five_reports(store, tmp_path):
    from blended_rag.evaluation import sparse_evaluate
    from blended_rag.sparse import SparseStore
    r = LocalRetriever(store)
    r.sparse_encoder = FakeEncoder()
    SparseStore(store.directory / "sparse").build(r.chunks, r.sparse_encoder)
    queries = tmp_path / "queries.json"
    queries.write_text(json.dumps({"label_provenance": "test-only", "queries": [
        {"id": "q1", "query": "accounts", "filters": {}, "relevant": {"rev5:ac-2": 1}}]}))
    report = sparse_evaluate(r, store, tmp_path / "report.json", queries)
    assert report["status"] == "complete"
    assert len(report["configurations"]) == 5
    assert all(Path(row["output"]).exists() for row in report["configurations"])
    metadata = json.loads((tmp_path / "report.json").read_text())["input_representations"]["dense"]
    assert metadata["model"] == r.model
    assert metadata["encoder_family"] == "TF-IDF + SVD (LSA)"
    assert metadata["vector_engine"] == "numpy-exact-cosine"
    assert metadata["dimensions"] == r.encoder.dimensions


def test_sparse_evaluation_rejects_gemini_before_provider_call(store, tmp_path):
    from blended_rag.evaluation import sparse_evaluate
    from blended_rag.providers import GeminiEncoder
    r = LocalRetriever(store)
    r.encoder = GeminiEncoder()
    r.encoder.encode = lambda *_: pytest.fail("cloud provider called by local comparison")
    r.search = lambda *_: pytest.fail("retrieval started before cloud rejection")
    target = tmp_path / "cloud.json"
    with pytest.raises(ValueError, match="local.*encoder"):
        sparse_evaluate(r, store, target)
    assert not target.exists()


def test_sparse_cli_api_and_atlas_rejection():
    from main import parser
    from blended_rag.api import Query
    assert parser().parse_args(["--sparse-weight", "1", "query", "x", "--method", "blended"]).method == "blended"
    assert Query(question="x", method="sparse").method == "sparse"
    from blended_rag.atlas import AtlasRetriever
    atlas = object.__new__(AtlasRetriever)
    for method in ("sparse", "blended"):
        with pytest.raises(ValueError, match="Atlas.*sparse"):
            atlas.search("x", method)


def test_sparse_model_source_and_values(store):
    from dataclasses import replace
    from blended_rag.models import digest
    from blended_rag.sparse import SparseStore
    chunks = LocalRetriever(store).chunks
    s = SparseStore(store.directory / "sparse")
    s.build(chunks, FakeEncoder())
    ready = s.path.read_bytes()
    with pytest.raises(ValueError, match="model"):
        s.load(chunks, {**FakeEncoder.identity, "revision": "other"})
    changed = [replace(chunks[0], text="changed", content_hash=digest("changed")), *chunks[1:]]
    with pytest.raises(ValueError, match="source"):
        s.load(changed, FakeEncoder.identity)
    class Negative(FakeEncoder):
        def encode_document(self, texts):
            return -super().encode_document(texts)
    with pytest.raises(ValueError, match="nonnegative"):
        s.build(changed, Negative(), batch_size=1)
    assert s.path.read_bytes() == ready
    assert s.load(chunks, FakeEncoder.identity)[0].nnz == len(chunks)


def test_postings_vocabulary_tamper_rejected(store):
    r = LocalRetriever(store)
    manifest = json.loads(r.postings.path.read_text())
    terms = manifest["artifacts"]["combined"]["terms"]
    terms[0], terms[-1] = terms[-1], terms[0]
    r.postings.path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="BM25"):
        LocalRetriever(store)


def test_blended_zero_and_unready_are_errors(store):
    r = LocalRetriever(store)
    with pytest.raises(ValueError, match="positive sparse_weight"):
        r.search("x", "blended")
    r.config = RetrievalConfig(lexical_weight=0, dense_weight=0, sparse_weight=1)
    with pytest.raises(ValueError, match="hybrid requires"):
        r.search("x", "hybrid")


def test_sparse_evaluation_failure_checkpoints(store, tmp_path):
    from blended_rag.evaluation import sparse_evaluate
    r = LocalRetriever(store)
    queries = tmp_path / "queries.json"
    queries.write_text(json.dumps({"label_provenance": "test-only", "queries": [
        {"id": "q1", "query": "accounts", "filters": {}, "relevant": {"rev5:ac-2": 1}}]}))
    result = sparse_evaluate(r, store, tmp_path / "failure.json", queries)
    assert result["status"] == "failed"
    assert [row["method"] for row in result["configurations"] if row["status"] == "failed"] == ["sparse", "blended"]


def test_cached_real_sparse_smoke():
    import os
    if os.getenv("RUN_CACHED_SPLADE_SMOKE") != "1":
        pytest.skip("Explicit cached actual-model evidence only")
    from blended_rag.sparse import SparseEncoder
    encoder = SparseEncoder()
    documents = encoder.encode_document(["Account Management ac-2 Remove accounts when employees leave."])
    query = encoder.encode_query("remove user accounts")
    assert documents.shape == (1, 30522) and query.nnz > 0
    assert (documents @ query.T)[0, 0] > 0
