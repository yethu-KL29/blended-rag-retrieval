"""Optional SPLADE provider and independent resumable sparse index.

Importing this module never imports Torch or downloads models. Only explicit setup
permits network access; document/query encoding always uses cached pinned files.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy import sparse

from .models import digest
from .postings import atomic_json, source_identity

MODEL = "naver/splade-cocondenser-ensembledistil"
REVISION = "49cf4c7b0db5b870a401ddf5e2669993ef3699c7"
DIMENSIONS = 30522
IDENTITY = {"model": MODEL, "revision": REVISION, "dimensions": DIMENSIONS,
            "max_length": 256, "representation": "title + control_id + text",
            "pooling": "max(log1p(relu(MLM logits)))", "normalization": "none"}


class SparseEncoder:
    dimensions = DIMENSIONS
    identity = IDENTITY

    def __init__(self, threads=4, batch_size=8):
        if not 1 <= threads <= 32 or not 1 <= batch_size <= 128:
            raise ValueError("Sparse threads must be 1..32; batch size 1..128")
        self.threads, self.batch_size = threads, batch_size

    @staticmethod
    def setup():
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ValueError("Install requirements-sparse.txt before sparse-setup") from None
        path = snapshot_download(MODEL, revision=REVISION)
        return {"model": MODEL, "revision": REVISION, "license": "CC-BY-NC-SA-4.0",
                "model_bytes": sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file()),
                "offline_queries": True}

    def _load(self):
        if hasattr(self, "model"):
            return
        try:
            import torch
            from sentence_transformers import SparseEncoder as STSparseEncoder
            from huggingface_hub import snapshot_download
            torch.set_num_threads(self.threads)
            # Resolve only the pinned local snapshot; never use a moving main revision.
            path = snapshot_download(MODEL, revision=REVISION, local_files_only=True)
            self.model = STSparseEncoder(path, device="cpu", local_files_only=True)
            self.model.max_seq_length = 256
            self.model_bytes = sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
        except (ImportError, OSError) as exc:
            raise ValueError("SPLADE unavailable: install requirements-sparse.txt and run sparse-setup explicitly; then sparse-index") from exc

    def _encode(self, texts, kind):
        self._load()
        method = self.model.encode_document if kind == "document" else self.model.encode_query
        result = method(texts, batch_size=self.batch_size, convert_to_sparse_tensor=True, show_progress_bar=False)
        result = result.detach().cpu().to_sparse_coo().coalesce()
        indices = result.indices().numpy()
        matrix = sparse.csr_matrix((result.values().numpy(), (indices[0], indices[1])), shape=tuple(result.shape))
        return validate_matrix(matrix, len(texts))

    def encode_document(self, texts):
        return self._encode(texts, "document")

    def encode_query(self, text):
        return self._encode([text], "query")

    def expansions(self, text, limit=30):
        vector = self.encode_query(text).tocoo()
        tokenizer = self.model.tokenizer
        return [{"token": tokenizer.convert_ids_to_tokens(int(i)), "dimension": int(i), "weight": float(w)}
                for i, w in sorted(zip(vector.col, vector.data), key=lambda x: (-x[1], x[0]))[:limit]]


def validate_matrix(matrix, rows):
    if not sparse.issparse(matrix):
        raise ValueError("SPLADE must produce sparse vectors")
    matrix = matrix.tocsr()
    matrix.check_format(full_check=True)
    if matrix.shape != (rows, DIMENSIONS) or not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
        raise ValueError("Sparse index count/vocabulary/nonnegative finite weights mismatch")
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return matrix


def save_matrix(path, matrix):
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    os.close(fd)
    try:
        sparse.save_npz(temporary, matrix)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {"file": path.name, "sha256": digest(path.read_bytes()), "bytes": path.stat().st_size}


class SparseStore:
    """CSR rows are document vectors; query dot products allocate only N scores."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.path = self.directory / "ready.json"

    @staticmethod
    def identity(chunks, model):
        # Equal chunk IDs alone cannot detect changed text or mixed model representations.
        return {"format": 1, "source": source_identity(chunks), "model": model, "count": len(chunks)}

    def _read_matrix(self, item, rows):
        path = self.directory / item["file"]
        if path.name != item["file"] or digest(path.read_bytes()) != item["sha256"]:
            raise ValueError("Sparse artifact hash mismatch")
        return validate_matrix(sparse.load_npz(path), rows)

    def load(self, chunks, model=IDENTITY):
        if not self.path.exists():
            raise ValueError("Ready SPLADE index missing; run sparse-setup then sparse-index explicitly")
        try:
            meta = json.loads(self.path.read_text())
            if meta["identity"] != self.identity(chunks, model):
                raise ValueError("Sparse source/model/format mismatch; run sparse-index")
            return self._read_matrix(meta, len(chunks)), meta
        except (OSError, KeyError, TypeError) as exc:
            raise ValueError("Sparse index invalid; run sparse-index") from exc

    def build(self, chunks, encoder=None, batch_size=8, progress=None):
        if not chunks or not 1 <= batch_size <= 128:
            raise ValueError("Sparse indexing requires chunks and batch size 1..128")
        encoder = encoder or SparseEncoder(batch_size=batch_size)
        identity = self.identity(chunks, encoder.identity)
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            existing = json.loads(self.path.read_text())
            if existing.get("identity") == identity:
                _, meta = self.load(chunks, encoder.identity)
                return {**meta, "changed": False, "reused_batches": 0}
        generation = digest(json.dumps(identity, sort_keys=True))
        checkpoint = self.directory / f"checkpoint-{generation}-{batch_size}.json"
        state = {"identity": identity, "batch_size": batch_size, "batches": [], "elapsed_seconds": 0.0}
        if checkpoint.exists():
            state = json.loads(checkpoint.read_text())
            if state["identity"] != identity or state["batch_size"] != batch_size:
                raise ValueError("Sparse checkpoint identity mismatch")
        # A batch is reusable only after its payload and checkpoint were atomically saved.
        batches = state.get("batches")
        expected_batches = (len(chunks) + batch_size - 1) // batch_size
        if (not isinstance(batches, list) or len(batches) > expected_batches or any(
                not isinstance(item, dict) or item.get("file") != f"batch-{generation}-{batch_size}-{batch}.npz"
                for batch, item in enumerate(batches))):
            raise ValueError("Sparse checkpoint batch count/order mismatch")
        reused = len(state["batches"])
        start = time.perf_counter()
        elapsed_prior = state["elapsed_seconds"]
        matrices = []
        for batch, offset in enumerate(range(0, len(chunks), batch_size)):
            selected = chunks[offset:offset + batch_size]
            if batch < reused:
                matrix = self._read_matrix(state["batches"][batch], len(selected))
            else:
                matrix = validate_matrix(encoder.encode_document([f"{c.title} {c.control_id} {c.text}" for c in selected]), len(selected))
                item = save_matrix(self.directory / f"batch-{generation}-{batch_size}-{batch}.npz", matrix)
                state["batches"].append(item)
                state["elapsed_seconds"] = elapsed_prior + time.perf_counter() - start
                atomic_json(checkpoint, state)
            matrices.append(matrix)
            if progress:
                progress({"completed_chunks": min(offset + batch_size, len(chunks)), "total_chunks": len(chunks),
                          "reused_batches": reused, "elapsed_seconds": elapsed_prior + time.perf_counter() - start,
                          "nnz": sum(m.nnz for m in matrices)})
        matrix = validate_matrix(sparse.vstack(matrices, format="csr"), len(chunks))
        meta = {**save_matrix(self.directory / f"index-{generation}.npz", matrix), "identity": identity,
                "nnz": matrix.nnz, "elapsed_seconds": elapsed_prior + time.perf_counter() - start,
                "model_bytes": getattr(encoder, "model_bytes", None), "license": "CC-BY-NC-SA-4.0",
                "external_api_cost_usd": 0, "batch_size": batch_size, "checkpoint_size": batch_size,
                "encoder_batch_size": getattr(encoder, "batch_size", None), "threads": getattr(encoder, "threads", None)}
        # The previous ready generation remains authoritative until the entire build passes.
        atomic_json(self.path, meta)
        return {**meta, "changed": True, "reused_batches": reused}
