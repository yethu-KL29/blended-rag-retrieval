"""Versioned weighted BM25 postings; query cost follows matching postings."""
import json
import os
import tempfile
from dataclasses import asdict

import numpy as np
from scipy.sparse import csc_matrix, save_npz, load_npz

from .models import digest


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            json.dump(value, out, sort_keys=True, allow_nan=False)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def source_identity(chunks):
    """Bind order and all source metadata, including text and citation hashes."""
    return digest(json.dumps([asdict(c) for c in chunks], sort_keys=True))


class BM25Postings:
    """CSC columns are terms; each column visits only documents containing it."""
    def __init__(self, directory, chunks, statistics):
        self.path = directory / "bm25-ready.json"
        identity = {"format": 1, "tokenizer": "ascii-hyphen-v1", "k1": 1.5, "b": .75,
                    "source": source_identity(chunks), "fields": list(statistics), "count": len(chunks)}
        self.matrices, self.vocabulary = {}, {}
        if self.path.exists():
            try:
                manifest = json.loads(self.path.read_text())
                if manifest["identity"] != identity:
                    raise ValueError("source/format mismatch")
                for field, item in manifest["artifacts"].items():
                    path = directory / item["file"]
                    if path.name != item["file"] or digest(path.read_bytes()) != item["sha256"]:
                        raise ValueError("hash mismatch")
                    matrix = load_npz(path).tocsc()
                    terms = item["terms"]
                    if field not in statistics or terms != sorted(statistics[field][3]):
                        raise ValueError("vocabulary/field mismatch")
                    matrix.check_format(full_check=True)
                    if matrix.shape != (len(chunks), len(terms)) or len(set(terms)) != len(terms) or not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
                        raise ValueError("shape/value mismatch")
                    self.matrices[field] = matrix
                    self.vocabulary[field] = {t: i for i, t in enumerate(terms)}
                if set(self.matrices) != set(statistics):
                    raise ValueError("field mismatch")
                return
            except (ValueError, KeyError, OSError, TypeError) as exc:
                raise ValueError("BM25 index invalid; remove bm25-ready.json and rebuild: " + str(exc)) from None
        artifacts = {}
        directory.mkdir(parents=True, exist_ok=True)
        for field, (counts, lengths, average, idf) in statistics.items():
            terms = sorted(idf)
            lookup = {t: i for i, t in enumerate(terms)}
            rows, cols, values = [], [], []
            for row, counter in enumerate(counts):
                for term, freq in counter.items():
                    rows.append(row)
                    cols.append(lookup[term])
                    values.append(idf[term] * freq * 2.5 / (freq + 1.5 * (.25 + .75 * lengths[row] / average)))
            matrix = csc_matrix((values, (rows, cols)), shape=(len(chunks), len(terms)))
            filename = f"bm25-{identity['source']}-{field}.npz"
            fd, temp = tempfile.mkstemp(dir=directory, suffix=".npz")
            os.close(fd)
            try:
                save_npz(temp, matrix)
                os.replace(temp, directory / filename)
            finally:
                if os.path.exists(temp):
                    os.unlink(temp)
            artifacts[field] = {"file": filename, "sha256": digest((directory / filename).read_bytes()), "terms": terms}
            self.matrices[field], self.vocabulary[field] = matrix, lookup
        # Publish only after every field payload exists; readers see a complete generation.
        atomic_json(self.path, {"identity": identity, "artifacts": artifacts})

    def score(self, field, terms):
        matrix = self.matrices[field]
        scores = np.zeros(matrix.shape[0])
        for term in set(terms):
            col = self.vocabulary[field].get(term)
            if col is not None:
                start, end = matrix.indptr[col:col + 2]
                scores[matrix.indices[start:end]] += matrix.data[start:end]
        return scores
