import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from blended_rag.ingestion import LocalStore, parse_catalog
from blended_rag.models import digest


@pytest.fixture
def source():
    return {"source_id": "test", "release": "rev5", "url": "https://example.org/catalog.json", "sha256": digest("source")}


@pytest.fixture
def document():
    return {"catalog": {"groups": [{"controls": [
        {"id": "ac-2", "title": "Account Management", "parts": [
            {"id": "ac-2_smt", "prose": "Manage user accounts and remove access when employees leave.",
             "parts": [{"id": "ac-2_smt.a", "prose": "Approve accounts before creation."}]}]},
        {"id": "cp-9", "title": "System Backup", "parts": [
            {"id": "cp-9_smt", "prose": "Back up system information and restore data after failure."}]}]}]}}


@pytest.fixture
def chunks(source, document):
    return parse_catalog(source, document)


@pytest.fixture
def store(tmp_path, chunks):
    store = LocalStore(tmp_path)
    store.ingest(chunks)
    return store
