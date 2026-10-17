"""Rejected optional Atlas methods must never reach cloud initialization."""
import json

import pytest

from main import main


@pytest.mark.parametrize("method", ["sparse", "blended"])
def test_atlas_sparse_rejected_before_backend(monkeypatch, capsys, method):
    monkeypatch.setattr("main.backend", lambda *_: pytest.fail("Atlas initialized for unsupported sparse request"))
    result = main(["--backend", "atlas", "query", "accounts", "--method", method])
    assert result == 2
    assert "Atlas" in json.loads(capsys.readouterr().err)["error"]
