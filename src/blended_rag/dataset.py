"""Pinned public corpus, checked before parsing. Bundled gzip supports offline use."""
import gzip
import json
from pathlib import Path

import requests

from .models import digest

DATA = Path(__file__).resolve().parents[1] / "data"


def manifest() -> dict:
    return json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))


def load_sources(download: bool = False):
    for source in manifest()["sources"]:
        path = DATA / (source["file"] + ".gz")
        if download:
            response = requests.get(source["url"], timeout=(10, 90))
            response.raise_for_status()
            raw = response.content
        else:
            raw = gzip.decompress(path.read_bytes())
        if digest(raw) != source["sha256"]:
            raise ValueError(f"Source hash mismatch: {source['release']}")
        if download:
            path.write_bytes(gzip.compress(raw, mtime=0))
        yield source, json.loads(raw)
