from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass


def digest(text: str | bytes) -> str:
    return hashlib.sha256(text.encode() if isinstance(text, str) else text).hexdigest()


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source_id: str
    release: str
    control_id: str
    section_id: str
    title: str
    text: str
    source_url: str
    source_hash: str
    content_hash: str

    def __post_init__(self):
        if any(not isinstance(v, str) or not v.strip() for v in asdict(self).values()):
            raise ValueError("Chunk fields must be nonempty strings")
        if digest(self.text) != self.content_hash:
            raise ValueError("Invalid content hash")
        if not self.source_url.startswith("https://"):
            raise ValueError("Source URL must use HTTPS")


@dataclass
class Hit:
    chunk: Chunk
    score: float


def validate_filters(filters: dict | None) -> dict[str, str]:
    filters = filters or {}
    if set(filters) - {"release", "source_id", "control_id"}:
        raise ValueError("Allowed filters: release, source_id, control_id")
    if any(not isinstance(v, str) or not v.strip() for v in filters.values()):
        raise ValueError("Filter values must be nonempty strings")
    if "release" in filters and filters["release"] not in {"rev4", "rev5"}:
        raise ValueError("release must be rev4 or rev5")
    return filters


def validate_query(query: str, k: int):
    if not isinstance(query, str) or not query.strip() or len(query) > 4000:
        raise ValueError("Query must contain 1 to 4000 characters")
    if not 1 <= k <= 100:
        raise ValueError("k must be between 1 and 100")
