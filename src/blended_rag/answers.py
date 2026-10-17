import difflib
from dataclasses import asdict

from .models import validate_filters


def answer(retriever, query, method="hybrid", k=5, filters=None):
    hits = retriever.search(query, method, k, filters)
    citations = [{**asdict(hit.chunk), "score": hit.score} for hit in hits]
    releases = sorted({hit.chunk.release for hit in hits})
    return {"answer": "\n\n".join(f"[{i}] {h.chunk.text}" for i, h in enumerate(hits, 1)) if hits
            else "Insufficient evidence: no matching source excerpts were retrieved.",
            "mode": "extractive evidence; no LLM synthesis or entailment verification",
            "citations": citations, "releases": releases,
            "notice": "Multiple releases returned; treat excerpts independently and compare before applying." if len(releases) > 1 else ""}


def compare(chunks, control_id, old="rev4", new="rev5"):
    validate_filters({"release": old, "control_id": control_id})
    validate_filters({"release": new})
    if old == new:
        raise ValueError("Comparison requires two different releases")
    paired = {}
    for release in [old, new]:
        chosen = sorted((c for c in chunks if c.release == release and c.control_id == control_id), key=lambda c: c.section_id)
        paired[release] = {"present": bool(chosen), "excerpts": [asdict(c) for c in chosen]}
    before = [c["text"] for c in paired[old]["excerpts"]]
    after = [c["text"] for c in paired[new]["excerpts"]]
    return {"control_id": control_id, "releases": paired,
            "textual_diff": "\n".join(difflib.unified_diff(before, after, fromfile=old, tofile=new, lineterm="")),
            "notice": "Textual comparison only. Changes do not establish legal or operational policy changes."}
