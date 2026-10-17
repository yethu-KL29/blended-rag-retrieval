"""Noninteractive reproducible CLI. Run from repository root."""
import argparse
import json
import sys
from pathlib import Path

from pymongo.errors import PyMongoError

from blended_rag.answers import answer, compare
from blended_rag.dataset import load_sources
from blended_rag.evaluation import evaluate
from blended_rag.ingestion import LocalStore, parse_catalog
from blended_rag.models import Chunk
from blended_rag.retrieval import LocalRetriever


def backend(store, name):
    if name == "offline":
        return LocalRetriever(store)
    from blended_rag.atlas import AtlasRetriever
    return AtlasRetriever(store)


def parser():
    root = argparse.ArgumentParser(description="Version-aware NIST compliance evidence retrieval")
    root.add_argument("--store", default="artifacts/runtime")
    root.add_argument("--backend", choices=["offline", "atlas"], default="offline")
    sub = root.add_subparsers(dest="command", required=True)
    sub.add_parser("download", help="Redownload pinned corpus and verify SHA256")
    ingestion = sub.add_parser("ingest")
    ingestion.add_argument("--encoder", choices=["lsa", "minilm"], default="lsa")
    sub.add_parser("indexes", help="Create Atlas Search indexes and wait for queryable readiness")
    query = sub.add_parser("query")
    query.add_argument("question")
    query.add_argument("--method", choices=["lexical", "dense", "hybrid"], default="hybrid")
    query.add_argument("--release", choices=["rev4", "rev5"])
    query.add_argument("--control-id")
    query.add_argument("--source-id")
    query.add_argument("-k", type=int, default=5)
    comparison = sub.add_parser("compare")
    comparison.add_argument("control_id")
    comparison.add_argument("--old", default="rev4", choices=["rev4", "rev5"])
    comparison.add_argument("--new", default="rev5", choices=["rev4", "rev5"])
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--output", default="artifacts/evaluation.json")
    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    store = LocalStore(args.store)
    retriever = None
    try:
        if args.command == "download":
            result = {"verified_sources": [s["release"] for s, _ in load_sources(download=True)]}
        elif args.command == "ingest":
            chunks = [chunk for source, doc in load_sources() for chunk in parse_catalog(source, doc)]
            result = store.ingest(chunks, args.encoder)
            if args.backend == "atlas":
                retriever = backend(store, "atlas")
                result["atlas"] = retriever.ingest(store)
            Path(args.store).mkdir(parents=True, exist_ok=True)
            (Path(args.store) / "last-ingestion.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        elif args.command == "serve":
            import uvicorn
            from blended_rag.api import create_app
            uvicorn.run(create_app(args.store, args.backend), host=args.host, port=args.port)
            return 0
        elif args.command == "compare":
            if args.backend == "atlas":
                retriever = backend(store, "atlas")
                active = retriever._filters({})
                chunks = [Chunk(**{key: row[key] for key in Chunk.__dataclass_fields__})
                          for row in retriever.collection.find({**active, "control_id": args.control_id})]
            else:
                chunks = [Chunk(**c) for c in store.read()["chunks"]]
            result = compare(chunks, args.control_id, args.old, args.new)
        else:
            retriever = backend(store, args.backend)
            if args.command == "indexes":
                if args.backend != "atlas":
                    raise ValueError("indexes requires --backend atlas")
                result = retriever.setup_indexes()
            elif args.command == "evaluate":
                result = evaluate(retriever, store, args.output)
            else:
                filters = {key: val for key, val in {"release": args.release, "control_id": args.control_id,
                                                   "source_id": args.source_id}.items() if val}
                result = answer(retriever, args.question, args.method, args.k, filters)
        print(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False))
        return 0
    except (ValueError, OSError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2
    except PyMongoError as exc:
        print(json.dumps({"error": f"Atlas {type(exc).__name__}; check connectivity, permissions, and index readiness"}), file=sys.stderr)
        return 2
    finally:
        if retriever and hasattr(retriever, "close"):
            retriever.close()


if __name__ == "__main__":
    raise SystemExit(main())
