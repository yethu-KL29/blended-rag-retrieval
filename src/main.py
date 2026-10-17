"""Noninteractive reproducible CLI. Run from repository root."""
import argparse
import json
import os
import sys
from pathlib import Path

from pymongo.errors import PyMongoError

from blended_rag.answers import answer, compare
from blended_rag.dataset import load_sources
from blended_rag.evaluation import evaluate
from blended_rag.ingestion import LocalStore, parse_catalog
from blended_rag.models import Chunk
from blended_rag.retrieval import LocalRetriever, FaissRetriever
from blended_rag.providers import GeminiEncoder, ProviderError, provenance


def backend(store, name="offline", mode=None, answer_engine=None):
    if mode == "local":
        if answer_engine == "gemini-grounded":
            raise ValueError("Local mode requires extractive-nlp answers")
        return FaissRetriever(store)
    if mode == "enterprise":
        name = "atlas"
        for key, value in {"MONGODB_COLLECTION": "aintropy_blended_nist_gemini768_v1",
                           "ATLAS_TEXT_INDEX": "aintropy_nist_gemini_text_v1",
                           "ATLAS_VECTOR_INDEX": "aintropy_nist_gemini_vector_v1"}.items():
            os.environ.setdefault(key, value)
            if os.environ[key] != value:
                raise ValueError("Enterprise mode requires the dedicated Gemini namespaces")
    if name == "offline":
        return LocalRetriever(store)
    from blended_rag.atlas import AtlasRetriever
    retriever = AtlasRetriever(store)
    if mode == "enterprise":
        if not isinstance(retriever.encoder, GeminiEncoder):
            retriever.close()
            raise ValueError("Enterprise mode requires a Gemini store; use --mode enterprise ingest")
        retriever.mode = "enterprise"
        retriever.answer_engine = answer_engine or "gemini-grounded"
    return retriever


def load_env(path):
    # Explicit opt-in only; never echo credentials or include them in stored artifacts.
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            key, sep, value = line.partition("=")
            if not sep or not key.strip().replace("_", "").isalnum():
                raise ValueError("Invalid env-file entry")
            value = value.strip()
            if value.startswith('"'):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    raise ValueError("Invalid JSON quoted env-file value") from None
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            os.environ[key.strip()] = value


def parser():
    root = argparse.ArgumentParser(description="Version-aware NIST compliance evidence retrieval")
    root.add_argument("--store")
    root.add_argument("--mode", choices=["local", "enterprise"])
    root.add_argument("--env-file")
    root.add_argument("--answer-engine", choices=["extractive-nlp", "gemini-grounded"])
    root.add_argument("--backend", choices=["offline", "atlas"], default="offline")
    sub = root.add_subparsers(dest="command", required=True)
    sub.add_parser("download", help="Redownload pinned corpus and verify SHA256")
    ingestion = sub.add_parser("ingest")
    ingestion.add_argument("--encoder", choices=["lsa", "minilm", "onnx", "gemini"])
    ingestion.add_argument("--cache", default="src/assets/corpus-minilm.npz")
    export = sub.add_parser("export-cache")
    export.add_argument("--output", default="src/assets/corpus-minilm.npz")
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
    # Keep explicit legacy backend/encoder routes. New default commands use local mode.
    legacy = args.backend == "atlas" or getattr(args, "encoder", None) in {"lsa", "minilm"} or (args.store and not args.mode)
    mode = args.mode or (None if legacy else "local")
    if mode == "local" and (args.backend == "atlas" or args.answer_engine == "gemini-grounded"):
        print(json.dumps({"error": "Local mode requires offline retrieval and extractive-nlp answers", "mode": mode}), file=sys.stderr)
        return 2
    if args.env_file:
        try:
            load_env(args.env_file)
        except (OSError, ValueError):
            print(json.dumps({"error": "Cannot load explicit env-file; check path and KEY=value syntax", "mode": mode}), file=sys.stderr)
            return 2
    if mode == "enterprise":
        args.backend = "atlas"
        os.environ.setdefault("MONGODB_COLLECTION", "aintropy_blended_nist_gemini768_v1")
        os.environ.setdefault("ATLAS_TEXT_INDEX", "aintropy_nist_gemini_text_v1")
        os.environ.setdefault("ATLAS_VECTOR_INDEX", "aintropy_nist_gemini_vector_v1")
        if os.environ["MONGODB_COLLECTION"] != "aintropy_blended_nist_gemini768_v1":
            print(json.dumps({"error": "Enterprise mode requires the dedicated Gemini collection"}), file=sys.stderr)
            return 2
    args.store = args.store or (f"artifacts/runtime-{mode}" if mode else "artifacts/runtime")
    store = LocalStore(args.store)
    retriever = None
    try:
        if args.command == "download":
            result = {"verified_sources": [s["release"] for s, _ in load_sources(download=True)]}
        elif args.command == "ingest":
            chunks = [chunk for source, doc in load_sources() for chunk in parse_catalog(source, doc)]
            expected = {"local": "onnx", "enterprise": "gemini"}.get(mode)
            if expected and args.encoder and args.encoder != expected:
                raise ValueError("Explicit encoder conflicts with selected execution mode")
            result = store.ingest(chunks, expected or args.encoder or "lsa", args.cache if mode == "local" else None)
            if args.backend == "atlas":
                retriever = backend(store, "atlas", mode, args.answer_engine)
                result["atlas"] = retriever.ingest(store)
            result.update({"mode": mode or "legacy", "embedding_model": {"name": store.read()["model_name"], "fingerprint": store.read()["model_fingerprint"]},
                           "vector_engine": "faiss-IndexFlatIP" if mode == "local" else "mongodb-atlas-vector-search" if args.backend == "atlas" else "numpy-exact-cosine",
                           "answer_engine": args.answer_engine or ("gemini-grounded" if mode == "enterprise" else "extractive-nlp")})
            Path(args.store).mkdir(parents=True, exist_ok=True)
            (Path(args.store) / "last-ingestion.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        elif args.command == "export-cache":
            result = store.export_cache(args.output)
        elif args.command == "serve":
            import uvicorn
            from blended_rag.api import create_app
            uvicorn.run(create_app(args.store, args.backend, mode, args.answer_engine), host=args.host, port=args.port)
            return 0
        elif args.command == "compare":
            if args.backend == "atlas":
                retriever = backend(store, "atlas", mode, args.answer_engine)
                active = retriever._filters({})
                chunks = [Chunk(**{key: row[key] for key in Chunk.__dataclass_fields__})
                          for row in retriever.collection.find({**active, "control_id": args.control_id})]
            else:
                if mode == "local":
                    retriever = backend(store, args.backend, mode, args.answer_engine)
                    chunks = retriever.chunks
                else:
                    chunks = [Chunk(**c) for c in store.read()["chunks"]]
            result = compare(chunks, args.control_id, args.old, args.new)
        else:
            retriever = backend(store, args.backend, mode, args.answer_engine)
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
        if retriever:
            result.update(provenance(retriever))
        elif args.command != "download":
            state = store.read()
            result.update({"mode": mode or "legacy", "embedding_model": {"name": state["model_name"], "fingerprint": state["model_fingerprint"]},
                           "vector_engine": "faiss-IndexFlatIP" if mode == "local" else "mongodb-atlas-vector-search" if args.backend == "atlas" else "numpy-exact-cosine",
                           "answer_engine": args.answer_engine or ("gemini-grounded" if mode == "enterprise" else "extractive-nlp")})
        print(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False))
        return 0
    except (ValueError, OSError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2
    except PyMongoError as exc:
        print(json.dumps({"error": f"Atlas {type(exc).__name__}; check connectivity, permissions, and index readiness"}), file=sys.stderr)
        return 2
    except ProviderError as exc:
        print(json.dumps({"error": str(exc), "mode": mode}), file=sys.stderr)
        return 2
    finally:
        if retriever and hasattr(retriever, "close"):
            retriever.close()


if __name__ == "__main__":
    raise SystemExit(main())
