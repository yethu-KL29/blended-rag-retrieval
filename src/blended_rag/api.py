from contextlib import asynccontextmanager
from threading import Lock
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from pymongo.errors import PyMongoError

from .answers import answer, compare
from .ingestion import LocalStore
from .models import Chunk
from .providers import ProviderError, provenance


class Query(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=4000)
    method: Literal["lexical", "dense", "hybrid"] = "hybrid"
    k: int = Field(default=5, ge=1, le=100)
    filters: dict[str, str] = Field(default_factory=dict)


def create_app(directory=None, backend="offline", mode=None, answer_engine=None):
    mode = mode or ("local" if directory is None and backend == "offline" else None)
    store = LocalStore(directory or f"artifacts/runtime-{mode}")

    @asynccontextmanager
    async def lifespan(app):
        from main import backend as make_backend
        app.state.retriever = make_backend(store, backend, mode, answer_engine)
        app.state.query_lock = Lock()
        yield
        if hasattr(app.state.retriever, "close"):
            app.state.retriever.close()

    app = FastAPI(title="Compliance evidence retrieval", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ready", "backend": backend, "model": app.state.retriever.model, **provenance(app.state.retriever)}

    @app.post("/query")
    def query(request: Query):
        try:
            # The shared provider counters span retrieval and synthesis: keep both snapshots
            # and all provider work in the same per-worker critical section.
            with app.state.query_lock:
                return answer(app.state.retriever, request.question, request.method, request.k, request.filters)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        except PyMongoError:
            raise HTTPException(503, "Atlas retrieval failed; check service connectivity and index readiness") from None
        except ProviderError as exc:
            raise HTTPException(503, str(exc)) from None

    @app.get("/compare/{control_id}")
    def comparison(control_id: str, old: str = "rev4", new: str = "rev5"):
        try:
            if backend == "offline" and mode != "enterprise":
                chunks = app.state.retriever.chunks
            else:
                retriever = app.state.retriever
                active = retriever._filters({})
                chunks = [Chunk(**{key: row[key] for key in Chunk.__dataclass_fields__})
                          for row in retriever.collection.find({**active, "control_id": control_id})]
            return {**compare(chunks, control_id, old, new), **provenance(app.state.retriever)}
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        except PyMongoError:
            raise HTTPException(503, "Atlas comparison failed") from None

    return app
