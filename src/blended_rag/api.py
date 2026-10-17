from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from pymongo.errors import PyMongoError

from .answers import answer, compare
from .ingestion import LocalStore
from .models import Chunk


class Query(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=4000)
    method: Literal["lexical", "dense", "hybrid"] = "hybrid"
    k: int = Field(default=5, ge=1, le=100)
    filters: dict[str, str] = Field(default_factory=dict)


def create_app(directory="artifacts/runtime", backend="offline"):
    store = LocalStore(directory)

    @asynccontextmanager
    async def lifespan(app):
        from main import backend as make_backend
        app.state.retriever = make_backend(store, backend)
        yield
        if hasattr(app.state.retriever, "close"):
            app.state.retriever.close()

    app = FastAPI(title="Compliance evidence retrieval", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ready", "backend": backend, "model": app.state.retriever.model}

    @app.post("/query")
    def query(request: Query):
        try:
            return answer(app.state.retriever, request.question, request.method, request.k, request.filters)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        except PyMongoError:
            raise HTTPException(503, "Atlas retrieval failed; check service connectivity and index readiness") from None

    @app.get("/compare/{control_id}")
    def comparison(control_id: str, old: str = "rev4", new: str = "rev5"):
        try:
            if backend == "offline":
                chunks = app.state.retriever.chunks
            else:
                retriever = app.state.retriever
                active = retriever._filters({})
                chunks = [Chunk(**{key: row[key] for key in Chunk.__dataclass_fields__})
                          for row in retriever.collection.find({**active, "control_id": control_id})]
            return compare(chunks, control_id, old, new)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        except PyMongoError:
            raise HTTPException(503, "Atlas comparison failed") from None

    return app
