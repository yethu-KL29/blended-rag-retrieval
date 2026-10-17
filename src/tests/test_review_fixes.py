import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Lock
from types import SimpleNamespace

from fastapi.testclient import TestClient

from blended_rag.api import create_app
from blended_rag.ingestion import LocalStore
from blended_rag.models import Hit
from blended_rag.providers import GeminiEncoder


def test_ingestion_preserves_unknown_billable_characters(tmp_path, chunks, monkeypatch):
    def client(self):
        return SimpleNamespace(models=SimpleNamespace(embed_content=lambda **kwargs:
            SimpleNamespace(metadata=None, embeddings=[SimpleNamespace(values=[1.] * self.dimensions)
                                                        for _ in kwargs["contents"]])))
    monkeypatch.setattr(GeminiEncoder, "client", client)
    result = LocalStore(tmp_path / "enterprise").ingest(chunks, "gemini")
    assert result["usage"]["embedding_requests"] == 1
    assert result["usage"]["embedding_characters"] > 0
    assert result["usage"]["embedding_billable_characters"] is None
    assert json.loads(json.dumps(result))["usage"]["embedding_billable_characters"] is None


def test_api_concurrent_requests_attribute_retrieval_and_synthesis_once(tmp_path, chunks, monkeypatch):
    encoder = GeminiEncoder()
    encoder._client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kwargs:
        SimpleNamespace(text=json.dumps({"answer": "Manage accounts [1]", "citation_ids": [1], "insufficient": False}),
                        usage_metadata=SimpleNamespace(prompt_token_count=12, candidates_token_count=4))))
    second_search = Event()
    counter_lock = Lock()
    state = {"active": 0, "peak": 0, "calls": 0}

    class MeasuredRetriever:
        mode = "enterprise"
        answer_engine = "gemini-grounded"
        model = {"name": "test"}

        def search(self, *args):
            with counter_lock:
                state["calls"] += 1
                call = state["calls"]
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
                encoder.usage["embedding_requests"] += 1
            if call == 1:
                # Without API serialization, the second search enters this window and
                # both requests observe its usage. With the lock it must wait.
                second_search.wait(timeout=.25)
            else:
                second_search.set()
            with counter_lock:
                state["active"] -= 1
            return [Hit(chunks[0], 1.)]

    retriever = MeasuredRetriever()
    retriever.encoder = encoder
    monkeypatch.setattr("main.backend", lambda *args: retriever)
    start = Barrier(2)
    with TestClient(create_app(tmp_path, mode="enterprise")) as client:
        def request():
            start.wait(timeout=5)
            response = client.post("/query", json={"question": "account"})
            assert response.status_code == 200
            return response.json()
        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(pool.map(lambda _: request(), range(2)))

    assert state["calls"] == 2 and state["peak"] == 1
    for response in responses:
        assert response["usage"]["embedding_requests"] == 1
        assert response["usage"]["generation_requests"] == 1
        assert response["usage"]["input_tokens"] == 12
        assert response["usage"]["output_tokens"] == 4
    assert encoder.usage["embedding_requests"] == 2
    assert encoder.usage["generation_requests"] == 2
