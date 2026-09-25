"""
tests/test_memory_api.py — Phase 12 (P12) tests: the Memory page's three endpoints, GET/PUT/DELETE
/api/memory[/{key}]. Contract: docs/contracts.md § 15 "Phase 12: Memory page".

Like test_memory.py, most tests here drive the real FastAPI app end to end with `TestClient`: real
graph, real guard — only the model is the free `FakeChatModel`, and the memory store (and knowledge
base) are temp folders with instant fake embeddings, so a test run never reads or writes the owner's
real data/chroma.
"""

import json

from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding
from langgraph.checkpoint.memory import InMemorySaver

from artlab.api import create_app
from artlab.memory.store import MemoryStore
from artlab.model import fake_model
from artlab.rag.knowledge import KnowledgeBase


def parse_sse(body: str) -> list[tuple[str, dict]]:
    """Split a full SSE response body into [(event name, data dict), ...], in order (test_api.py)."""
    events = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["event"], json.loads(fields["data"])))
    return events


def send(client: TestClient, message: str) -> list[tuple[str, dict]]:
    """POST one message to a brand-new chat and return its parsed events."""
    response = client.post("/api/chat", json={"message": message, "thread_id": None})
    assert response.status_code == 200
    return parse_sse(response.text)


def empty_kb(tmp_path) -> KnowledgeBase:
    """An empty knowledge base in a temp folder with instant fake embeddings (test_api.py)."""
    return KnowledgeBase(tmp_path / "chroma", embeddings=DeterministicFakeEmbedding(size=32), min_score=-1)


def empty_memory(tmp_path) -> MemoryStore:
    """An empty long-term memory store in a temp folder with instant fake embeddings — the same
    shape `empty_kb` gives the knowledge base, so a test run never touches the owner's real memory."""
    return MemoryStore(tmp_path / "chroma-memory", embeddings=DeterministicFakeEmbedding(size=32))


def app_client(tmp_path, memory: MemoryStore | None = None) -> TestClient:
    """A TestClient wired to temp stores, so tests never touch real data (test_api.py's `client`
    fixture, inlined here since each test needs its own choice of pre-seeded memory)."""
    app = create_app(
        model=fake_model(), checkpointer=InMemorySaver(), knowledge=empty_kb(tmp_path),
        memory=memory or empty_memory(tmp_path), ideas_dir=tmp_path / "ideas",
    )
    return TestClient(app)


def test_list_shows_saved_facts_newest_first(tmp_path):
    """GET /api/memory returns every fact, newest first — the order the Memory page's card list
    relies on (docs/contracts.md § 15)."""
    memory = empty_memory(tmp_path)
    memory.save("niche", "budget desk gear", "t1")
    memory.save("audience", "new streamers", "t1")  # saved second, so it's newer

    with app_client(tmp_path, memory) as client:
        res = client.get("/api/memory")

    assert res.status_code == 200
    keys = [f["key"] for f in res.json()]
    assert keys == ["audience", "niche"]


def test_edit_changes_the_value_normalised(tmp_path):
    """PUT /api/memory/{key} changes the fact's value, running it through the store's own
    normalisation (stripped, capped at MAX_VALUE_CHARS) the same way `save` does."""
    memory = empty_memory(tmp_path)
    memory.save("niche", "budget desk gear", "t1")

    with app_client(tmp_path, memory) as client:
        res = client.put("/api/memory/niche", json={"value": "  premium desk gear  "})
        assert res.status_code == 200
        assert res.json()["value"] == "premium desk gear"

        listed = client.get("/api/memory").json()
        assert next(f for f in listed if f["key"] == "niche")["value"] == "premium desk gear"


def test_edit_unknown_key_is_404(tmp_path):
    """PUT /api/memory/{key} for a key that was never saved returns 404, not a silently-created
    fact — editing only ever touches something that already exists."""
    with app_client(tmp_path) as client:
        res = client.put("/api/memory/no_such_key", json={"value": "anything"})

    assert res.status_code == 404


def test_delete_returns_204_and_the_fact_is_gone(tmp_path):
    """DELETE /api/memory/{key} removes the fact (204) and it no longer shows up in the list."""
    memory = empty_memory(tmp_path)
    memory.save("niche", "budget desk gear", "t1")

    with app_client(tmp_path, memory) as client:
        res = client.delete("/api/memory/niche")
        assert res.status_code == 204

        listed = client.get("/api/memory").json()
        assert listed == []


def test_delete_unknown_key_is_404(tmp_path):
    """DELETE /api/memory/{key} for a key that doesn't exist returns 404, so a double-click or a
    stale page can't look like it deleted something."""
    with app_client(tmp_path) as client:
        res = client.delete("/api/memory/no_such_key")

    assert res.status_code == 404


def test_s12_deleting_a_fact_through_the_api_removes_it_from_recall(tmp_path):
    """S12: save "my niche is budget desk gear" through a chat message (the normal extraction path),
    delete it through the Memory page's DELETE endpoint, then start a brand-new chat (a different
    thread) and check its recall trace no longer mentions the niche — proving a Memory-page delete
    reaches the same store `recall` reads from, not just a copy."""
    with app_client(tmp_path) as client:
        events = send(client, "my niche is budget desk gear")
        save_trace = next(d for n, d in events if n == "trace" and d["stage"] == "memory" and d["detail"].startswith("saved"))
        assert save_trace["detail"] == "saved niche = budget desk gear"

        del_res = client.delete("/api/memory/niche")
        assert del_res.status_code == 204

        new_events = send(client, "Give me more video ideas")  # thread_id=None: a brand-new chat

    recall_trace = next(d for n, d in new_events if n == "trace" and d["stage"] == "memory" and d["detail"].startswith(("recalled", "no facts")))
    assert recall_trace["detail"] == "no facts yet"
