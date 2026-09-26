"""
tests/test_delete_thread.py — deleting a chat from the sidebar: DELETE /api/threads/{thread_id}.

A chat's data lives in three stores, all keyed by its thread_id (see api.py → delete_thread): the
checkpointer (history), the run log (Runs page rows) and long-term memory (facts it saved). These
tests prove a delete reaches all three — and only that chat's data, never another chat's.

Like test_memory_api.py, they drive the real FastAPI app end to end with `TestClient`: real graph,
free `FakeChatModel`, and temp stores, so no test touches the owner's real data or costs anything.
"""

import json

from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding
from langgraph.checkpoint.memory import InMemorySaver

from artlab.api import create_app
from artlab.memory.store import MemoryStore
from artlab.model import fake_model
from artlab.rag.knowledge import KnowledgeBase
from artlab.runs.store import RunStore


def parse_sse(body: str) -> list[tuple[str, dict]]:
    """Split a full SSE response body into [(event name, data dict), ...], in order (test_api.py)."""
    events = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["event"], json.loads(fields["data"])))
    return events


def new_chat(client: TestClient, message: str) -> str:
    """Send `message` as the first message of a brand-new chat and return the chat's thread_id."""
    response = client.post("/api/chat", json={"message": message, "thread_id": None})
    assert response.status_code == 200
    return next(data["thread_id"] for name, data in parse_sse(response.text) if name == "start")


def temp_memory(tmp_path) -> MemoryStore:
    """A private memory store in a temp folder with instant fake embeddings (test_memory_api.py)."""
    return MemoryStore(tmp_path / "chroma-memory", embeddings=DeterministicFakeEmbedding(size=32))


def app_client(tmp_path, memory: MemoryStore, runs: RunStore) -> TestClient:
    """A TestClient on temp stores. The test keeps `memory` and `runs` to look inside them afterwards."""
    app = create_app(
        model=fake_model(), checkpointer=InMemorySaver(),
        knowledge=KnowledgeBase(tmp_path / "chroma", embeddings=DeterministicFakeEmbedding(size=32), min_score=-1),
        memory=memory, runs=runs, ideas_dir=tmp_path / "ideas",
    )
    return TestClient(app)


def test_delete_removes_the_chat_from_the_sidebar_and_its_history(tmp_path):
    """After a delete, the chat is gone from GET /api/threads and its history is a 404 — the two
    things the sidebar and the chat pane read, so the UI can never show a deleted chat again."""
    with app_client(tmp_path, temp_memory(tmp_path), RunStore(tmp_path / "runs.db")) as client:
        thread_id = new_chat(client, "hello Arty")

        res = client.delete(f"/api/threads/{thread_id}")
        assert res.status_code == 204

        listed = [t["thread_id"] for t in client.get("/api/threads").json()]
        assert thread_id not in listed
        assert client.get(f"/api/threads/{thread_id}").status_code == 404


def test_delete_removes_only_that_chats_runs(tmp_path):
    """The Runs page loses the deleted chat's rows but keeps every other chat's — a delete must be
    scoped to one thread_id, not wipe the whole log."""
    runs = RunStore(tmp_path / "runs.db")
    with app_client(tmp_path, temp_memory(tmp_path), runs) as client:
        doomed = new_chat(client, "hello Arty")
        kept = new_chat(client, "hi again")

        assert client.delete(f"/api/threads/{doomed}").status_code == 204

    assert {r["thread_id"] for r in runs.list()} == {kept}


def test_delete_removes_facts_learned_in_that_chat_only(tmp_path):
    """A fact saved by the deleted chat (through the normal extraction path) is forgotten; a fact
    saved by another chat survives — long-term memory is shared, so the delete must stay precise."""
    memory = temp_memory(tmp_path)
    with app_client(tmp_path, memory, RunStore(tmp_path / "runs.db")) as client:
        doomed = new_chat(client, "my niche is budget desk gear")  # the fake model extracts niche=...
        memory.save("audience", "new streamers", "some-other-chat")
        assert {f["key"] for f in memory.all()} == {"niche", "audience"}

        assert client.delete(f"/api/threads/{doomed}").status_code == 204

    assert [f["key"] for f in memory.all()] == ["audience"]


def test_delete_unknown_chat_is_404(tmp_path):
    """Deleting an id that never existed (or was already deleted — a double-click) is a 404, so the
    UI can't mistake "nothing happened" for a successful delete."""
    with app_client(tmp_path, temp_memory(tmp_path), RunStore(tmp_path / "runs.db")) as client:
        assert client.delete("/api/threads/no-such-chat").status_code == 404


def test_fact_belongs_to_the_chat_that_saved_it_last(tmp_path):
    """Saving the same key from a second chat moves the fact to that chat (save() upserts by key and
    stamps the new thread_id). So deleting the FIRST chat must leave it — the newer chat still
    stands behind that fact — and deleting the second one removes it."""
    memory = temp_memory(tmp_path)
    memory.save("niche", "budget desk gear", "chat-a")
    memory.save("niche", "premium desk gear", "chat-b")

    assert memory.delete_thread("chat-a") == 0
    assert [f["value"] for f in memory.all()] == ["premium desk gear"]
    assert memory.delete_thread("chat-b") == 1
    assert memory.all() == []
