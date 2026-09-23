"""API tests. They use the fake model and an in-memory checkpointer: no API calls, no cost."""

import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langgraph.checkpoint.memory import InMemorySaver

from artlab.api import create_app
from artlab.graph import MAX_INPUT_CHARS
from artlab.model import fake_model


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["event"], json.loads(fields["data"])))
    return events


def send(client: TestClient, message: str, thread_id: str | None = None) -> list[tuple[str, dict]]:
    response = client.post("/api/chat", json={"message": message, "thread_id": thread_id})
    assert response.status_code == 200
    return parse_sse(response.text)


def answer(events) -> str:
    return "".join(data["text"] for name, data in events if name == "token")


@pytest.fixture
def client():
    app = create_app(model=fake_model(["Hello from the fake model."]), checkpointer=InMemorySaver())
    with TestClient(app) as c:
        yield c


def test_chat_streams_answer_and_trace(client):
    events = send(client, "hi")
    names = [name for name, _ in events]

    assert names[0] == "start" and names[-1] == "done"
    assert answer(events) == "Hello from the fake model."
    stages = [(d["stage"], d["status"]) for n, d in events if n == "trace"]
    assert stages == [("guard", "ok"), ("llm", "ok")]
    assert events[-1][1]["cost_usd"] == 0


def test_history_persists_across_turns(client):
    thread_id = send(client, "first")[0][1]["thread_id"]
    send(client, "second", thread_id)

    history = client.get(f"/api/threads/{thread_id}").json()["messages"]
    assert [(m["role"], m["content"]) for m in history] == [
        ("user", "first"),
        ("assistant", "Hello from the fake model."),
        ("user", "second"),
        ("assistant", "Hello from the fake model."),
    ]


def test_oversized_message_is_blocked_before_the_model():
    never_called = GenericFakeChatModel(messages=iter([]))  # raises if the graph ever calls it
    app = create_app(model=never_called, checkpointer=InMemorySaver())
    with TestClient(app) as client:
        events = send(client, "x" * (MAX_INPUT_CHARS + 1))

        assert "error" not in [name for name, _ in events]
        traces = [d for n, d in events if n == "trace"]
        assert [(t["stage"], t["status"]) for t in traces] == [("guard", "blocked")]
        assert answer(events).startswith("Blocked by the input guard")

        thread_id = events[0][1]["thread_id"]
        history = client.get(f"/api/threads/{thread_id}").json()["messages"]
        assert [m["role"] for m in history] == ["assistant"]  # the oversized message was dropped


def test_threads_listed_newest_first_with_titles(client):
    older = send(client, "about cable trays")[0][1]["thread_id"]
    newer = send(client, "about desk lamps")[0][1]["thread_id"]

    threads = client.get("/api/threads").json()
    assert [(t["thread_id"], t["title"]) for t in threads] == [
        (newer, "about desk lamps"),
        (older, "about cable trays"),
    ]


def test_unknown_thread_is_404(client):
    assert client.get("/api/threads/does-not-exist").status_code == 404
