"""
Phase 7 wiring test: `summarize` really sits in the running graph (guard → summarize → recall), not just
in its own unit tests (tests/test_summarize.py). Drives the real API, like the browser, on the fake model.
"""

from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from artlab.agents.summarize import KEEP_RECENT, SUMMARY_TRIGGER
from artlab.api import create_app
from artlab.model import fake_model
from tests.test_memory import empty_kb, empty_memory, send


def test_a_long_chat_gets_summarized_through_the_real_graph(tmp_path):
    """Keep chatting past SUMMARY_TRIGGER messages: on the turn that crosses it, the trace shows a
    "summarized N messages" memory line, and the saved chat starts with the summary followed by exactly
    the KEEP_RECENT newest messages plus this turn's reply. Protects the wiring: without the
    guard → summarize edge, long chats would just keep growing and every model call would get dearer."""
    app = create_app(model=fake_model(["ok"]), checkpointer=InMemorySaver(), knowledge=empty_kb(tmp_path), memory=empty_memory(tmp_path))
    with TestClient(app) as client:
        thread_id = send(client, "hello there")[0][1]["thread_id"]
        summarized_on = None
        for turn in range(SUMMARY_TRIGGER):  # each turn adds 2 messages, so this crosses the trigger
            events = send(client, f"message number {turn}", thread_id)
            if any(d.get("stage") == "memory" and d["detail"].startswith("summarized") for n, d in events if n == "trace"):
                summarized_on = turn
                break

        assert summarized_on is not None, "the chat never got summarized"
        history = client.get(f"/api/threads/{thread_id}").json()["messages"]
        # After the fold: the summary, then the KEEP_RECENT kept messages, then this turn's answer.
        assert len(history) == 1 + KEEP_RECENT + 1
        assert history[0]["content"].startswith("Summary of the earlier conversation")
