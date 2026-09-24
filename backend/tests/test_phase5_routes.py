"""
tests/test_phase5_routes.py — T10 (Phase 5, last step) tests: the three Phase 5 workers
(youtube_researcher, content_ideator, english_coach) are real routes, not just nodes that exist —
the supervisor's fake router can actually reach them, GET /api/agents lists them with their real
tools, and the two routes that already existed (respond, rag_agent) still work exactly as before.

Like test_api.py, these drive the real FastAPI app end to end with `TestClient`: real graph, real
guard, real tool gateway, real (tiny, temp-folder) knowledge base — only the model is swapped for the
free `FakeChatModel`, so every test here is instant and costs $0 (CLAUDE.md's cost rule).

S1 and S4 are the design book's own scenarios for these two workers (docs/design.html): S1 is a trend
question that must reach youtube_researcher and use its tools; S4 is a grammar fix that must reach
english_coach and use none. The last test checks the fake router's own safety rule (model.py,
`_pick_route`/`_allowed_routes`): it must never hand back a route that the bound RouteDecision schema
doesn't actually allow, even when a graph is built with only some workers registered.
"""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from artlab.agents.workers import WORKERS
from artlab.api import create_app
from artlab.graph import build_graph
from artlab.model import fake_model
from artlab.rag.ingest import ingest
from artlab.rag.knowledge import KnowledgeBase
from artlab.tools.registry import ToolRegistry


def parse_sse(body: str) -> list[tuple[str, dict]]:
    """Split a full SSE response body into [(event name, data dict), ...], in order (test_api.py)."""
    events = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["event"], json.loads(fields["data"])))
    return events


def send(client: TestClient, message: str, thread_id: str | None = None) -> list[tuple[str, dict]]:
    """POST one chat message and return its parsed events. thread_id None starts a new chat."""
    response = client.post("/api/chat", json={"message": message, "thread_id": thread_id})
    assert response.status_code == 200
    return parse_sse(response.text)


def answer(events) -> str:
    """Join all `token` events back into the full answer text, as the browser does."""
    return "".join(data["text"] for name, data in events if name == "token")


def trace_lines(events) -> list[dict]:
    """The `trace` events' data dicts, in order — one per graph step."""
    return [data for name, data in events if name == "trace"]


def empty_kb(tmp_path) -> KnowledgeBase:
    """An empty knowledge base in a temp folder with instant fake embeddings (min_score -1: keep every
    hit) — the same helper test_api.py uses."""
    return KnowledgeBase(tmp_path / "chroma", embeddings=DeterministicFakeEmbedding(size=32), min_score=-1)


def is_tainted(client: TestClient, app, thread_id: str) -> bool:
    """Whether a chat is tainted right now, read straight from the graph's saved state — the same
    pattern test_api.py's `test_a_cited_answer_taints_the_chat_and_it_stays_tainted` uses.
    `client.portal` bridges this sync test into the checkpointer's async `aget_state`."""
    config = {"configurable": {"thread_id": thread_id}}
    return client.portal.call(app.state.graph.aget_state, config).values.get("tainted", False)


@pytest.fixture
def client(tmp_path):
    """A test client with an empty knowledge base and the default fake model — the same fixture shape
    test_api.py uses, for the tests below that don't need any real documents."""
    app = create_app(model=fake_model(), checkpointer=InMemorySaver(), knowledge=empty_kb(tmp_path))
    with TestClient(app) as c:
        yield c


def test_trending_question_routes_to_youtube_researcher_and_taints_the_chat(tmp_path):
    """S1: "What's trending in desk-setup videos?" must route to youtube_researcher, which runs the
    model-driven tool loop (docs/contracts.md § 9) against the real query_youtube_trends stub, and
    answer with real streamed text. The stub's output is untrusted by default, so the chat ends up
    tainted even though nothing in the message itself was suspicious — this is what will keep
    data-changing tools locked in this chat later (docs/contracts.md § 1)."""
    app = create_app(model=fake_model(), checkpointer=InMemorySaver(), knowledge=empty_kb(tmp_path))
    with TestClient(app) as client:
        events = send(client, "What's trending in desk-setup videos?")
        lines = trace_lines(events)

        assert any(d["stage"] == "arty" and d["detail"].startswith("→ youtube_researcher") for d in lines)
        assert any(d["stage"] == "tool" and d["detail"].startswith("query_youtube_trends") for d in lines)
        assert any(d["stage"] == "youtube_researcher" for d in lines)
        assert answer(events)  # a real answer streamed

        thread_id = events[0][1]["thread_id"]
        assert is_tainted(client, app, thread_id) is True


def test_fix_grammar_request_routes_to_english_coach_with_no_tools_and_no_taint(tmp_path):
    """S4: "Fix the grammar: me and him goes to shoot video tomorrow." must route to english_coach.
    english_coach has no tools (docs/contracts.md § 9's table), so the turn is exactly one model call:
    no `tool` trace line at all, and the chat stays untainted — nothing untrusted was read this turn."""
    app = create_app(model=fake_model(), checkpointer=InMemorySaver(), knowledge=empty_kb(tmp_path))
    with TestClient(app) as client:
        events = send(client, "Fix the grammar: me and him goes to shoot video tomorrow.")
        lines = trace_lines(events)

        assert any(d["stage"] == "arty" and d["detail"].startswith("→ english_coach") for d in lines)
        assert any(d["stage"] == "english_coach" for d in lines)
        assert not any(d["stage"] == "tool" for d in lines)
        assert answer(events)

        thread_id = events[0][1]["thread_id"]
        assert is_tainted(client, app, thread_id) is False


def test_video_ideas_request_routes_to_content_ideator(client):
    """A request for video ideas must route to content_ideator, not youtube_researcher or
    english_coach — the content_ideator hint (idea/hook/outline/brainstorm) has to win here, and
    nothing about "3" or "desk setups" should accidentally trip an earlier hint in the check order."""
    events = send(client, "Give me 3 video ideas about desk setups")
    lines = trace_lines(events)

    assert any(d["stage"] == "arty" and d["detail"].startswith("→ content_ideator") for d in lines)
    assert any(d["stage"] == "content_ideator" for d in lines)


def test_hi_still_routes_to_respond(client):
    """S11, unchanged: plain small talk ("hi") must still be answered by Arty himself. Adding three
    new hints ahead of KNOWLEDGE_HINTS must not change the common, no-hint-at-all case."""
    events = send(client, "hi")
    lines = trace_lines(events)

    assert any(d["stage"] == "arty" and d["detail"].startswith("answering myself") for d in lines)


def test_sponsorship_policy_question_still_routes_to_rag_agent(tmp_path):
    """rag_agent's own routing (the original KNOWLEDGE_HINTS) must still fire for a studio-policy
    question, unaffected by the three new hints now being checked ahead of it."""
    doc = tmp_path / "sponsorship.md"
    doc.write_text("# Sponsorship policy\n\n## Disclosure\n\nDisclose every sponsorship out loud.")
    kb = empty_kb(tmp_path)
    ingest([doc], kb)
    app = create_app(model=fake_model(), checkpointer=InMemorySaver(), knowledge=kb)
    with TestClient(app) as client:
        events = send(client, "What equipment sponsorships do we accept?")
        lines = trace_lines(events)

        assert any(d["stage"] == "arty" and d["detail"].startswith("→ rag_agent") for d in lines)


def test_get_agents_lists_every_phase5_worker_with_its_real_tools(tmp_path):
    """GET /api/agents (the sidebar's Team panel, X3) must list Arty, rag_agent and the three Phase 5
    workers, each with its real tools: youtube_researcher gets both YouTube stubs, read-only; the
    other two have none yet (their tools arrive in later phases, docs/contracts.md § 9's table)."""
    app = create_app(model=fake_model(), checkpointer=InMemorySaver(), knowledge=empty_kb(tmp_path))
    with TestClient(app) as client:
        body = client.get("/api/agents").json()

    assert [entry["name"] for entry in body] == [
        "Arty", "rag_agent", "youtube_researcher", "content_ideator", "english_coach",
    ]

    researcher = next(e for e in body if e["name"] == "youtube_researcher")
    assert {t["name"] for t in researcher["tools"]} == {"query_youtube_trends", "fetch_comments"}
    assert all(t["tier"] == "read-only" for t in researcher["tools"])

    ideator = next(e for e in body if e["name"] == "content_ideator")
    coach = next(e for e in body if e["name"] == "english_coach")
    assert ideator["tools"] == []
    assert coach["tools"] == []


def test_fake_router_never_picks_a_route_the_schema_disallows():
    """T10's schema guard (model.py, `_pick_route`/`_allowed_routes`): the fake router must never hand
    back a `next` value the bound RouteDecision schema doesn't actually accept. Here the graph is built
    with only "respond" registered, so english_coach isn't a valid route even though its keyword hint
    (grammar) fires first — the fake must fall through every later hint too and land on "respond",
    the only route this graph's schema allows, exactly like `workers=` stubs in test_supervisor.py."""
    respond_spec = next(w for w in WORKERS if w.name == "respond")
    graph = build_graph(fake_model(), InMemorySaver(), ToolRegistry(), workers=(respond_spec,))
    config = {"configurable": {"thread_id": "t"}}

    state = asyncio.run(graph.ainvoke(
        {"messages": [HumanMessage("Fix the grammar: me and him goes to shoot video tomorrow.")]}, config,
    ))

    assert state["answered_by"] == "respond"
