"""
tests/test_handoffs.py — H1 (Phase 9) tests: the supervisor plans a multi-step turn upfront, and each
step's answer is handed to the next step as an artifact (docs/contracts.md § 13).

Like test_phase5_routes.py, most tests here drive the real FastAPI app end to end with `TestClient`:
real graph, real guard, real tool gateway, real (tiny, temp-folder) knowledge base — only the model is
the free `FakeChatModel`, so every test is instant and costs $0 (CLAUDE.md's cost rule). The `then`
hygiene and breaker tests build the graph directly with stub workers instead, to isolate the
supervisor's own plan bookkeeping (`_clean_plan`, the artifact-passing logic) from the real workers'
tool loops.
"""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

import artlab.agents.supervisor as supervisor_module
from artlab.agents.workers import WorkerSpec
from artlab.api import create_app
from artlab.graph import build_graph
from artlab.memory.store import MemoryStore
from artlab.model import FakeChatModel, fake_model
from artlab.rag.knowledge import KnowledgeBase
from artlab.tools.registry import ToolRegistry


# ── TestClient helpers (test_phase5_routes.py) ──────────────────────────────────────────────────────

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
    """An empty knowledge base in a temp folder with instant fake embeddings (test_phase5_routes.py)."""
    return KnowledgeBase(tmp_path / "chroma", embeddings=DeterministicFakeEmbedding(size=32), min_score=-1)


def empty_memory(tmp_path) -> MemoryStore:
    """A private, temp-folder memory store with instant fake embeddings — every real graph needs one
    now that `recall`/`remember` are permanent nodes (test_memory.py)."""
    return MemoryStore(tmp_path / "chroma-memory", embeddings=DeterministicFakeEmbedding(size=32))


def is_tainted(client: TestClient, app, thread_id: str) -> bool:
    """Whether a chat is tainted right now, read straight from the graph's saved state
    (test_phase5_routes.py)."""
    config = {"configurable": {"thread_id": thread_id}}
    return client.portal.call(app.state.graph.aget_state, config).values.get("tainted", False)


@pytest.fixture
def client(tmp_path):
    """A test client with an empty knowledge base, a temp ideas folder and the default fake model —
    every real worker registered, so a plan can actually run start to finish."""
    app = create_app(
        model=fake_model(), checkpointer=InMemorySaver(), knowledge=empty_kb(tmp_path),
        memory=empty_memory(tmp_path), ideas_dir=tmp_path / "ideas",
    )
    with TestClient(app) as c:
        yield c


# ── Direct-graph helpers (test_supervisor.py) ───────────────────────────────────────────────────────

def build(model, tmp_path, workers):
    """A graph with the given model and worker registry, free storage, and no real tools."""
    return build_graph(model, InMemorySaver(), ToolRegistry(), empty_memory(tmp_path), workers=workers)


async def _run_traced(graph, message: str, thread_id: str = "t") -> tuple[list[dict], dict]:
    """Run one message, collecting every trace line alongside the final state (test_supervisor.py)."""
    config = {"configurable": {"thread_id": thread_id}}
    traces = [chunk async for chunk in graph.astream({"messages": [HumanMessage(message)]}, config, stream_mode="custom")]
    state = await graph.aget_state(config)
    return traces, state.values


def run_traced(graph, message: str, thread_id: str = "t") -> tuple[list[dict], dict]:
    return asyncio.run(_run_traced(graph, message, thread_id))


def stub_worker(name: str, calls: list[str]) -> WorkerSpec:
    """A worker that records that it ran and answers with a fixed message, then finishes (no
    `handoff`) — the supervisor's own `plan` state is what drives it to the next step, not the worker."""

    async def node(state):
        calls.append(name)
        return {"messages": [AIMessage(f"{name} answered")], "answered_by": name}

    return WorkerSpec(name=name, description=f"stub worker {name}", make_node=lambda model, tools: node)


class RouteCountingModel(FakeChatModel):
    """The fake model, but it counts every time it's asked to fill in RouteDecision — proves that
    running a whole multi-step plan needs exactly one routing call (H1, docs/contracts.md § 13),
    because the plan is decided once upfront, not re-asked per step."""

    route_calls: list[str] = Field(default_factory=list)

    def _reply(self, messages):
        if self.tool_name == "RouteDecision":
            self.route_calls.append("route")
        return super()._reply(messages)


class HumanMessageSpy(FakeChatModel):
    """The fake model, but it records every HumanMessage it's asked to answer — including the tool
    loop's own `HumanMessage(task)` (agents/tool_loop.py) — so a test can inspect exactly what task
    text a worker (not just the supervisor) was given."""

    human_messages: list[str] = Field(default_factory=list)

    def _reply(self, messages):
        human = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        if human is not None:
            self.human_messages.append(str(human))
        return super()._reply(messages)


def test_plan_runs_every_step_in_order_with_one_routing_call_and_taints_the_chat(tmp_path):
    """S2: "Find a niche in budget desk gear and give me 3 polished ideas" needs all three workers.
    The trace must show the whole plan on the routing line, the three workers must answer in that
    order, and — because the plan is decided once upfront, not re-asked after every step
    (docs/contracts.md § 13) — the supervisor's RouteDecision must be filled in exactly once. The
    research step reads real (untrusted) tool output, so the chat ends up tainted even though nothing
    in the request itself was suspicious."""
    model = RouteCountingModel()
    app = create_app(
        model=model, checkpointer=InMemorySaver(), knowledge=empty_kb(tmp_path),
        memory=empty_memory(tmp_path), ideas_dir=tmp_path / "ideas",
    )
    with TestClient(app) as client:
        events = send(client, "Find a niche in budget desk gear and give me 3 polished ideas")
        lines = trace_lines(events)

        route_trace = next(d for d in lines if d["stage"] == "arty" and d["detail"].startswith("→ youtube_researcher"))
        assert "plan: youtube_researcher → content_ideator → english_coach" in route_trace["detail"]

        worker_stages = [d["stage"] for d in lines if d["stage"] in {"youtube_researcher", "content_ideator", "english_coach"}]
        assert list(dict.fromkeys(worker_stages)) == ["youtube_researcher", "content_ideator", "english_coach"]

        assert len(model.route_calls) == 1
        assert answer(events)

        thread_id = events[0][1]["thread_id"]
        assert is_tainted(client, app, thread_id) is True


def test_ideator_task_carries_the_wrapped_research_artifact(tmp_path):
    """The second step's task must show the first step's answer as "Input from youtube_researcher
    (research):", with the text wrapped in <untrusted_retrieval source="artifact:youtube_researcher">
    — an answer built from untrusted tool output is itself untrusted (CLAUDE.md), so content_ideator
    must see it as data, never as an instruction from Art Lab itself (docs/contracts.md § 13)."""
    spy = HumanMessageSpy()
    app = create_app(
        model=spy, checkpointer=InMemorySaver(), knowledge=empty_kb(tmp_path),
        memory=empty_memory(tmp_path), ideas_dir=tmp_path / "ideas",
    )
    with TestClient(app) as client:
        send(client, "Find a niche in budget desk gear and give me 3 polished ideas")

    ideator_task = next(m for m in spy.human_messages if "Input from youtube_researcher" in m)
    assert "Input from youtube_researcher (research):" in ideator_task
    assert '<untrusted_retrieval source="artifact:youtube_researcher">' in ideator_task


def test_an_untainted_artifact_is_passed_plain(tmp_path):
    """"Give me 3 video ideas and polish them" plans content_ideator → english_coach. content_ideator
    has no reason to taint a plain ideas answer (no tool it would call here is untrusted), so the
    coach's task must carry that answer as plain text, not wrapped in <untrusted_retrieval> — the
    wrapper is only for untrusted content (docs/contracts.md § 13)."""
    spy = HumanMessageSpy()
    app = create_app(
        model=spy, checkpointer=InMemorySaver(), knowledge=empty_kb(tmp_path),
        memory=empty_memory(tmp_path), ideas_dir=tmp_path / "ideas",
    )
    with TestClient(app) as client:
        events = send(client, "Give me 3 video ideas and polish them")
        thread_id = events[0][1]["thread_id"]
        assert is_tainted(client, app, thread_id) is False

    coach_task = next(m for m in spy.human_messages if "Input from content_ideator" in m)
    assert "Input from content_ideator (ideas):" in coach_task
    assert "<untrusted_retrieval" not in coach_task


def test_single_step_requests_are_unchanged(client):
    """Regression: plain small talk still goes straight to `respond`, and a grammar fix still goes
    straight to english_coach — single-hint routing behaves exactly as it did before H1, and neither
    trace line mentions a plan (docs/contracts.md § 13: "leave empty for single-step requests")."""
    events = send(client, "hi")
    route_trace = next(d for d in trace_lines(events) if d["stage"] == "arty" and d["detail"].startswith("answering myself"))
    assert "plan:" not in route_trace["detail"]

    events = send(client, "Fix the grammar: me and him goes to shoot video tomorrow.")
    route_trace = next(d for d in trace_lines(events) if d["stage"] == "arty" and d["detail"].startswith("→ english_coach"))
    assert "plan:" not in route_trace["detail"]


def test_then_hygiene_cleans_duplicates_and_invalid_entries(tmp_path):
    """A scripted RouteDecision hands back a messy `then`: it repeats `next`, names "respond", and
    repeats a worker. Code — not the model — must clean this into at most MAX_PLAN - 1 = 2 distinct,
    real plan steps (`_clean_plan`, docs/contracts.md § 13), and the whole cleaned plan must actually
    run, in order, with nothing extra."""

    class MessyThen(FakeChatModel):
        """Always routes to "A" with a `then` designed to need every cleaning rule at once."""

        def _reply(self, messages):
            if self.tool_name != "RouteDecision":
                return super()._reply(messages)
            return AIMessage("", tool_calls=[{
                "name": self.tool_name,
                "args": {
                    "next": "A", "then": ["A", "respond", "B", "B", "C"],
                    "reason": "messy then", "question": "go",
                },
                "id": "fake-call", "type": "tool_call",
            }])

    calls: list[str] = []
    workers = (
        stub_worker("respond", calls), stub_worker("A", calls),
        stub_worker("B", calls), stub_worker("C", calls),
    )
    traces, state = run_traced(build(MessyThen(), tmp_path, workers), "go")

    route_trace = next(t for t in traces if t["stage"] == "arty" and t["detail"].startswith("→ A"))
    assert "plan: A → B → C" in route_trace["detail"]  # "A" (repeats next) and "respond" dropped, "B" deduped

    assert calls == ["A", "B", "C"]
    assert state["plan"] == []  # the whole (cleaned) plan ran, nothing left over


def test_breaker_stops_a_runaway_plan(tmp_path, monkeypatch):
    """The step-limit circuit breaker (docs/contracts.md § 3) still applies to a planned run. With
    MAX_STEPS lowered to 2, a 3-step plan must stop after its second dispatch — the step-limit note
    appended, the third step never run — instead of the plan overriding the limit."""
    monkeypatch.setattr(supervisor_module, "MAX_STEPS", 2)

    class RouteToThreeStepPlan(FakeChatModel):
        """Always plans A -> B -> C, regardless of the message."""

        def _reply(self, messages):
            if self.tool_name != "RouteDecision":
                return super()._reply(messages)
            return AIMessage("", tool_calls=[{
                "name": self.tool_name,
                "args": {"next": "A", "then": ["B", "C"], "reason": "3-step plan", "question": "go"},
                "id": "fake-call", "type": "tool_call",
            }])

    calls: list[str] = []
    workers = (stub_worker("A", calls), stub_worker("B", calls), stub_worker("C", calls))
    traces, state = run_traced(build(RouteToThreeStepPlan(), tmp_path, workers), "go")

    assert calls == ["A", "B"]  # C never ran: the breaker stopped the plan after step 2
    assert state["messages"][-1].content == supervisor_module.STEP_LIMIT_NOTICE
    assert [t["status"] for t in traces].count("stopped") == 1
