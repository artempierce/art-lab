"""
tests/test_caps.py — G2 (Phase 10) tests: a turn stops at MAX_TURN_SECONDS or MAX_TURN_USD, keeping
whatever answer has already been produced (docs/contracts.md § 14, design book FR-5).

Like test_tool_loop.py, the tool-loop-level tests build the smallest graph that can give
`run_tool_loop` a stream writer (`get_stream_writer()` only works inside a running graph) and use tiny
scripted subclasses of the free `FakeChatModel` with `usage_metadata`, so a model call has a real,
non-zero cost without ever calling a real, paid model (CLAUDE.md's cost rule). The supervisor-level
tests build the real graph directly with stub workers, the same way test_handoffs.py's own breaker
test and test_output_guard.py's step-limit test do. Every clock check is done by monkeypatching
`time.time` (never `time.sleep`), so these tests are instant.
"""

import asyncio
import time

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from pydantic import Field

import artlab.guards.caps as caps
from artlab.agents.state import ChatState
from artlab.agents.tool_loop import LoopResult, run_tool_loop
from artlab.agents.workers import WorkerSpec
from artlab.api import create_app
from artlab.graph import build_graph
from artlab.memory.store import MemoryStore
from artlab.model import FakeChatModel, cost_usd, fake_model
from artlab.rag.knowledge import KnowledgeBase
from artlab.tools.agents import make_agent_tool
from artlab.tools.registry import ToolRegistry

# One model call's fixed cost in every scripted model below (100 in / 20 out tokens, the same numbers
# test_tool_loop.py's TokenCountingModel uses) — a plain constant so every test's math is legible.
CALL_COST = cost_usd(100, 20)
USAGE = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}


# ── tool-loop helpers (test_tool_loop.py's own `run_loop`, plus a `limits` kwarg) ───────────────────

def run_loop(model, tools: ToolRegistry, agent: str, task: str, limits: caps.Limits | None = None) -> tuple[LoopResult, list[dict]]:
    """Run `run_tool_loop` once and return (its `LoopResult`, the trace lines it wrote) — identical to
    test_tool_loop.py's `run_loop`, with one addition: a `limits` kwarg, so a test can hand
    `run_tool_loop` a turn budget without going through a whole graph/supervisor turn."""
    results: list[LoopResult] = []

    async def node(state: ChatState) -> dict:
        r = await run_tool_loop(model, tools, agent, "You are a test worker.", state["task"], tainted_in=False, limits=limits)
        results.append(r)
        return {"messages": [r.reply], "spent_usd": r.spent_usd, "answered_by": agent}

    async def _run() -> list[dict]:
        graph = (
            StateGraph(ChatState)
            .add_node(agent, node)
            .add_edge(START, agent)
            .compile(checkpointer=InMemorySaver())
        )
        config = {"configurable": {"thread_id": "test"}}
        traces: list[dict] = []
        async for mode, chunk in graph.astream({"messages": [HumanMessage(task)], "task": task}, config, stream_mode=["custom", "values"]):
            if mode == "custom":
                traces.append(chunk)
        return traces

    traces = asyncio.run(_run())
    return results[0], traces


class CostlyAlwaysAsksForATool(FakeChatModel):
    """Like test_tool_loop.py's `AlwaysAsksForATool` — ignores the fake model's own "one tool, then
    answer" restraint and keeps asking for a tool for as long as any are bound — but every reply also
    carries `usage_metadata`, so each model call has a fixed, non-zero dollar cost (CALL_COST). Used to
    prove the turn's own COST cap (guards/caps.py), not MAX_TOOL_CALLS, is what can stop a tool loop
    mid-way."""

    calls: list[int] = Field(default_factory=list)

    def _reply(self, messages):
        self.calls.append(1)
        if self.tool_specs:
            name = self.tool_specs[0]["function"]["name"]
            return AIMessage(
                "", tool_calls=[{"name": name, "args": {"query": "x"}, "id": f"call-{len(self.calls)}", "type": "tool_call"}],
                usage_metadata=USAGE,
            )
        return AIMessage("done", usage_metadata=USAGE)


def test_cost_cap_stops_a_tool_loop_after_the_third_model_call():
    """docs/contracts.md § 14: `run_tool_loop` checks `limits.spend_left` before every model call, not
    just MAX_TOOL_CALLS. With a limit of 2.5 x CALL_COST, the running spend after calls 1, 2 and 3 is
    1x, 2x and 3x CALL_COST — none of the first three checks (0, 1x, 2x against 2.5x) trip, but the
    check before what would be call 4 (3x >= 2.5x) does: the loop must stop right there, having made
    exactly 3 model calls, with trace status "stopped" and the reply set to the stop notice — not the
    model's own "done" text, and not run to MAX_TOOL_CALLS + 1 calls the way test_tool_loop.py's
    identical-looking scenario (with no cap) does."""
    model = CostlyAlwaysAsksForATool()
    tools = ToolRegistry()
    tools.register("echo", lambda query: f"result for {query}", "read_only", {"test_agent"}, "test tool")
    limits = caps.Limits(deadline=time.time() + 1000, spend_left=2.5 * CALL_COST)

    result, traces = run_loop(model, tools, "test_agent", "hello", limits=limits)

    assert len(model.calls) == 3  # the loop stopped before a 4th model call
    assert result.tool_calls == 3  # all three tools the model asked for before the stop actually ran
    assert result.reply.content == caps.STOP_NOTICE.format(kind="cost")
    stopped = [t for t in traces if t["stage"] == "test_agent" and t["status"] == "stopped"]
    assert len(stopped) == 1 and "cost" in stopped[0]["detail"]


# ── nested agent call: the caller's remaining budget, not a fresh one ──────────────────────────────

class NestedSpendModel(FakeChatModel):
    """Plays both sides of a nested `ask_<callee>` call at once, told apart by which tools are bound
    (docs/contracts.md § 13): as the OUTER caller (bound to `ask_callee`) it asks for it once, then
    (if ever asked again) answers "done"; as the INNER callee (bound to `echo`) it keeps asking for
    `echo`, the same "never stops on its own" trick `CostlyAlwaysAsksForATool` uses above — so the
    INNER loop only ever stops because its own cap says so, never because the model ran out of things
    to ask for. Every reply carries the same `usage_metadata`, so every model call, outer or nested,
    costs exactly CALL_COST."""

    def _reply(self, messages):
        names = [spec["function"]["name"] for spec in (self.tool_specs or [])]
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        if "ask_callee" in names and not tool_msgs:
            return AIMessage(
                "", tool_calls=[{"name": "ask_callee", "args": {"question": "go"}, "id": "call-outer", "type": "tool_call"}],
                usage_metadata=USAGE,
            )
        if "echo" in names:
            return AIMessage(
                "", tool_calls=[{"name": "echo", "args": {"query": "x"}, "id": f"call-{len(tool_msgs)}", "type": "tool_call"}],
                usage_metadata=USAGE,
            )
        return AIMessage("done", usage_metadata=USAGE)


def test_nested_agent_call_respects_the_callers_remaining_budget():
    """docs/contracts.md § 14, ticket G2: an `ask_<callee>` tool's nested loop must run against the
    CALLER's remaining budget, not a fresh MAX_TURN_USD of its own — otherwise a nested call could
    always spend a full turn's worth on top of whatever the caller already spent.

    The caller's own turn limit is 1.5 x CALL_COST. Its first (and only) model call spends 1x, leaving
    0.5x for the nested call — carried through `tools/agents.py`'s `ask()` via `tool_loop.limits_ctx`,
    the same contextvar trick `caller_ctx`/`tainted_in_ctx` already use. The nested ("callee") loop's
    own first model call (1x) already exceeds that 0.5x, so its check before a SECOND model call must
    stop it — proving the caller's remaining budget, not a fresh one, reached the nested loop at all,
    since nothing in this test ever builds a `Limits` for "callee" directly."""
    model = NestedSpendModel()
    tools = ToolRegistry()
    ask_callee = make_agent_tool(model, tools, "callee", "You are callee.")
    tools.register("ask_callee", ask_callee, "read_only", {"caller"}, "ask callee")
    tools.register("echo", lambda query: f"result for {query}", "read_only", {"callee"}, "test tool")
    limits = caps.Limits(deadline=time.time() + 1000, spend_left=1.5 * CALL_COST)

    result, traces = run_loop(model, tools, "caller", "go", limits=limits)

    nested_stopped = [t for t in traces if t["stage"] == "callee" and t["status"] == "stopped"]
    assert len(nested_stopped) == 1
    assert "cost" in nested_stopped[0]["detail"]
    assert "↳ for caller" in nested_stopped[0]["detail"]  # it really was the nested call, not caller's own
    # The caller's own running spend includes the nested loop's spend (docs/contracts.md § 13's shared
    # budget) — one outer call plus one nested call, both CALL_COST, nothing more.
    assert result.spent_usd == pytest.approx(2 * CALL_COST)


# ── supervisor-level checks (direct graph, like test_handoffs.py's / test_output_guard.py's breaker tests) ──

def stub_worker(name: str, calls: list[str]) -> WorkerSpec:
    """A worker that records that it ran and answers with a fixed message, then finishes — the same
    helper test_handoffs.py and test_output_guard.py each define locally."""

    async def node(state):
        calls.append(name)
        return {"messages": [AIMessage(f"{name} answered")], "answered_by": name}

    return WorkerSpec(name=name, description=f"stub worker {name}", make_node=lambda model, tools: node)


def empty_memory(tmp_path) -> MemoryStore:
    """A private, temp-folder memory store with instant fake embeddings (every real graph needs one)."""
    return MemoryStore(tmp_path / "chroma-memory", embeddings=DeterministicFakeEmbedding(size=32))


async def _run_traced(graph, message: str, thread_id: str = "t") -> tuple[list[dict], dict]:
    config = {"configurable": {"thread_id": thread_id}}
    traces = [chunk async for chunk in graph.astream({"messages": [HumanMessage(message)]}, config, stream_mode="custom")]
    state = await graph.aget_state(config)
    return traces, state.values


def run_traced(graph, message: str, thread_id: str = "t") -> tuple[list[dict], dict]:
    return asyncio.run(_run_traced(graph, message, thread_id))


class RouteToTwoStepPlan(FakeChatModel):
    """Always plans A -> B, regardless of the message (test_handoffs.py's own `RouteToThreeStepPlan`,
    shortened to two steps — this test only needs a second dispatch to check the cap in front of)."""

    def _reply(self, messages):
        if self.tool_name != "RouteDecision":
            return super()._reply(messages)
        return AIMessage("", tool_calls=[{
            "name": self.tool_name,
            "args": {"next": "A", "then": ["B"], "reason": "2-step plan", "question": "go"},
            "id": "fake-call", "type": "tool_call",
        }])


def test_time_cap_in_supervisor_stops_before_the_next_plan_step(tmp_path, monkeypatch):
    """docs/contracts.md § 14: the supervisor checks the turn caps before every dispatch, including a
    planned step (branch A). `time.time` is monkeypatched (never `time.sleep`) to still return the
    turn's own start (`t0`) until worker A has run, then a time past MAX_TURN_SECONDS after that — so
    the deadline is only "in the past" once the supervisor comes back to dispatch step B (tied to
    `ran`, not a call count, since `caps.check`'s own `dict.get(..., time.time())` fallback means the
    exact number of `time.time()` calls per check isn't part of this file's contract). Step A must
    still run (the turn wasn't over budget when it was dispatched); step B must never run; the stop
    notice must replace what would have been the next answer; and — Phase 10's "stop paths go to
    output_guard too" rule — an "output ok" guard trace line must still follow the stop."""
    t0 = time.time()
    ran: list[str] = []

    def fake_time():
        return t0 + caps.MAX_TURN_SECONDS + 1 if ran else t0

    monkeypatch.setattr(caps.time, "time", fake_time)

    workers = (stub_worker("A", ran), stub_worker("B", ran))
    graph = build_graph(RouteToTwoStepPlan(), InMemorySaver(), ToolRegistry(), empty_memory(tmp_path), workers=workers)
    traces, state = run_traced(graph, "go")

    assert ran == ["A"]  # B never ran: the time cap stopped the plan after step A
    assert state["messages"][-1].content == caps.STOP_NOTICE.format(kind="time")

    statuses = [t["status"] for t in traces]
    assert statuses.count("stopped") == 1
    stopped = next(t for t in traces if t["status"] == "stopped")
    assert "time" in stopped["detail"]

    guard_trace = next(t for t in traces if t["stage"] == "guard" and t["detail"].startswith("output"))
    assert guard_trace["status"] == "ok" and guard_trace["detail"] == "output ok"
    # The guard's line must come after the stop, not before — output_guard checks the notice too.
    assert traces.index(guard_trace) > traces.index(stopped)


# ── a normal turn never trips a cap, and the guard resets the clock each turn ───────────────────────

def _kb(tmp_path) -> KnowledgeBase:
    return KnowledgeBase(tmp_path / "chroma", embeddings=DeterministicFakeEmbedding(size=32), min_score=-1)


def parse_sse(body: str) -> list[tuple[str, dict]]:
    """Split a full SSE response body into [(event name, data dict), ...] (test_handoffs.py's own helper)."""
    import json

    events = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["event"], json.loads(fields["data"])))
    return events


def send(client: TestClient, message: str, thread_id: str | None = None) -> list[tuple[str, dict]]:
    response = client.post("/api/chat", json={"message": message, "thread_id": thread_id})
    assert response.status_code == 200
    return parse_sse(response.text)


def trace_lines(events) -> list[dict]:
    return [data for name, data in events if name == "trace"]


def test_a_normal_turn_never_trips_a_cap(tmp_path):
    """The common case: an ordinary message, answered well within 90 seconds and $0.05 (the fake model
    costs $0), must never produce a "stopped" trace line — turn caps are a safety net, not something a
    normal turn should ever brush against."""
    app = create_app(model=fake_model(), checkpointer=InMemorySaver(), knowledge=_kb(tmp_path), memory=empty_memory(tmp_path), ideas_dir=tmp_path / "ideas")
    with TestClient(app) as client:
        events = send(client, "hi")

    assert not any(d.get("status") == "stopped" for d in trace_lines(events))


def test_guard_resets_the_turn_start_each_turn(tmp_path, monkeypatch):
    """docs/contracts.md § 14: `agents/guard.py` sets `turn_started_at` fresh every turn. `time.time` is
    monkeypatched so that, between two turns of the SAME chat, "now" jumps forward by far more than
    MAX_TURN_SECONDS — standing in for a chat that's been sitting open a while. If the guard failed to
    reset `turn_started_at` on the second turn (reusing the first turn's stale start), the second turn
    would look instantly over the time cap. It must not: the guard's reset makes the jump irrelevant to
    the turn that starts after it."""
    real_time = time.time
    jumped = {"on": False}

    def fake_time():
        return real_time() + 200 if jumped["on"] else real_time()

    monkeypatch.setattr(caps.time, "time", fake_time)

    app = create_app(model=fake_model(), checkpointer=InMemorySaver(), knowledge=_kb(tmp_path), memory=empty_memory(tmp_path), ideas_dir=tmp_path / "ideas")
    with TestClient(app) as client:
        first = send(client, "hi")
        thread_id = first[0][1]["thread_id"]
        assert not any(d.get("status") == "stopped" for d in trace_lines(first))

        jumped["on"] = True  # simulate a long idle gap before the next message
        second = send(client, "hi again", thread_id=thread_id)

    assert not any(d.get("status") == "stopped" for d in trace_lines(second))
