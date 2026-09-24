"""
tests/test_supervisor.py — supervisor v2 tests (T1, Phase 3): worker registry, step-limit circuit
breaker, and the retry-then-fallback rule for invalid routing output.

Everything here runs the real graph (`build_graph`) with the free fake model, an in-memory
checkpointer, and — where a test needs to isolate routing/breaker behaviour from the real workers —
tiny stub `WorkerSpec`s instead of `respond`/`rag_agent`. No real API call is ever made.
Contract: docs/contracts.md § 3.
"""

import asyncio
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field, PrivateAttr

from artlab.agents.supervisor import STEP_LIMIT_NOTICE
from artlab.agents.workers import WORKERS, WorkerSpec
from artlab.graph import build_graph
from artlab.model import FakeChatModel, fake_model
from artlab.tools.registry import ToolRegistry


def build(model, workers=WORKERS):
    """A graph with the given model and worker registry, free storage, and no real tools."""
    return build_graph(model, InMemorySaver(), ToolRegistry(), workers=workers)


def run(graph, message: str, thread_id: str = "t") -> dict:
    """Run one message to completion and return the final state values (steps, answered_by, …)."""
    config = {"configurable": {"thread_id": thread_id}}
    return asyncio.run(graph.ainvoke({"messages": [HumanMessage(message)]}, config))


async def _run_traced(graph, message: str, thread_id: str) -> tuple[list[dict], dict]:
    """Run one message, collecting every trace line (the dicts nodes write with get_stream_writer())
    alongside the final state — the same "custom" stream api.py turns into `trace` SSE events."""
    config = {"configurable": {"thread_id": thread_id}}
    traces = [chunk async for chunk in graph.astream({"messages": [HumanMessage(message)]}, config, stream_mode="custom")]
    state = await graph.aget_state(config)
    return traces, state.values


def run_traced(graph, message: str, thread_id: str = "t") -> tuple[list[dict], dict]:
    return asyncio.run(_run_traced(graph, message, thread_id))


def stub_worker(name: str, calls: list[str], *, handoff: str = "") -> WorkerSpec:
    """A worker that records that it ran and answers with a fixed message. If `handoff` is set, it
    asks that worker to continue instead of finishing the turn — used to drive the step-limit test."""

    async def node(state):
        calls.append(name)
        update = {"messages": [AIMessage(f"{name} answered")], "answered_by": name}
        if handoff:
            update["handoff"] = handoff
        return update

    return WorkerSpec(name=name, description=f"stub worker {name}", make_node=lambda model, tools: node)


def test_hi_is_answered_by_respond_and_rag_agent_never_runs():
    """S11: a question with no knowledge-base hint ("hi") is routed to `respond`; `rag_agent` (here a
    spy standing in for it) must never run, and the turn counts as exactly one dispatch. Protects the
    common case: registry + breaker plumbing didn't change simple single-step routing."""
    calls: list[str] = []
    respond_spec = next(w for w in WORKERS if w.name == "respond")
    spy_rag = stub_worker("rag_agent", calls)

    state = run(build(fake_model(), workers=(respond_spec, spy_rag)), "hi")

    assert calls == []
    assert state["answered_by"] == "respond"
    assert state["steps"] == 1


def test_breaker_stops_a_handoff_loop_after_five_dispatches():
    """S7: two stub workers that always hand off to each other never finish on their own. The
    supervisor must stop them at MAX_STEPS (5) dispatches, append the step-limit note, and — because
    a handoff never needs a model call (docs/contracts.md § 3) — only the very first routing decision
    should ever ask the model; a second model call would mean the breaker isn't code-driven."""

    class RouteToAOnce(FakeChatModel):
        """Routes the very first message to worker "A". Any later call would mean the supervisor
        asked the model again for a handoff, which the contract says should never happen."""

        def _reply(self, messages):
            if self.tool_name == "RouteDecision":
                return AIMessage("", tool_calls=[{
                    "name": self.tool_name, "args": {"next": "A", "reason": "start", "question": "go"},
                    "id": "fake-call", "type": "tool_call",
                }])
            raise AssertionError("the model was called again after the first route (handoffs need no model)")

    calls: list[str] = []
    worker_a = stub_worker("A", calls, handoff="B")
    worker_b = stub_worker("B", calls, handoff="A")

    traces, state = run_traced(build(RouteToAOnce(), workers=(worker_a, worker_b)), "go")

    assert calls == ["A", "B", "A", "B", "A"]  # exactly 5 dispatches, then stopped
    assert state["messages"][-1].content == STEP_LIMIT_NOTICE
    assert [t["status"] for t in traces].count("stopped") == 1


def test_invalid_routing_output_is_retried_once_and_the_retry_is_used():
    """Invalid output once: the router's first reply names a worker that isn't registered, so it
    fails RouteDecision validation (parsed=None); the supervisor asks once more, and the retry's
    (valid) decision is the one actually used — not a fallback."""

    class FailsOnceThenRoutes(FakeChatModel):
        """First routing reply is invalid (an unregistered worker name); the second is a normal,
        valid decision to "respond"."""

        _attempt: Any = PrivateAttr(default_factory=lambda: iter([1, 2]))

        def _reply(self, messages):
            if self.tool_name != "RouteDecision":
                return super()._reply(messages)
            attempt = next(self._attempt)
            next_value = "not-a-real-worker" if attempt == 1 else "respond"
            return AIMessage("", tool_calls=[{
                "name": self.tool_name,
                "args": {"next": next_value, "reason": "picked on retry", "question": "hi"},
                "id": "fake-call", "type": "tool_call",
            }])

    traces, state = run_traced(build(FailsOnceThenRoutes()), "hi")

    assert state["answered_by"] == "respond"
    assert state["steps"] == 1
    route_trace = next(t for t in traces if t["stage"] == "arty" and "step 1/5" in t["detail"])
    assert "picked on retry" in route_trace["detail"]  # the retry's own reason, not a fallback


def test_invalid_routing_output_twice_falls_back_to_respond():
    """Invalid output twice: both routing replies name an unregistered worker, so both fail to
    parse. The supervisor doesn't fail the turn — it falls back to "respond" and says so in the
    trace line (docs/contracts.md § 3)."""

    class AlwaysInvalid(FakeChatModel):
        """Every routing reply names a worker that isn't registered, so it never parses."""

        def _reply(self, messages):
            if self.tool_name != "RouteDecision":
                return super()._reply(messages)
            return AIMessage("", tool_calls=[{
                "name": self.tool_name,
                "args": {"next": "not-a-real-worker", "reason": "bad", "question": "hi"},
                "id": "fake-call", "type": "tool_call",
            }])

    traces, state = run_traced(build(AlwaysInvalid()), "hi")

    assert state["answered_by"] == "respond"
    route_trace = next(t for t in traces if t["stage"] == "arty" and "step 1/5" in t["detail"])
    assert route_trace["detail"].startswith("answering myself")  # the existing prefix, unchanged
    assert "invalid" in route_trace["detail"]


def test_unknown_handoff_target_ends_the_turn_with_an_error():
    """A worker asking to hand off to a name that isn't in the registry is a bug in that worker, not
    a crash: the turn ends at the supervisor with an "error" trace line instead of routing anywhere."""

    async def node(state):
        return {"messages": [AIMessage("done for now")], "answered_by": "A", "handoff": "ghost"}

    worker = WorkerSpec(name="A", description="the only worker", make_node=lambda model, tools: node)

    class RouteToOnlyWorker(FakeChatModel):
        """Routes to "A", the only worker this graph's registry offers."""

        def _reply(self, messages):
            if self.tool_name == "RouteDecision":
                return AIMessage("", tool_calls=[{
                    "name": self.tool_name, "args": {"next": "A", "reason": "only option", "question": "hi"},
                    "id": "fake-call", "type": "tool_call",
                }])
            return super()._reply(messages)

    traces, state = run_traced(build(RouteToOnlyWorker(), workers=(worker,)), "hi")

    error_traces = [t for t in traces if t["status"] == "error"]
    assert len(error_traces) == 1 and "ghost" in error_traces[0]["detail"]
    assert state["messages"][-1].content == "done for now"  # the worker's own answer; no notice added


def test_routes_come_from_the_registry_not_from_a_fixed_list():
    """A worker that isn't one of the two built-ins, added only through `workers=`, must still be a
    valid route (the schema is built from the registry) and its description must appear in the
    supervisor's prompt (so the model actually knows when to pick it) — docs/contracts.md § 3."""

    calls: list[str] = []
    extra = stub_worker("extra_worker", calls)
    extra = WorkerSpec(name="extra_worker", description="EXTRA_MARKER: handles made-up test requests.", make_node=extra.make_node)

    class RouteToExtra(FakeChatModel):
        """Routes to the extra worker and records the system prompt it was given."""

        prompts: list[str] = Field(default_factory=list)

        def _reply(self, messages):
            if self.tool_name == "RouteDecision":
                self.prompts.append(str(messages[0].content))
                return AIMessage("", tool_calls=[{
                    "name": self.tool_name,
                    "args": {"next": "extra_worker", "reason": "picked the extra one", "question": "hi"},
                    "id": "fake-call", "type": "tool_call",
                }])
            return super()._reply(messages)

    model = RouteToExtra()
    state = run(build(model, workers=(*WORKERS, extra)), "hi")

    assert calls == ["extra_worker"]
    assert state["answered_by"] == "extra_worker"
    assert model.prompts and "EXTRA_MARKER" in model.prompts[0]
