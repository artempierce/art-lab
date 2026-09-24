"""
tests/test_tool_loop.py — unit tests for agents/tool_loop.py, the model-driven tool loop every Phase 5
worker uses (docs/contracts.md § 9): the 3-call budget, the gateway's refusals and taint lock working
inside the loop (not just across turns), and the fake model's own tool-calling behaviour (model.py).

Like test_rag_agent.py, `get_stream_writer()` only works inside a running graph, so `run_loop` wraps
`run_tool_loop` in the smallest graph that can give it one — a single node — and reads back the trace
lines it wrote alongside the merged state. Every test uses the free FakeChatModel (or a small scripted
subclass of it) and a real ToolRegistry with tiny fake tools, so every test here is instant and $0.
"""

import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from pydantic import Field

from artlab.agents.state import ChatState
from artlab.agents.tool_loop import LoopResult, MAX_TOOL_CALLS, run_tool_loop
from artlab.model import FakeChatModel, cost_usd, fake_model
from artlab.tools.registry import ToolRegistry

SYSTEM_PROMPT = "You are a test worker."


def run_loop(model, tools: ToolRegistry, agent: str, task: str, tainted_in: bool = False) -> tuple[LoopResult, list[dict]]:
    """Run `run_tool_loop` once and return (its `LoopResult`, the trace lines it wrote).

    `get_stream_writer()` needs a graph context, so this builds the smallest graph that can give it
    one: a single node named `agent`. The `LoopResult` itself isn't part of `ChatState` — only the
    pieces a real worker returns are (`messages`, `spent_usd`, `answered_by`) — so the node stashes it
    in `results`, a list this closure can read back after the graph finishes.
    """
    results: list[LoopResult] = []

    async def node(state: ChatState) -> dict:
        r = await run_tool_loop(model, tools, agent, SYSTEM_PROMPT, state["task"], tainted_in=tainted_in)
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


def registry(**tools) -> ToolRegistry:
    """A real tool gateway with one entry per keyword: `name=(fn, tier)` (tier defaults to read_only),
    all allowed only for "test_agent" — the agent name every test in this file uses."""
    reg = ToolRegistry()
    for name, spec in tools.items():
        fn, tier = spec if isinstance(spec, tuple) else (spec, "read_only")
        reg.register(name, fn, tier=tier, allowed_agents={"test_agent"}, description=f"test tool {name!r}")
    return reg


class AlwaysAsksForATool(FakeChatModel):
    """Ignores the fake model's own default "one tool, then answer" restraint (model.py) and keeps
    asking for a tool for as long as it has any bound at all. Used to prove that MAX_TOOL_CALLS — the
    loop's own cap — is what stops the loop, not the model's cooperation: a model that never stops
    asking still can't run more than 3 tools, because the 4th model call is made with no tools bound.
    `calls` counts every time the model was invoked, bound copies included (see model.py's `bind_tools`:
    it returns a *copy*, so a counter appended to on the copy is still visible here — the same trick
    test_rag_agent.py's SpyFakeModel relies on).
    """

    calls: list[int] = Field(default_factory=list)

    def _reply(self, messages):
        self.calls.append(1)
        if self.tool_specs:
            name = self.tool_specs[0]["function"]["name"]
            return AIMessage("", tool_calls=[{"name": name, "args": {"query": "x"}, "id": f"call-{len(self.calls)}", "type": "tool_call"}])
        return AIMessage("done")


class FourToolsAtOnce(FakeChatModel):
    """A single reply that asks for four tool calls at once — one more than MAX_TOOL_CALLS — to prove
    the budget is enforced per *requested call*, in order, even within one reply. Once unbound (no
    tools left to ask for), it echoes back every ToolMessage it was given, so a test can check exactly
    what the 4th (budget-denied) call was told."""

    def _reply(self, messages):
        if self.tool_specs:
            name = self.tool_specs[0]["function"]["name"]
            calls = [{"name": name, "args": {"query": f"q{i}"}, "id": f"call-{i}", "type": "tool_call"} for i in range(4)]
            return AIMessage("", tool_calls=calls)
        return AIMessage(" | ".join(str(m.content) for m in messages if isinstance(m, ToolMessage)))


class AsksThenEchoesToolMessages(FakeChatModel):
    """Asks for exactly the tool calls in `first_calls` once, then — as soon as any ToolMessage is in
    the conversation — answers by echoing every ToolMessage it has seen so far, joined by " | ". Lets a
    test read back the exact text a refusal, a bad-arguments error, or a budget message put in front of
    the model, without `run_tool_loop` needing to expose its internal message list."""

    first_calls: list[dict] = Field(default_factory=list)

    def _reply(self, messages):
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        if not tool_msgs:
            return AIMessage("", tool_calls=list(self.first_calls))
        return AIMessage(" | ".join(str(m.content) for m in tool_msgs))


class SearchThenSaveModel(FakeChatModel):
    """First asks for the untrusted "search" tool; once that result is in the conversation, asks for
    the mutating "save" tool; once *that* is answered (refused or not), echoes it. Used to prove taint
    from earlier in the *same* loop — not just an earlier turn — is enough to refuse a mutating tool
    (docs/contracts.md § 1)."""

    def _reply(self, messages):
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        if len(tool_msgs) == 0:
            return AIMessage("", tool_calls=[{"name": "search", "args": {"query": "x"}, "id": "call-1", "type": "tool_call"}])
        if len(tool_msgs) == 1:
            return AIMessage("", tool_calls=[{"name": "save", "args": {"text": "x"}, "id": "call-2", "type": "tool_call"}])
        return AIMessage(str(tool_msgs[-1].content))


class TokenCountingModel(FakeChatModel):
    """Every reply carries `usage_metadata`, like real Claude's do, so a test can check that
    `run_tool_loop` sums the tokens (and so the cost) of *every* model call it makes, not just the
    last one — a worker that calls a tool always makes at least two."""

    def _reply(self, messages):
        usage = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
        if not any(isinstance(m, ToolMessage) for m in messages):
            return AIMessage("", tool_calls=[{"name": "echo", "args": {"query": "x"}, "id": "call-1", "type": "tool_call"}], usage_metadata=usage)
        return AIMessage("done", usage_metadata=usage)


def test_no_tools_is_one_model_call():
    """An agent with no registered tools still goes through run_tool_loop unchanged (docs/contracts.md
    § 9: "an agent with no tools is just one model call through the same loop") — it must not try to
    bind an empty tool list or otherwise behave differently from a normal answer."""
    result, traces = run_loop(fake_model(["hi there"]), registry(), "test_agent", "hello")

    assert result.reply.content == "hi there"
    assert result.tool_calls == 0
    model_traces = [t for t in traces if t["stage"] == "test_agent"]
    assert len(model_traces) == 1 and model_traces[0]["detail"].endswith("answer")


def test_default_fake_model_calls_one_tool_then_answers():
    """The fake model's own §9 behaviour (model.py): given a real, unforced tool list, it calls the
    first tool, then quotes its result. This also protects the gateway wrapping along the way: the
    tool's output is untrusted by default, so what the fake model quotes must already be wrapped in
    <untrusted_retrieval> — proving the wrapped text, not raw text, is what reaches the model."""
    result, traces = run_loop(fake_model(), registry(echo=lambda query: f"result for {query}"), "test_agent", "hello there")

    assert result.tool_calls == 1
    assert not result.reply.tool_calls  # a real, text-only answer
    assert "<untrusted_retrieval" in result.reply.content
    tool_traces = [t for t in traces if t["stage"] == "tool"]
    assert len(tool_traces) == 1 and tool_traces[0]["status"] == "ok"


def test_model_that_always_asks_for_a_tool_is_capped_at_three():
    """Even a model that never stops asking for tools can't run more than MAX_TOOL_CALLS: the loop's
    own cap terminates it, not the model's cooperation. Worst case is MAX_TOOL_CALLS + 1 model calls —
    the last one unbound, so it has no choice but to answer in text (docs/contracts.md § 9, step 4)."""
    model = AlwaysAsksForATool()
    result, traces = run_loop(model, registry(echo=lambda query: f"result for {query}"), "test_agent", "hello there")

    assert result.tool_calls == MAX_TOOL_CALLS == 3
    assert result.reply.content == "done"
    assert len(model.calls) == MAX_TOOL_CALLS + 1 == 4
    model_traces = [t for t in traces if t["stage"] == "test_agent"]
    assert len(model_traces) == 4


class AlwaysAsksForAForbiddenTool(AlwaysAsksForATool):
    """Like AlwaysAsksForATool, but always asks for a tool the gateway refuses (not on the allow-list):
    what a model does when an injection keeps telling it to call something it may not."""

    def _reply(self, messages):
        self.calls.append(1)
        if self.tool_specs:
            return AIMessage("", tool_calls=[{"name": "forbidden", "args": {}, "id": f"call-{len(self.calls)}", "type": "tool_call"}])
        return AIMessage("done")


def test_refused_requests_use_up_the_budget_so_the_loop_still_ends():
    """Regression guard: refused (and malformed) requests count towards MAX_TOOL_CALLS too. If only tools
    that actually ran counted, a model that keeps asking for a refused tool would loop forever, paying
    for a model call each round. Here nothing ever runs, and the loop still stops at MAX_TOOL_CALLS + 1."""
    model = AlwaysAsksForAForbiddenTool()
    result, traces = run_loop(model, registry(echo=lambda query: f"result for {query}"), "test_agent", "hello there")

    assert result.tool_calls == 0  # nothing ran…
    assert len(model.calls) == MAX_TOOL_CALLS + 1  # …and the loop still ended
    assert result.reply.content == "done"
    assert sum(t["status"] == "error" for t in traces if t["stage"] == "tool") == MAX_TOOL_CALLS


def test_four_tools_requested_at_once_only_three_run():
    """docs/contracts.md § 9: the budget check applies per requested call, in order, even when several
    arrive in the same reply — the 4th gets a fixed "budget used up" ToolMessage instead of running,
    and the tool itself is never called for it."""
    ran: list[str] = []
    result, traces = run_loop(FourToolsAtOnce(), registry(echo=lambda query: ran.append(query) or f"result for {query}"), "test_agent", "hello")

    assert result.tool_calls == 3
    assert ran == ["q0", "q1", "q2"]  # the 4th never actually ran
    assert "budget" in result.reply.content and "used up" in result.reply.content
    tool_traces = [t for t in traces if t["stage"] == "tool"]
    assert len(tool_traces) == 4
    assert tool_traces[3]["status"] == "stopped" and "budget" in tool_traces[3]["detail"]


def test_unknown_tool_is_refused_then_the_loop_still_answers():
    """A tool name the model asks for that isn't registered (or isn't on this agent's allow-list) must
    be refused by the gateway, not crash the loop — and a refusal never counts against the 3-call
    budget, since nothing actually ran."""
    model = AsksThenEchoesToolMessages(first_calls=[{"name": "delete_everything", "args": {}, "id": "call-1", "type": "tool_call"}])
    result, traces = run_loop(model, registry(), "test_agent", "hello")

    assert result.tool_calls == 0
    assert "refused" in result.reply.content and "unknown tool" in result.reply.content
    tool_traces = [t for t in traces if t["stage"] == "tool"]
    assert len(tool_traces) == 1 and tool_traces[0]["status"] == "error"


def test_bad_arguments_are_caught_without_running_the_tool():
    """docs/contracts.md § 9: arguments that don't fit the tool's real signature are caught before the
    tool ever runs — no timeout, no retry, and the loop must not crash."""
    ran: list[str] = []
    model = AsksThenEchoesToolMessages(first_calls=[{"name": "echo", "args": {"wrong_arg": "x"}, "id": "call-1", "type": "tool_call"}])
    result, traces = run_loop(model, registry(echo=lambda query: ran.append(query) or "ok"), "test_agent", "hello")

    assert result.tool_calls == 0
    assert ran == []  # the tool itself never ran
    assert "bad arguments" in result.reply.content
    tool_traces = [t for t in traces if t["stage"] == "tool"]
    assert tool_traces[0]["status"] == "error"


def test_a_mutating_tool_still_waits_for_approval_once_the_same_loop_read_untrusted_content():
    """Taint is sticky within one loop, not just across turns (docs/contracts.md § 1): a data-changing
    tool asked for right after an untrusted result, in the *same* run_tool_loop call, still needs to go
    through approval, carrying `tainted=True` on its card — otherwise a single crafted turn could read
    a poisoned tool result and immediately act on it, without ever reaching the chat-wide `tainted` flag
    or surfacing the risk to the owner.

    Updated for Phase 6 (docs/contracts.md § 10): before approvals existed, a tainted mutating request
    was refused outright ("read untrusted content"). Now it stops the loop for your Approve/Reject
    click instead, the same as an untainted mutating request — `pending["tainted"]` is what carries the
    warning through to the approval card."""
    tools = registry(search=lambda query: f"found {query}", save=(lambda text: "saved", "mutating"))
    result, traces = run_loop(SearchThenSaveModel(), tools, "test_agent", "hello", tainted_in=False)

    assert result.tainted is True
    assert result.tool_calls == 1  # only "search" actually ran; "save" stopped the loop for approval
    assert "waiting for your approval" in result.reply.content
    assert result.pending is not None
    assert result.pending["tool"] == "save" and result.pending["tainted"] is True
    approval_traces = [t for t in traces if t["stage"] == "tool" and t["status"] == "approval"]
    assert len(approval_traces) == 1 and "save" in approval_traces[0]["detail"]


def test_specs_for_hides_defaults_and_only_lists_allowed_tools():
    """docs/contracts.md § 9: a parameter with a default must never reach the model — our own code
    keeps control of limits like how many results come back — and specs_for must never leak another
    agent's tools into this one's list."""
    tools = ToolRegistry()
    tools.register("search_knowledge", lambda query, k=4: [], tier="read_only", allowed_agents={"rag_agent"}, description="Search.")
    tools.register("fetch_comments", lambda video_id: [], tier="read_only", allowed_agents={"youtube_researcher"}, description="Fetch.")

    specs = tools.specs_for("rag_agent")
    assert len(specs) == 1
    schema = specs[0]["function"]
    assert schema["name"] == "search_knowledge" and schema["description"] == "Search."
    assert schema["parameters"]["required"] == ["query"]
    assert "k" not in schema["parameters"]["properties"]

    assert [s["function"]["name"] for s in tools.specs_for("youtube_researcher")] == ["fetch_comments"]
    assert tools.specs_for("some_other_agent") == []


def test_cost_sums_every_model_call_in_the_loop():
    """spent_usd is the cost of every model call the loop makes, not just the final one (docs/contracts.md
    § 9, step 5) — a worker that calls a tool always makes at least two model calls in one turn."""
    result, traces = run_loop(TokenCountingModel(), registry(echo=lambda query: "ok"), "test_agent", "hello")

    assert result.tool_calls == 1
    expected = cost_usd(100, 20) * 2  # two model calls in this run, same usage each
    assert result.spent_usd == pytest.approx(expected)
