"""
tests/test_agent_tools.py — H2 tests: "agent calls agent" (Phase 9b, docs/contracts.md § 13).
content_ideator can hand youtube_researcher a question through a tool (`ask_youtube_researcher`,
tools/agents.py) and read back its answer, capped at one level deep and sharing the turn's own cost
budget with the nested call.

What's checked here, one file per concern the same way test_tool_loop.py and test_youtube_researcher.py
split theirs: the happy path end to end (W10), the depth cap that stops an agent calling an agent from
inside another agent's own call, the shared cost budget, the registry's new async-tool support
(tools/registry.py), and the allow-list refusing this tool for every agent but content_ideator.

Every test uses the free FakeChatModel (or a small scripted subclass of it) and a real ToolRegistry
with tiny or real stub tools — no real model call, no real YouTube API, $0 (CLAUDE.md's cost rule).
"""

import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from pydantic import Field

from artlab.agents import youtube_researcher
from artlab.agents.state import ChatState
from artlab.agents.tool_loop import LoopResult, run_tool_loop
from artlab.model import FakeChatModel, cost_usd, fake_model
from artlab.tools.agents import make_agent_tool
from artlab.tools.registry import ToolDenied, ToolRegistry
from artlab.tools.stubs import query_youtube_trends

SYSTEM_PROMPT = "You are a test worker."


def run_loop(
    model, tools: ToolRegistry, agent: str, task: str, tainted_in: bool = False, depth: int = 0
) -> tuple[LoopResult, list[dict]]:
    """Run `run_tool_loop` once and return (its `LoopResult`, the trace lines it wrote) — the same
    single-node-graph trick test_tool_loop.py's own `run_loop` uses, since `get_stream_writer()` only
    works inside a running graph."""
    results: list[LoopResult] = []

    async def node(state: ChatState) -> dict:
        r = await run_tool_loop(model, tools, agent, SYSTEM_PROMPT, state["task"], tainted_in=tainted_in, depth=depth)
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


def agent_tool_registry(model: FakeChatModel) -> ToolRegistry:
    """A real tool gateway with content_ideator's `ask_youtube_researcher` (tools/agents.py) wired to
    a real youtube_researcher, who in turn gets the real `query_youtube_trends` stub — the same shape
    tools/catalog.py wires for the real app, just without the tools content_ideator/youtube_researcher
    don't need for these tests (save_ideas, fetch_comments, load_skill).

    `model` is shared by both agents' turns — there's only one model in the real app too (model.py) —
    so a test can script it to play both roles at once (see AgentCallThenAnswer, below).
    """
    tools = ToolRegistry()
    tools.register(
        "query_youtube_trends", query_youtube_trends, tier="read_only",
        allowed_agents={"youtube_researcher"}, description="Look up trending videos in a niche.",
    )
    tools.register(
        "ask_youtube_researcher",
        make_agent_tool(model, tools, "youtube_researcher", youtube_researcher.PROMPT),
        tier="read_only", allowed_agents={"content_ideator"},
        description="Ask youtube_researcher a question and get its research answer back.",
        untrusted_output=True, timeout_s=60.0,
    )
    return tools


def test_content_ideator_asks_youtube_researcher_and_the_nested_call_taints_the_chat():
    """W10: "ideas based on trending desk setups" makes the default fake model call
    ask_youtube_researcher, which runs a real nested turn for youtube_researcher — whose own default
    fake-model behaviour (model.py) calls query_youtube_trends, then answers. The nested tool call's
    trace line must carry the "↳ for content_ideator" prefix (docs/contracts.md § 13), and because the
    researcher's answer is untrusted output (the YouTube stub's data, wrapped again as
    "agent:youtube_researcher"), the whole chat ends up tainted even though content_ideator itself
    never read anything untrusted directly."""
    model = fake_model()
    tools = agent_tool_registry(model)

    result, traces = run_loop(model, tools, "content_ideator", "ideas based on trending desk setups")

    # content_ideator called ask_youtube_researcher.
    ask_trace = next(t for t in traces if t["stage"] == "tool" and t["detail"].startswith("ask_youtube_researcher"))
    assert ask_trace["status"] == "ok"

    # The nested call, one level down, ran query_youtube_trends — and every line it wrote (both its
    # own model-call line and the tool line) says who it's nested for.
    nested_tool = next(t for t in traces if t["stage"] == "tool" and "query_youtube_trends" in t["detail"])
    assert nested_tool["detail"].startswith("↳ for content_ideator ·")
    nested_model_lines = [t for t in traces if t["stage"] == "youtube_researcher"]
    assert nested_model_lines and all(t["detail"].startswith("↳ for content_ideator ·") for t in nested_model_lines)

    # content_ideator's own answer isn't nested, so its lines carry no prefix at all.
    ideator_lines = [t for t in traces if t["stage"] == "content_ideator"]
    assert ideator_lines and all("↳ for" not in t["detail"] for t in ideator_lines)

    assert not result.reply.tool_calls  # content_ideator finished with a real text answer
    assert result.reply.content
    assert result.tainted is True  # the researcher's answer is untrusted output


def test_a_nested_loop_is_never_offered_agent_tools_and_refuses_one_if_asked_anyway():
    """Depth cap (docs/contracts.md § 13): at depth 1, `ask_*` tools are dropped from what's bound to
    the model at all (checked with a spy on every `bind_tools` call — this agent also has an ordinary
    tool, so the spec list isn't simply empty) — and even a model that asks for one anyway, ignoring
    its own tool list entirely (an injected comment could do this), gets refused with the exact
    depth-limit message instead of recursing into a second nested loop. The tool itself never runs
    either way."""
    calls: list[int] = []

    def fake_ask_something() -> str:
        calls.append(1)
        return "should never run"

    class AsksForAgentToolAnyway(FakeChatModel):
        """Scripted: at depth 1, ignores that ask_youtube_researcher was never offered and asks for it
        by name regardless, instead of the one ordinary tool it was actually bound to."""

        seen_specs: list[list] = Field(default_factory=list)

        def bind_tools(self, tools: list, **kwargs):
            self.seen_specs.append(tools)
            return super().bind_tools(tools, **kwargs)

        def _reply(self, messages):
            tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
            if not tool_msgs:
                return AIMessage("", tool_calls=[{"name": "ask_youtube_researcher", "args": {"question": "x"}, "id": "call-1", "type": "tool_call"}])
            return AIMessage(" | ".join(str(m.content) for m in tool_msgs))  # echoes the refusal back

    model = AsksForAgentToolAnyway()
    tools = ToolRegistry()
    tools.register(
        "peek", lambda: "peek result", tier="read_only",
        allowed_agents={"youtube_researcher"}, description="an ordinary tool, so the nested spec list isn't just empty",
    )
    tools.register(
        "ask_youtube_researcher", fake_ask_something, tier="read_only",
        allowed_agents={"youtube_researcher"}, description="test-only stand-in for the real agent tool",
    )

    result, traces = run_loop(model, tools, "youtube_researcher", "anything", depth=1)

    # Never offered: every bind_tools call this loop made left ask_youtube_researcher out, even though
    # it's on this agent's own allow-list — "peek" is still there, so this isn't just an empty list.
    assert model.seen_specs  # bind_tools was actually called at least once
    for specs in model.seen_specs:
        names = [spec["function"]["name"] for spec in specs]
        assert "peek" in names
        assert "ask_youtube_researcher" not in names

    # Asked for anyway: refused, with the depth-limit reason, not run.
    assert calls == []
    assert result.tool_calls == 0
    assert "refused" in result.reply.content
    assert "depth limit 1" in result.reply.content
    refusal_trace = next(t for t in traces if t["stage"] == "tool" and t["status"] == "error")
    assert "depth limit 1" in refusal_trace["detail"]


def test_nested_model_calls_cost_is_folded_into_the_callers_spent_usd():
    """Shared budget (docs/contracts.md § 13): `LoopResult.spent_usd` for content_ideator's own turn
    must include every model call the nested youtube_researcher turn made too, not just
    content_ideator's own two calls (ask, then answer) — four model calls in total here, each with the
    same fake `usage_metadata`, so the expected cost is exactly 4x one call's cost."""

    class AgentCallThenAnswer(FakeChatModel):
        """Plays both roles the shared model has to in this turn: content_ideator (asks
        ask_youtube_researcher, then answers) and, nested one level down, youtube_researcher (asks
        query_youtube_trends, then answers) — told apart by which tool names it was actually bound to,
        the same way the real fake model tells worker roles apart (model.py)."""

        def _reply(self, messages):
            usage = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
            has_result = any(isinstance(m, ToolMessage) for m in messages)
            names = [s["function"]["name"] for s in self.tool_specs] if self.tool_specs else []
            if not has_result and "ask_youtube_researcher" in names:
                return AIMessage("", tool_calls=[{"name": "ask_youtube_researcher", "args": {"question": "trending?"}, "id": "call-1", "type": "tool_call"}], usage_metadata=usage)
            if not has_result and "query_youtube_trends" in names:
                return AIMessage("", tool_calls=[{"name": "query_youtube_trends", "args": {"niche": "desk setups"}, "id": "call-2", "type": "tool_call"}], usage_metadata=usage)
            return AIMessage("done", usage_metadata=usage)

    model = AgentCallThenAnswer()
    tools = agent_tool_registry(model)

    result, _ = run_loop(model, tools, "content_ideator", "what's trending in desk setups?")

    assert result.tool_calls == 1  # only ask_youtube_researcher, from content_ideator's own count
    expected = cost_usd(100, 20) * 4  # content_ideator's 2 calls + youtube_researcher's own 2, nested
    assert result.spent_usd == pytest.approx(expected)


def test_an_async_tool_runs_times_out_and_retries_like_a_sync_one():
    """docs/contracts.md § 13: the gateway now awaits a coroutine-function tool directly
    (`inspect.iscoroutinefunction`, tools/registry.py) instead of running it in a worker thread — the
    path an agent tool like ask_youtube_researcher needs, since it has to `await` a nested model call.
    But it must still get exactly the sync path's own timeout and one retry: a tiny timeout against an
    async tool that sleeps too long proves it actually ran (the counter), was retried once, and came
    back as a failed ToolResult instead of hanging."""
    calls: list[int] = []

    async def slow_async() -> str:
        calls.append(1)
        await asyncio.sleep(0.5)
        return "too slow"

    tools = ToolRegistry()
    tools.register("slow_async", slow_async, tier="read_only", allowed_agents={"rag_agent"}, description="", timeout_s=0.05)

    result = asyncio.run(tools.call("rag_agent", "slow_async"))

    assert result.ok is False
    assert "timeout" in result.error
    assert result.attempts == 2
    assert len(calls) == 2  # really awaited twice: retried once, not zero or more than twice


def test_ask_youtube_researcher_is_refused_for_any_agent_other_than_content_ideator():
    """docs/contracts.md § 13: ask_youtube_researcher is registered content_ideator-only
    (tools/catalog.py) — the ordinary allow-list check every tool goes through (tools/registry.py § 4),
    so any other agent asking for it (including youtube_researcher itself) is refused before anything
    runs, the same as an agent asking for a tool that was never meant for it."""
    tools = ToolRegistry()
    tools.register(
        "ask_youtube_researcher",
        make_agent_tool(fake_model(), tools, "youtube_researcher", youtube_researcher.PROMPT),
        tier="read_only", allowed_agents={"content_ideator"},
        description="Ask youtube_researcher a question and get its research answer back.",
    )

    with pytest.raises(ToolDenied, match="youtube_researcher may not use ask_youtube_researcher"):
        asyncio.run(tools.call("youtube_researcher", "ask_youtube_researcher", question="what's trending?"))

    with pytest.raises(ToolDenied, match="rag_agent may not use ask_youtube_researcher"):
        asyncio.run(tools.call("rag_agent", "ask_youtube_researcher", question="what's trending?"))
