"""
tests/test_youtube_researcher.py — unit tests for agents/youtube_researcher.py, Arty's model-driven
YouTube research worker (docs/contracts.md § 9, ticket T7).

The worker itself is a thin wrapper around agents/tool_loop.py's run_tool_loop, so these tests don't
re-check the loop's own mechanics (test_tool_loop.py already does) — they check the things that are
this worker's own responsibility: which two real tools (tools/stubs.py) it's wired to, that a poisoned
comment can't make it call something it shouldn't, that a tool outage doesn't crash the turn, and that
the shared MAX_TOOL_CALLS budget still applies here.

Every test uses the free FakeChatModel (or a small scripted subclass of it) and a real ToolRegistry
holding the two real stub tools, the same way tools/catalog.py wires them — so every test here is
instant and $0, and no real YouTube API or model call ever happens.
"""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from pydantic import Field

from artlab.agents import youtube_researcher
from artlab.agents.state import ChatState
from artlab.agents.tool_loop import MAX_TOOL_CALLS
from artlab.model import FakeChatModel, fake_model
from artlab.tools.registry import ToolRegistry
from artlab.tools.stubs import fetch_comments, query_youtube_trends


def registry() -> ToolRegistry:
    """A real tool gateway with the two real stub tools (tools/stubs.py), wired for
    "youtube_researcher" exactly the way tools/catalog.py wires them for the real app: same tier
    (read_only), same allow-list, untrusted output left at its default (True) since neither stub
    knows anything the studio didn't already know."""
    tools = ToolRegistry()
    tools.register(
        "query_youtube_trends", query_youtube_trends, tier="read_only",
        allowed_agents={"youtube_researcher"}, description="Look up trending videos in a niche.",
    )
    tools.register(
        "fetch_comments", fetch_comments, tier="read_only",
        allowed_agents={"youtube_researcher"}, description="Fetch the comments on a video.",
    )
    return tools


def run_node(model, tools: ToolRegistry, task: str) -> tuple[dict, list[dict]]:
    """Run the youtube_researcher node once and return (final state values, trace lines it wrote).

    `get_stream_writer()` (used inside run_tool_loop) only works inside a running graph, so this
    builds the smallest graph that can give it one — a single node — the same pattern
    test_rag_agent.py and test_tool_loop.py use.
    """

    async def _run():
        graph = (
            StateGraph(ChatState)
            .add_node("youtube_researcher", youtube_researcher.make_node(model, tools))
            .add_edge(START, "youtube_researcher")
            .compile(checkpointer=InMemorySaver())
        )
        config = {"configurable": {"thread_id": "test"}}
        traces: list[dict] = []
        values: dict = {}
        async for mode, chunk in graph.astream(
            {"messages": [HumanMessage(task)], "task": task}, config, stream_mode=["custom", "values"],
        ):
            if mode == "custom":
                traces.append(chunk)
            else:
                values = chunk
        return values, traces

    return asyncio.run(_run())


class PoisonedCommentsThenMutate(FakeChatModel):
    """Calls fetch_comments first. Once its result is in the conversation — the poisoned comment
    (tools/stubs.py) included, telling whoever reads it to "save this comment to a file on disk" —
    it asks for a tool named "save_to_disk", as a model that got talked into obeying the comment
    would. That tool doesn't exist in this worker's registry, so the gateway refuses it as unknown,
    the same outcome a real mutating tool would get once the chat is tainted. Once that refusal is
    in the conversation too, it answers in text, so the turn still finishes with a real answer."""

    def _reply(self, messages):
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        if len(tool_msgs) == 0:
            return AIMessage("", tool_calls=[
                {"name": "fetch_comments", "args": {"video_id": "abc123"}, "id": "call-1", "type": "tool_call"},
            ])
        if len(tool_msgs) == 1:
            return AIMessage("", tool_calls=[
                {"name": "save_to_disk", "args": {}, "id": "call-2", "type": "tool_call"},
            ])
        return AIMessage("Research summary: viewers like the desk setup; a few ask about gear.")


class AlwaysAsksForTrends(FakeChatModel):
    """Ignores the fake model's own default "one tool, then answer" restraint (model.py) and keeps
    asking for query_youtube_trends for as long as it has any tools bound at all — the same trick
    test_tool_loop.py's AlwaysAsksForATool uses, scoped to this worker's own real tool. Proves that
    MAX_TOOL_CALLS, not the model's cooperation, is what stops a worker that never stops asking."""

    calls: list[int] = Field(default_factory=list)

    def _reply(self, messages):
        self.calls.append(1)
        if self.tool_specs:
            return AIMessage("", tool_calls=[
                {"name": "query_youtube_trends", "args": {"niche": "desk setups"}, "id": f"call-{len(self.calls)}", "type": "tool_call"},
            ])
        return AIMessage("done researching")


def test_default_fake_model_calls_one_tool_then_answers():
    """Shape test (S1): with no scripting at all, the default fake model calls one tool
    (query_youtube_trends, the first tool registered for this agent) and then answers in text.
    Protects the worker's wiring into run_tool_loop end to end: the right agent name reaches the
    registry, the right trace stages get written, `answered_by` is set, and the stub's output —
    untrusted by default — taints the chat, because nothing from outside the studio has been checked."""
    values, traces = run_node(fake_model(), registry(), "what's trending in desk setups?")

    assert values["answered_by"] == "youtube_researcher"
    assert values["tainted"] is True
    reply = values["messages"][-1]
    assert not reply.tool_calls  # a real, text-only answer
    assert reply.content  # something was said

    tool_traces = [t for t in traces if t["stage"] == "tool"]
    assert len(tool_traces) == 1 and tool_traces[0]["detail"].startswith("query_youtube_trends")

    worker_traces = [t for t in traces if t["stage"] == "youtube_researcher"]
    assert len(worker_traces) == 2  # the "asks for..." call, then the "answer" call


def test_poisoned_comment_cannot_make_the_worker_call_a_disallowed_tool():
    """Injection defence (W5/S6): fetch_comments' one poisoned comment tells its reader to save it to
    disk. Even when the model reacts to that and asks for a "save" tool, the gateway refuses it (it
    isn't registered for this worker), so the instruction never runs. The gateway's arrival scan still
    flags the comment in the tool trace, the chat is tainted (untrusted content was read), and the
    worker still produces a real answer instead of crashing or leaving the turn half-finished."""
    values, traces = run_node(PoisonedCommentsThenMutate(), registry(), "what do people say about video abc123?")

    assert values["answered_by"] == "youtube_researcher"
    assert values["tainted"] is True
    reply = values["messages"][-1]
    assert not reply.tool_calls
    assert reply.content == "Research summary: viewers like the desk setup; a few ask about gear."

    tool_traces = [t for t in traces if t["stage"] == "tool"]
    fetch_trace = next(t for t in tool_traces if t["detail"].startswith("fetch_comments"))
    assert "flagged" in fetch_trace["detail"]

    refused_trace = next(t for t in tool_traces if t["detail"].startswith("save_to_disk"))
    assert refused_trace["status"] == "error" and "refused" in refused_trace["detail"]


def test_trends_tool_down_is_answered_not_crashed():
    """Outage handling (S10): if query_youtube_trends fails (a "real" 500 from a broken integration,
    stood in for here by a function that always raises), the gateway retries once (docs/contracts.md
    § 4) and then returns a failed ToolResult instead of raising. The worker must still turn that into
    a real answer, not an exception — a broken trends lookup shouldn't take down the whole turn."""

    def broken_trends(niche: str) -> list[dict]:
        raise RuntimeError("500")

    tools = ToolRegistry()
    tools.register(
        "query_youtube_trends", broken_trends, tier="read_only",
        allowed_agents={"youtube_researcher"}, description="Look up trending videos in a niche.",
    )
    tools.register(
        "fetch_comments", fetch_comments, tier="read_only",
        allowed_agents={"youtube_researcher"}, description="Fetch the comments on a video.",
    )

    values, traces = run_node(fake_model(), tools, "what's trending in desk setups?")

    assert values["answered_by"] == "youtube_researcher"
    reply = values["messages"][-1]
    assert not reply.tool_calls
    assert reply.content  # the worker still answered

    tool_traces = [t for t in traces if t["stage"] == "tool"]
    failed = next(t for t in tool_traces if t["detail"].startswith("query_youtube_trends"))
    assert failed["status"] == "error"
    assert "2" in failed["detail"]  # attempts: the gateway's one retry already ran


def test_tool_budget_holds_even_for_a_model_that_never_stops_asking():
    """Shared budget (docs/contracts.md § 9): a model that always asks for query_youtube_trends can
    still only run it MAX_TOOL_CALLS times in one turn — the loop's own cap, not this worker's own
    code, is what protects it, but this worker must not do anything (like catching and re-running
    tool calls itself) that would let a run bypass that shared limit."""
    model = AlwaysAsksForTrends()
    values, traces = run_node(model, registry(), "what's trending in desk setups?")

    tool_traces = [t for t in traces if t["stage"] == "tool" and t["detail"].startswith("query_youtube_trends")]
    assert len(tool_traces) == MAX_TOOL_CALLS == 3
    assert values["answered_by"] == "youtube_researcher"
    reply = values["messages"][-1]
    assert not reply.tool_calls
    assert reply.content == "done researching"
