"""
tests/test_content_ideator.py — unit tests for agents/content_ideator.py: the studio's idea
specialist, built on the model-driven tool loop (agents/tool_loop.py, docs/contracts.md § 9).

Phase 5 gives this worker no tools at all, so the main things to protect are: that it still behaves
like any tool_loop worker with nothing registered (one model call, no tool trace, untainted), that its
prompt actually carries the "exactly 3 ideas" rule and the turn's own question, and that — because this
file never names a tool itself — a tool the *registry* later grants it is offered to the model with no
change here. That last point is what lets Phase 6's save_ideas and Phase 8's load_skill arrive later
without editing this worker at all.

Every test uses the free FakeChatModel (never a real API call), and a tiny one-node graph (`run_node`)
so `get_stream_writer()` — which only works inside a running graph — has somewhere to write to, the
same pattern test_rag_agent.py and test_tool_loop.py use.
"""

import asyncio

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from pydantic import Field

from artlab.agents import content_ideator
from artlab.agents.state import ChatState
from artlab.model import FakeChatModel, fake_model
from artlab.tools.registry import ToolRegistry


def run_node(model, tools: ToolRegistry, task: str) -> tuple[dict, list[dict]]:
    """Run the content_ideator node once and return (final state values, trace lines it wrote).

    Builds the smallest graph that can give `get_stream_writer()` a context to write to: one node
    named "content_ideator". `stream_mode=["custom", "values"]` yields both the trace dicts the node
    writes and the merged state after it runs, in one pass.
    """

    async def _run():
        graph = (
            StateGraph(ChatState)
            .add_node("content_ideator", content_ideator.make_node(model, tools))
            .add_edge(START, "content_ideator")
            .compile(checkpointer=InMemorySaver())
        )
        config = {"configurable": {"thread_id": "test"}}
        traces: list[dict] = []
        values: dict = {}
        async for mode, chunk in graph.astream(
            {"messages": [HumanMessage(task)], "task": task}, config, stream_mode=["custom", "values"]
        ):
            if mode == "custom":
                traces.append(chunk)
            else:
                values = chunk
        return values, traces

    return asyncio.run(_run())


class SpyFakeModel(FakeChatModel):
    """The fake model, but it records the full message list it was asked to reply to, so a test can
    check exactly what the *prompt* looked like (system + human messages) rather than only what came
    back. `bind_tools` returns a *copy* of the model (model.py), so `calls` — kept on the instance a
    test holds onto — is a mutable list shared with that copy: appending inside `_reply` (which runs
    on the copy) is still visible here, the same trick test_tool_loop.py's AlwaysAsksForATool and
    test_rag_agent.py's SpyFakeModel rely on."""

    calls: list[list] = Field(default_factory=list)

    def _reply(self, messages):
        self.calls.append(list(messages))
        return super()._reply(messages)


def test_no_tools_is_one_model_call_answered_by_content_ideator_and_not_tainted():
    """docs/contracts.md § 9: with nothing registered for this agent yet (Phase 5's real state), a turn
    must still be exactly one model call — no tool trace lines — and the worker must report
    answered_by="content_ideator" and leave the chat untainted. Getting any of these wrong would either
    break the supervisor's "who answered" check or lock out data-changing tools for no reason."""
    values, traces = run_node(fake_model(["3 ideas here"]), ToolRegistry(), "desk setup ideas")

    assert values["answered_by"] == "content_ideator"
    assert values["messages"][-1].content == "3 ideas here"
    assert not values.get("tainted")  # never set to True: nothing untrusted was read
    assert not [t for t in traces if t["stage"] == "tool"]  # no tool ran
    model_traces = [t for t in traces if t["stage"] == "content_ideator"]
    assert len(model_traces) == 1 and model_traces[0]["detail"].endswith("answer")


def test_prompt_carries_the_three_ideas_rule_and_the_task():
    """This worker has no tool of its own to fall back on — its whole job is defined by PROMPT — so the
    exact rule the ticket cares about (EXACTLY 3 ideas) and the turn's own standalone question must
    actually reach the model. A spy model records the real message list run_tool_loop builds, so this
    checks the prompt's content directly instead of just trusting that PROMPT is wired in somewhere."""
    model = SpyFakeModel()
    task = "Give me ideas for my next desk-setup video"
    run_node(model, ToolRegistry(), task)

    assert len(model.calls) == 1
    system_msg, human_msg = model.calls[0]
    assert isinstance(system_msg, SystemMessage) and "EXACTLY 3" in system_msg.content
    assert isinstance(human_msg, HumanMessage) and human_msg.content == task


def test_a_tool_registered_for_content_ideator_is_offered_to_the_model():
    """docs/contracts.md § 9: content_ideator never names a tool — whatever the *registry* lists for
    it, run_tool_loop offers to the model. Registering one read-only tool here, with no change to
    content_ideator.py, must make the fake model call it: proof that Phase 6's save_ideas and Phase 8's
    load_skill will reach this worker later with zero code changes to this file."""
    tools = ToolRegistry()
    tools.register(
        "peek_trends", lambda query: f"trending: {query}", tier="read_only",
        allowed_agents={"content_ideator"}, description="test-only trend lookup",
    )

    values, traces = run_node(fake_model(), tools, "desk setup ideas")

    assert values["answered_by"] == "content_ideator"
    tool_traces = [t for t in traces if t["stage"] == "tool"]
    assert len(tool_traces) == 1
    assert tool_traces[0]["detail"].startswith("peek_trends") and tool_traces[0]["status"] == "ok"
