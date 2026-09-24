"""
tests/test_rag_agent.py — unit tests for agents/rag_agent.py: the rephrase-and-search-again cap,
and graceful handling of a failed search (docs/contracts.md § 6).

These call the `rag_agent` node directly, not through the full API-level graph (test_api.py already
covers the happy path end to end). `get_stream_writer()` only works inside a running graph, so each
test wraps the node in a tiny one-node graph (`run_node`) instead of patching it — closer to how the
node really runs, and it lets us also read back the trace lines it wrote.

Every test uses the free FakeChatModel (never a real API call), and a scripted `search_knowledge`
function instead of a real knowledge base, so these tests are instant and cost $0.
"""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from pydantic import Field

from artlab.agents import rag_agent
from artlab.agents.rag_agent import NOT_FOUND
from artlab.agents.state import ChatState
from artlab.model import FakeChatModel, fake_model
from artlab.rag.knowledge import Hit, SearchResult
from artlab.tools.registry import ToolRegistry, ToolResult


def registry(search_fn) -> ToolRegistry:
    """A real tool gateway with one search_knowledge tool wired to `search_fn(query) -> SearchResult`,
    the same way tools/catalog.py wires the real knowledge base."""
    tools = ToolRegistry()
    tools.register("search_knowledge", search_fn, tier="read_only", allowed_agents={"rag_agent"}, description="test search")
    return tools


def run_node(model, tools, task: str) -> tuple[dict, list[dict]]:
    """Run the rag_agent node once and return (final state values, trace lines it wrote).

    `get_stream_writer()` needs a graph context, so this builds the smallest graph that can give it
    one: a single "rag_agent" node. `stream_mode=["custom", "values"]` yields both the trace dicts the
    node writes and the merged state after it runs, in one pass — so a scripted search function is
    only ever called the number of times the node itself calls it.
    """

    async def _run():
        graph = (
            StateGraph(ChatState)
            .add_node("rag_agent", rag_agent.make_node(model, tools))
            .add_edge(START, "rag_agent")
            .compile(checkpointer=InMemorySaver())
        )
        config = {"configurable": {"thread_id": "test"}}
        traces: list[dict] = []
        values: dict = {}
        async for mode, chunk in graph.astream({"messages": [HumanMessage(task)], "task": task}, config, stream_mode=["custom", "values"]):
            if mode == "custom":
                traces.append(chunk)
            else:
                values = chunk
        return values, traces

    return asyncio.run(_run())


class SpyFakeModel(FakeChatModel):
    """The fake model, but it records every time it was asked to *answer* (as opposed to fill in
    structured output like RouteDecision or Rephrase). `with_structured_output` calls `bind_tools`,
    which returns a *copy* of the model (see model.py), so this counter — kept on the original
    instance a test holds onto — only grows when the real answering call in rag_agent's step 4 runs."""

    answer_calls: list[str] = Field(default_factory=list)

    def _reply(self, messages):
        if not self.tool_name:
            self.answer_calls.append(str(messages[-1].content))
        return super()._reply(messages)


class InvalidRephraseModel(FakeChatModel):
    """Like the fake model, but when asked to fill in a `Rephrase` it replies with plain text and no
    tool call — simulating structured output that fails to parse (`parsed` comes back `None`, the
    same failure mode `supervisor.py` handles for `RouteDecision`). Used to check rag_agent skips the
    second search rather than retrying with garbage or crashing."""

    def _reply(self, messages):
        if self.tool_name == "Rephrase":
            return AIMessage("that's not a tool call")
        return super()._reply(messages)


def test_hits_on_first_search_needs_no_rephrase():
    """If the first search already finds something, rag_agent must not spend a second search or a
    rephrase call: the 2-search cap in docs/contracts.md § 6 is a ceiling, not a target."""
    calls: list[str] = []

    def search_fn(query: str) -> SearchResult:
        calls.append(query)
        return SearchResult(query, [Hit(1, "knowledge/policy.md", "Rule", "Disclose sponsorships.", 0.9)], 0)

    values, traces = run_node(fake_model(), registry(search_fn), "What is the sponsor rule?")

    assert calls == ["What is the sponsor rule?"]  # exactly 1 search
    tool_traces = [t for t in traces if t["stage"] == "tool"]
    assert len(tool_traces) == 1 and tool_traces[0]["detail"].endswith("search 1/2")
    rag_traces = [t for t in traces if t["stage"] == "rag_agent"]
    assert len(rag_traces) == 1  # no separate rephrase trace line → 0 rephrase calls
    assert "[1]" in values["messages"][-1].content


def test_zero_hits_then_rephrase_finds_something():
    """Zero hits on the first search → rag_agent asks the model to reword the question and searches
    again, once, with the *new* query. Protects the retry path itself: the second search must use the
    rephrased query, not repeat the original one."""
    task = "What is the sponsor rule?"
    rephrased = f"{task} policy"  # FakeChatModel's deterministic Rephrase (model.py)
    calls: list[str] = []

    def search_fn(query: str) -> SearchResult:
        calls.append(query)
        if query == task:
            return SearchResult(query, [], 0)
        return SearchResult(query, [Hit(1, "knowledge/policy.md", "Rule", "Disclose sponsorships.", 0.9)], 0)

    values, traces = run_node(fake_model(), registry(search_fn), task)

    assert calls == [task, rephrased]  # exactly 2 searches, second with the rephrased query
    tool_traces = [t for t in traces if t["stage"] == "tool"]
    assert tool_traces[0]["detail"].endswith("search 1/2")
    assert tool_traces[1]["detail"].endswith("search 2/2")
    assert "[1]" in values["messages"][-1].content


def test_zero_hits_twice_gives_up_without_an_answer_call():
    """Two empty searches → NOT_FOUND. That's code's decision, not the model's: the answering model
    is never asked, so nothing from outside is ever shown to it, and the chat isn't tainted."""
    calls: list[str] = []

    def search_fn(query: str) -> SearchResult:
        calls.append(query)
        return SearchResult(query, [], 0)

    model = SpyFakeModel()
    values, traces = run_node(model, registry(search_fn), "What is the sponsor rule?")

    assert len(calls) == 2  # first search, then the rephrased retry — both empty
    assert values["messages"][-1].content == NOT_FOUND
    assert model.answer_calls == []  # the answering model was never called
    assert values.get("tainted", False) is False
    rag_traces = [t for t in traces if t["stage"] == "rag_agent"]
    assert "no answer call" in rag_traces[-1]["detail"]


def test_invalid_rephrase_output_skips_the_second_search():
    """If the model's Rephrase doesn't parse, rag_agent must not retry with garbage or crash — it
    gives up after just the first search."""
    calls: list[str] = []

    def search_fn(query: str) -> SearchResult:
        calls.append(query)
        return SearchResult(query, [], 0)

    values, traces = run_node(InvalidRephraseModel(), registry(search_fn), "What is the sponsor rule?")

    assert calls == ["What is the sponsor rule?"]  # no second search
    assert values["messages"][-1].content == NOT_FOUND
    rag_traces = [t for t in traces if t["stage"] == "rag_agent"]
    assert any("invalid output" in t["detail"] for t in rag_traces)


def test_failed_search_is_answered_without_a_model_call():
    """A search that fails outright (e.g. the tool gateway's retry already used up — T2) must not
    crash the turn: a fixed answer, no model call, and the tool trace shows the error. Uses a stub
    registry rather than the real gateway, since T2's ok=False path isn't on this branch yet."""

    class FailingRegistry:
        async def call(self, agent, name, *, tainted=False, **args):
            return ToolResult(tool=name, ok=False, data=None, text="", untrusted=True, error="timeout after 10s", attempts=2)

    model = SpyFakeModel()
    values, traces = run_node(model, FailingRegistry(), "What is the sponsor rule?")

    assert model.answer_calls == []  # no model call
    assert values["messages"][-1].content == "I couldn't search the knowledge base just now (timeout after 10s)."
    assert values["answered_by"] == "rag_agent"
    assert values.get("tainted", False) is False
    tool_traces = [t for t in traces if t["stage"] == "tool"]
    assert tool_traces[0]["status"] == "error" and "timeout after 10s" in tool_traces[0]["detail"]
