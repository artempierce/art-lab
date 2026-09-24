"""
tests/test_summarize.py — unit tests for agents/summarize.py: the no-op below the trigger, the fold
above it (order, removed ids, cost, trace line), a later fold swallowing an earlier summary, and that
the node never touches the `tainted` flag (docs/contracts.md § 11).

Like test_tool_loop.py and test_rag_agent.py, `get_stream_writer()` only works inside a running graph,
so `run_summarize_turns` wraps the node in the smallest graph that can give it one — a single node —
and reads back both the trace lines it wrote and the *merged* state, so message order is checked
through the real `add_messages` reducer, not by re-implementing it in the test. Every test uses the
free FakeChatModel (or a small scripted subclass of it), so all of these are instant and $0.
"""

import asyncio

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from pydantic import Field

from artlab.agents import summarize
from artlab.agents.state import ChatState
from artlab.agents.summarize import KEEP_RECENT, SUMMARY_TRIGGER
from artlab.model import FakeChatModel, cost_usd


def build_messages(start: int, n: int) -> list[BaseMessage]:
    """`n` alternating Human/AI messages, ids and content numbered from `start` — e.g.
    build_messages(0, 3) gives ids "m0", "m1", "m2" with content "message 0", "message 1", "message 2".
    Two turns in the same test pass different `start` values so their ids never collide."""
    msgs = []
    for i in range(start, start + n):
        cls = HumanMessage if i % 2 == 0 else AIMessage
        msgs.append(cls(f"message {i}", id=f"m{i}"))
    return msgs


def run_summarize_turns(model, turns: list[list[BaseMessage]]) -> list[tuple[dict, dict, list[dict]]]:
    """Run the `summarize` node once per entry in `turns`, all on the *same* thread, so state
    accumulates across turns the way a real chat does (needed to test that a later fold swallows an
    earlier summary). Returns one `(raw update dict, merged state after that turn, trace lines written
    during that turn)` per turn.

    The raw update dict (what the node itself returned, before the reducer merges it) is what lets
    `test_tainted_flag_is_not_written` check a key is *absent*, not just unchanged — the merged state
    alone can't tell "wrote False" apart from "wrote nothing" for an `operator.or_` field.
    """
    node_fn = summarize.make_node(model)
    updates: list[dict] = []

    async def node(state: ChatState) -> dict:
        update = await node_fn(state)
        updates.append(update)
        return update

    async def _run() -> list[tuple[dict, dict, list[dict]]]:
        graph = (
            StateGraph(ChatState)
            .add_node("summarize", node)
            .add_edge(START, "summarize")
            .compile(checkpointer=InMemorySaver())
        )
        config = {"configurable": {"thread_id": "test"}}
        results = []
        for messages in turns:
            traces: list[dict] = []
            values: dict = {}
            async for mode, chunk in graph.astream({"messages": messages}, config, stream_mode=["custom", "values"]):
                if mode == "custom":
                    traces.append(chunk)
                else:
                    values = chunk
            results.append((updates[-1], values, traces))
        return results

    return asyncio.run(_run())


def run_summarize(model, messages: list[BaseMessage]) -> tuple[dict, dict, list[dict]]:
    """Run the summarize node once, on a fresh thread. See `run_summarize_turns`."""
    return run_summarize_turns(model, [messages])[0]


class CountingModel(FakeChatModel):
    """The fake model, but every answer carries fixed `usage_metadata` (like real Claude's replies do)
    and every call is recorded — so a test can check summarize makes exactly one model call per fold,
    and that the cost it reports matches `cost_usd()` for that usage."""

    calls: list[int] = Field(default_factory=list)

    def _reply(self, messages):
        self.calls.append(1)
        return AIMessage("This is the summary.", usage_metadata={"input_tokens": 400, "output_tokens": 40, "total_tokens": 440})


def test_short_chat_is_untouched():
    """docs/contracts.md § 11: a chat at or under SUMMARY_TRIGGER messages must pass straight through —
    no model call, no trace line, and every message exactly as it was (same ids, same order)."""
    model = CountingModel()
    messages = build_messages(0, SUMMARY_TRIGGER)  # exactly at the trigger: "<=" means still a no-op

    update, values, traces = run_summarize(model, messages)

    assert update == {}
    assert model.calls == []
    assert traces == []
    assert [m.id for m in values["messages"]] == [f"m{i}" for i in range(SUMMARY_TRIGGER)]


def test_long_chat_folds_old_messages_into_one_summary():
    """docs/contracts.md § 11: once a chat passes SUMMARY_TRIGGER messages, summarize must fold
    everything but the last KEEP_RECENT into one SystemMessage summary placed BEFORE those kept
    messages, remove every old message's id, make exactly one model call, and add its cost — plus a
    trace line reporting how many messages were folded."""
    model = CountingModel()
    n = SUMMARY_TRIGGER + 1  # one over the trigger
    messages = build_messages(0, n)

    update, values, traces = run_summarize(model, messages)

    assert len(model.calls) == 1

    final = values["messages"]
    assert len(final) == 1 + KEEP_RECENT  # one summary + the kept recent messages
    summary, *kept = final
    assert isinstance(summary, SystemMessage)
    assert summary.content == "Summary of the earlier conversation: This is the summary."

    # The kept messages are the real last KEEP_RECENT, in their original order — checked by content,
    # since summarize must reissue their ids to place the summary before them (see summarize.py's
    # module docstring for why).
    assert [m.content for m in kept] == [f"message {i}" for i in range(n - KEEP_RECENT, n)]

    # Every old message's id is gone from the merged state.
    old_ids = {f"m{i}" for i in range(n - KEEP_RECENT)}
    assert old_ids.isdisjoint({m.id for m in final})

    expected_cost = cost_usd(400, 40)
    assert update["spent_usd"] == pytest.approx(expected_cost)
    assert values["spent_usd"] == pytest.approx(expected_cost)

    assert len(traces) == 1
    trace = traces[0]
    assert trace["stage"] == "memory"
    assert "summarized 21 messages" in trace["detail"]  # n - KEEP_RECENT old messages folded
    assert "440 tokens" in trace["detail"]


def test_second_summarize_folds_the_first_summary_in():
    """docs/contracts.md § 11: "a previous summary SystemMessage counts as an old message" — once a
    chat that already has a summary grows past SUMMARY_TRIGGER again, the next fold must swallow that
    earlier summary into the new one too, rather than special-casing it or ever leaving two summaries
    in the chat."""
    model = CountingModel()
    first_turn = build_messages(0, SUMMARY_TRIGGER + 1)  # 31 messages -> first fold: summary + 10 kept
    second_turn = build_messages(31, 20)  # 11 + 20 = 31 again -> second fold

    (_, first_values, _), (_, second_values, second_traces) = run_summarize_turns(model, [first_turn, second_turn])

    first_summary_id = first_values["messages"][0].id
    final = second_values["messages"]

    assert len(model.calls) == 2  # one fold per triggering turn
    assert len(final) == 1 + KEEP_RECENT
    summary, *kept = final
    assert isinstance(summary, SystemMessage)
    assert first_summary_id not in {m.id for m in final}  # the old summary was folded in, not kept
    assert [m.content for m in kept] == [f"message {i}" for i in range(41, 51)]  # the newest 10
    assert len(second_traces) == 1 and second_traces[0]["stage"] == "memory"


def test_tainted_flag_is_not_written():
    """docs/contracts.md § 11: "the chat's tainted flag is unchanged" — summarize doesn't decide taint
    at all, it just never writes `tainted` or `taint_sources`, so `ChatState`'s `operator.or_` reducer
    (agents/state.py) has nothing new to combine and whatever the flag already was survives untouched."""
    model = CountingModel()
    update, _, _ = run_summarize(model, build_messages(0, SUMMARY_TRIGGER + 1))

    assert "tainted" not in update
    assert "taint_sources" not in update
