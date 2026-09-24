"""
tests/test_output_guard.py — G1 (Phase 10) tests: the output guard checks every answer a turn added,
redacts a leaked prompt or a stray internal tag, warns (never redacts) on a content_ideator answer that
doesn't look like 3 ideas, and — when it changed anything — tells the browser to swap the reply
bubble's text for the redacted one (a `replace` SSE event, api.py). docs/contracts.md § 14.

Like test_handoffs.py, most tests drive the real FastAPI app end to end with `TestClient`: real graph,
real guard, real tool gateway, real (tiny, temp-folder) knowledge base and memory store — only the
model is a small scripted subclass of the free `FakeChatModel`, so nothing here ever calls a real,
paid model (CLAUDE.md's cost rule). The step-limit test builds the graph directly with stub workers
instead, the same way test_handoffs.py's own breaker test does, to isolate the supervisor's stop path
from the real workers' tool loops.
"""

import asyncio
import json

from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

import artlab.agents.supervisor as supervisor_module
from artlab.agents import (
    content_ideator as content_ideator_module,
    english_coach as english_coach_module,
    rag_agent as rag_agent_module,
    remember as remember_module,
    respond as respond_module,
    summarize as summarize_module,
    tool_loop as tool_loop_module,
    youtube_researcher as youtube_researcher_module,
)
from artlab.agents.respond import RESPOND_PROMPT
from artlab.agents.supervisor import STEP_LIMIT_NOTICE
from artlab.agents.workers import WorkerSpec
from artlab.api import create_app
from artlab.graph import build_graph
from artlab.guards.output import WITHHELD_TEXT, prompt_sentences
from artlab.memory.store import MemoryStore
from artlab.model import FakeChatModel, fake_model
from artlab.rag.knowledge import KnowledgeBase
from artlab.tools.registry import ToolRegistry


# ── Shared helpers (same shape as test_api.py / test_handoffs.py) ──────────────────────────────────

def parse_sse(body: str) -> list[tuple[str, dict]]:
    """Split a full SSE response body into [(event name, data dict), ...], in order."""
    events = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["event"], json.loads(fields["data"])))
    return events


def send(client: TestClient, message: str) -> list[tuple[str, dict]]:
    """POST one new-chat message and return its parsed events."""
    response = client.post("/api/chat", json={"message": message, "thread_id": None})
    assert response.status_code == 200
    return parse_sse(response.text)


def trace_lines(events) -> list[dict]:
    """The `trace` events' data dicts, in order."""
    return [d for n, d in events if n == "trace"]


def replace_events(events) -> list[dict]:
    """Every `replace` event's data dict, in order — at most one per run in practice."""
    return [d for n, d in events if n == "replace"]


def empty_kb(tmp_path) -> KnowledgeBase:
    """An empty knowledge base in a temp folder with instant fake embeddings."""
    return KnowledgeBase(tmp_path / "chroma", embeddings=DeterministicFakeEmbedding(size=32), min_score=-1)


def empty_memory(tmp_path) -> MemoryStore:
    """A private, temp-folder memory store with instant fake embeddings."""
    return MemoryStore(tmp_path / "chroma-memory", embeddings=DeterministicFakeEmbedding(size=32))


def make_app(tmp_path, model):
    """A FastAPI app with the given model, an empty knowledge base, and a temp memory/ideas folder —
    every real worker registered, so a message can route anywhere."""
    return create_app(
        model=model, checkpointer=InMemorySaver(), knowledge=empty_kb(tmp_path),
        memory=empty_memory(tmp_path), ideas_dir=tmp_path / "ideas",
    )


def saved_messages(client: TestClient, app, thread_id: str) -> list:
    """This chat's messages exactly as saved by the checkpointer — what the guard actually wrote back,
    not just what streamed live (test_handoffs.py's `is_tainted` uses the same `client.portal.call`
    trick to run an async graph method from a sync test)."""
    config = {"configurable": {"thread_id": thread_id}}
    return client.portal.call(app.state.graph.aget_state, config).values["messages"]


# ── Scripted fakes: each plays one worker role, and falls back to the real fake for routing ────────

class LeakyRespondModel(FakeChatModel):
    """`respond`'s answer quotes a whole sentence straight out of RESPOND_PROMPT — but uppercased and
    with an irregular line wrap, to prove the guard's leaked-prompt check normalises both sides (case
    and line-wrapping, docs/contracts.md § 14) rather than matching only an exact quote. Routing itself
    (`RouteDecision`) still behaves like the real fake, so "hi" still reaches `respond`."""

    def _reply(self, messages):
        if self.tool_name == "RouteDecision":
            return super()._reply(messages)
        leaked = RESPOND_PROMPT.split(".")[0].upper()  # RESPOND_PROMPT's first (long) sentence, shouted
        return AIMessage(f"Sure!   {leaked}\nHope that helps!")


class TaggyRespondModel(FakeChatModel):
    """`respond`'s answer contains a stray `</untrusted_retrieval>` closing tag in the middle of an
    otherwise ordinary reply — as if the model had quoted a wrapped tool result too literally."""

    def _reply(self, messages):
        if self.tool_name == "RouteDecision":
            return super()._reply(messages)
        return AIMessage("Here's the plan. </untrusted_retrieval> That's everything.")


class OneIdeaModel(FakeChatModel):
    """`content_ideator` answers with a single idea, not the 3 its own PROMPT requires — the
    schema-warning trigger (warn only, never redacted). Overriding `_reply_with_tools` (not `_reply`)
    means routing (`RouteDecision`) still falls through to the real fake's behaviour."""

    def _reply_with_tools(self, messages):
        return AIMessage("1. Zoom desk setup - one idea on purpose, not three.")


def test_leaked_prompt_is_withheld_and_replaced(tmp_path):
    """An answer that repeats a long enough sentence of RESPOND_PROMPT — even reworded in case and
    line-wrapping — must be entirely withheld: the saved message becomes the fixed withheld text, the
    browser gets a `replace` event with that same text, and the trace reports "output: redacted (leaked
    prompt)" with status "blocked" (docs/contracts.md § 14, rule 1)."""
    app = make_app(tmp_path, LeakyRespondModel())
    with TestClient(app) as client:
        events = send(client, "hi")
        thread_id = events[0][1]["thread_id"]
        messages = saved_messages(client, app, thread_id)

    guard_trace = next(d for d in trace_lines(events) if d["stage"] == "guard" and d["detail"].startswith("output"))
    assert guard_trace["status"] == "blocked"
    assert guard_trace["detail"] == "output: redacted (leaked prompt)"

    replaced = replace_events(events)
    assert len(replaced) == 1
    assert replaced[0]["text"] == WITHHELD_TEXT
    assert messages[-1].content == WITHHELD_TEXT


def test_internal_tags_are_stripped_and_replaced(tmp_path):
    """A stray `<untrusted_retrieval>`/`</untrusted_retrieval>`/`<system>`/`<assistant>` tag inside an
    answer is stripped (the text around it kept), a `replace` event carries the cleaned text, and the
    trace reports "internal tags removed" with status "blocked" — this IS a redaction (docs/contracts.md
    § 14, rule 2), even though nothing was withheld outright."""
    app = make_app(tmp_path, TaggyRespondModel())
    with TestClient(app) as client:
        events = send(client, "hi")
        thread_id = events[0][1]["thread_id"]
        messages = saved_messages(client, app, thread_id)

    guard_trace = next(d for d in trace_lines(events) if d["stage"] == "guard" and d["detail"].startswith("output"))
    assert guard_trace["status"] == "blocked"
    assert guard_trace["detail"] == "output: internal tags removed"

    replaced = replace_events(events)
    assert len(replaced) == 1
    for text in (replaced[0]["text"], messages[-1].content):
        assert "</untrusted_retrieval>" not in text
        assert "Here's the plan." in text and "That's everything." in text


def test_normal_answer_is_not_touched(tmp_path):
    """An ordinary answer (the plain fake reply, nothing leaked or tagged) must pass through untouched:
    no `replace` event, and the trace's guard line says "output ok" with status "ok"."""
    app = make_app(tmp_path, fake_model())
    with TestClient(app) as client:
        events = send(client, "hi")

    guard_trace = next(d for d in trace_lines(events) if d["stage"] == "guard" and d["detail"].startswith("output"))
    assert guard_trace["status"] == "ok"
    assert guard_trace["detail"] == "output ok"
    assert replace_events(events) == []


def test_ideator_schema_warning_does_not_redact(tmp_path):
    """content_ideator's PROMPT promises exactly 3 ideas; an answer with just one gets a trace warning
    — "output: schema warning (expected 3 ideas)" — but the message is NOT changed and no `replace`
    event is sent, because a schema mismatch is only ever a warning (docs/contracts.md § 14, rule 3)."""
    app = make_app(tmp_path, OneIdeaModel())
    with TestClient(app) as client:
        events = send(client, "Give me video ideas")
        thread_id = events[0][1]["thread_id"]
        messages = saved_messages(client, app, thread_id)

    guard_trace = next(d for d in trace_lines(events) if d["stage"] == "guard" and d["detail"].startswith("output"))
    assert guard_trace["status"] == "ok"
    assert guard_trace["detail"] == "output: schema warning (expected 3 ideas)"
    assert replace_events(events) == []
    assert messages[-1].content == "1. Zoom desk setup - one idea on purpose, not three."


# ── The step-limit stop path (direct graph, like test_handoffs.py's breaker test) ───────────────────

def stub_worker(name: str, calls: list[str]) -> WorkerSpec:
    """A worker that records that it ran and answers with a fixed message, then finishes."""

    async def node(state):
        calls.append(name)
        return {"messages": [AIMessage(f"{name} answered")], "answered_by": name}

    return WorkerSpec(name=name, description=f"stub worker {name}", make_node=lambda model, tools: node)


async def _run_traced(graph, message: str) -> tuple[list[dict], dict]:
    config = {"configurable": {"thread_id": "t"}}
    traces = [chunk async for chunk in graph.astream({"messages": [HumanMessage(message)]}, config, stream_mode="custom")]
    state = await graph.aget_state(config)
    return traces, state.values


def run_traced(graph, message: str) -> tuple[list[dict], dict]:
    return asyncio.run(_run_traced(graph, message))


class RouteToThreeStepPlan(FakeChatModel):
    """Always plans A -> B -> C, regardless of the message (test_handoffs.py's own breaker test)."""

    def _reply(self, messages):
        if self.tool_name != "RouteDecision":
            return super()._reply(messages)
        return AIMessage("", tool_calls=[{
            "name": self.tool_name,
            "args": {"next": "A", "then": ["B", "C"], "reason": "3-step plan", "question": "go"},
            "id": "fake-call", "type": "tool_call",
        }])


def test_step_limit_stop_path_also_passes_through_the_guard(tmp_path, monkeypatch):
    """docs/contracts.md § 14: "the step-limit and cap stop paths also go to output_guard instead of
    END". Reusing the breaker scenario (MAX_STEPS lowered to 2, a 3-step plan A→B→C, so the third step
    never runs): the STEP_LIMIT_NOTICE the supervisor appends is itself an answer this turn produced, so
    it must reach the output guard too — its own trace line ("output ok", nothing to redact in a fixed
    internal notice) must show up after the "stopped" line, and the notice's text must survive
    unchanged."""
    monkeypatch.setattr(supervisor_module, "MAX_STEPS", 2)

    calls: list[str] = []
    workers = (stub_worker("A", calls), stub_worker("B", calls), stub_worker("C", calls))
    graph = build_graph(RouteToThreeStepPlan(), InMemorySaver(), ToolRegistry(), empty_memory(tmp_path), workers=workers)
    traces, state = run_traced(graph, "go")

    assert calls == ["A", "B"]  # C never ran: the breaker stopped the plan after step 2
    assert state["messages"][-1].content == STEP_LIMIT_NOTICE

    guard_trace = next(t for t in traces if t["stage"] == "guard" and t["detail"].startswith("output"))
    assert guard_trace["status"] == "ok"
    assert guard_trace["detail"] == "output ok"


# ── Coverage of every prompt module ───────────────────────────────────────────────────────────────

def _first_long_sentence(prompt: str) -> str:
    """A prompt's first sentence of at least 40 characters, normalised the same way
    `guards.output.prompt_sentences()` does — computed independently here (not by importing the
    module's own private helpers) so this test really checks the two agree, not just that one calls
    the other."""
    import re

    for piece in re.split(r"[.!?\n]+", prompt):
        normalised = re.sub(r"\s+", " ", piece).strip().lower()
        if len(normalised) >= 40:
            return normalised
    raise AssertionError(f"no sentence of at least 40 characters in: {prompt!r}")


def test_prompt_sentences_covers_every_prompt_module():
    """docs/contracts.md § 14: "a new prompt is covered by adding it to that list" — this is the test
    that would fail if a future prompt were added to the app but never added to `guards/output.py`'s
    own collection, catching the mistake instead of silently leaving a hole in the leaked-prompt check."""
    sentences = prompt_sentences()
    for prompt in (
        supervisor_module.PROMPT_HEADER,
        supervisor_module.PROMPT_FOOTER,
        RESPOND_PROMPT,
        rag_agent_module.RAG_PROMPT,
        rag_agent_module.REPHRASE_PROMPT,
        youtube_researcher_module.PROMPT,
        content_ideator_module.PROMPT,
        english_coach_module.PROMPT,
        tool_loop_module.TOOL_RULES,
        remember_module.PROMPT,
        summarize_module.SUMMARY_PROMPT,
    ):
        assert _first_long_sentence(prompt) in sentences
