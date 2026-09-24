"""
tests/test_approvals.py — Phase 6 tests (P6a, docs/contracts.md § 10): a mutating tool never just
runs from a model's request — it waits for your Approve/Reject click, streamed as an `approval` SSE
event and answered through POST /api/chat/resume. Covers the whole path end to end: content_ideator's
first mutating tool (save_ideas), the approval graph node (`interrupt()`/`Command(resume=...)`), the
gateway's `ApprovalRequired`/`run_approved`, and the API's approve/reject/cancel rules.

Like test_api.py, these drive the real FastAPI app end to end with `TestClient`: real graph, real
guard, real tool gateway — only the model is the free `FakeChatModel` and `save_ideas` is pointed at a
temp folder (`ideas_dir=tmp_path / "ideas"`), so no test ever writes into the repo's own data/, and
every test here is instant and costs $0 (CLAUDE.md's cost rule).
"""

import asyncio
import json
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from artlab.api import create_app
from artlab.memory.store import MemoryStore
from artlab.model import FakeChatModel, fake_model
from artlab.rag.ingest import ingest
from artlab.rag.knowledge import KnowledgeBase
from artlab.tools.registry import ToolDenied, ToolRegistry


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


def resume(client: TestClient, thread_id: str, approval_id: str, approve: bool):
    """POST /api/chat/resume and return the raw response (so a test can check the status code itself
    for the 404/409 cases, not just the 200 happy path)."""
    return client.post("/api/chat/resume", json={"thread_id": thread_id, "id": approval_id, "approve": approve})


def answer(events) -> str:
    """Join all `token` events back into the full answer text, as the browser does (test_api.py)."""
    return "".join(data["text"] for name, data in events if name == "token")


def event_names(events) -> list[str]:
    """Just the event names, in order — e.g. to check an `approval` event is (or isn't) present."""
    return [name for name, _ in events]


def approval_payload(events) -> dict:
    """The one `approval` event's data dict. Fails loudly if there isn't exactly one — a test asserting
    on its fields should never silently pass because the event was missing."""
    approvals = [data for name, data in events if name == "approval"]
    assert len(approvals) == 1, f"expected exactly one approval event, got {len(approvals)}"
    return approvals[0]


def empty_kb(tmp_path) -> KnowledgeBase:
    """An empty knowledge base in a temp folder with instant fake embeddings (test_api.py)."""
    return KnowledgeBase(tmp_path / "chroma", embeddings=DeterministicFakeEmbedding(size=32), min_score=-1)


def empty_memory(tmp_path) -> MemoryStore:
    """A private, temp-folder memory store with instant fake embeddings (Phase 7, M1) — every real
    graph needs one now that `recall`/`remember` are permanent nodes."""
    return MemoryStore(tmp_path / "chroma-memory", embeddings=DeterministicFakeEmbedding(size=32))


def kb_with_policy(tmp_path) -> KnowledgeBase:
    """A knowledge base holding one small policy file, so a question about it taints the chat
    (test_api.py's `kb_with_policy` fixture, reused here as a plain function)."""
    doc = tmp_path / "sponsorship.md"
    doc.write_text("# Sponsorship policy\n\n## Disclosure\n\nDisclose every sponsorship out loud in the first 30 seconds.")
    kb = empty_kb(tmp_path)
    ingest([doc], kb)
    return kb


def saved_files(ideas_dir) -> list:
    """Every ideas file written so far, or [] if save_ideas has never run — used to prove a reject (or
    a still-pending request) really did write nothing."""
    return sorted(ideas_dir.glob("*.md")) if ideas_dir.exists() else []


@contextmanager
def app_client(tmp_path, model, knowledge=None):
    """A TestClient wired the way every test here needs it: the given fake model, in-memory chat
    storage, and save_ideas pointed at a temp folder so nothing touches the repo's real data/ideas.
    A context manager (not a plain fixture) because `with TestClient(app)` is what actually runs the
    app's startup (lifespan), building the graph and tool registry — see test_api.py's `client` fixture.
    Yields (client, app) — `app` is needed to peek at saved state directly (`app.state.graph`).
    """
    app = create_app(
        model=model, checkpointer=InMemorySaver(),
        knowledge=knowledge or empty_kb(tmp_path), ideas_dir=tmp_path / "ideas",
        memory=empty_memory(tmp_path),
    )
    with TestClient(app) as client:
        yield client, app


def pending_approval(client: TestClient, app, thread_id: str):
    """Read `pending_approval` straight from the graph's saved state (not from any SSE event) — used
    to prove it's really been cleared, the same `client.portal.call` pattern test_api.py's
    `test_a_cited_answer_taints_the_chat_and_it_stays_tainted` uses to bridge a sync test into the
    checkpointer's async `aget_state`."""
    config = {"configurable": {"thread_id": thread_id}}
    return client.portal.call(app.state.graph.aget_state, config).values.get("pending_approval")


def test_approve_saves_the_ideas_and_clears_the_pending_request(tmp_path):
    """S3 (approve): "give me ideas" is answered directly, no card — content_ideator's save_ideas hint
    (FAKE_TOOL_HINTS) only fires on "save". "Save those ideas" then asks for save_ideas, which the
    gateway never runs directly (ApprovalRequired): the stream carries an `approval` event instead of
    more answer text, and nothing is written to disk yet. Approving through /api/chat/resume actually
    runs the tool, and the file appears with the exact text that was on the card."""
    with app_client(tmp_path, fake_model(["Here are 3 desk-setup ideas."])) as (client, app):
        ideas_dir = tmp_path / "ideas"

        events = send(client, "Give me 3 video ideas about desk setups")
        thread_id = events[0][1]["thread_id"]
        assert "approval" not in event_names(events)
        assert answer(events) == "Here are 3 desk-setup ideas."
        assert saved_files(ideas_dir) == []

        events = send(client, "Save those ideas", thread_id)
        pending = approval_payload(events)
        assert pending["tool"] == "save_ideas" and pending["agent"] == "content_ideator"
        assert saved_files(ideas_dir) == []  # still nothing written — only the click runs it

        response = resume(client, thread_id, pending["id"], approve=True)
        assert response.status_code == 200
        resumed = parse_sse(response.text)
        assert "Saved" in answer(resumed)

        files = saved_files(ideas_dir)
        assert len(files) == 1
        assert pending["args"]["ideas"] in files[0].read_text()
        assert pending_approval(client, app, thread_id) is None


def test_reject_writes_no_file_and_says_so(tmp_path):
    """S3 (reject): the same request, but rejected — no file is ever written, the reply says nothing
    was saved, and the pending request is cleared just like an approval clears it."""
    with app_client(tmp_path, fake_model(["Here are 3 desk-setup ideas."])) as (client, app):
        ideas_dir = tmp_path / "ideas"

        thread_id = send(client, "Give me 3 video ideas about desk setups")[0][1]["thread_id"]
        pending = approval_payload(send(client, "Save those ideas", thread_id))

        response = resume(client, thread_id, pending["id"], approve=False)
        assert response.status_code == 200
        resumed = parse_sse(response.text)
        assert "nothing was saved" in answer(resumed)
        assert saved_files(ideas_dir) == []
        assert pending_approval(client, app, thread_id) is None


def test_tainted_chat_approval_card_names_the_research_tool(tmp_path):
    """A chat that has already read untrusted content (here: a knowledge-base search, rag_agent's own
    taint source per docs/contracts.md § 10) still only needs approval, not a harder refusal — but the
    card must say so: `tainted` is True and `taint_sources` names what tainted it, even though this
    turn's own tool call (save_ideas) never itself produced untrusted output.

    Uses a knowledge-base question as the "research" step (rather than youtube_researcher's trend
    search) because rag_agent is one of the two workers P6a wires up to report `taint_sources`
    (docs/contracts.md § 10's ownership table) — the mechanism under test is the same either way."""
    with app_client(
        tmp_path, fake_model(["Here are 3 desk-setup ideas."]), knowledge=kb_with_policy(tmp_path)
    ) as (client, app):
        thread_id = send(client, "What is our sponsorship disclosure rule?")[0][1]["thread_id"]
        config = {"configurable": {"thread_id": thread_id}}
        state = client.portal.call(app.state.graph.aget_state, config)
        assert state.values.get("tainted") is True

        pending = approval_payload(send(client, "Save those ideas", thread_id))
        assert pending["tainted"] is True
        assert "search_knowledge" in pending["taint_sources"]


def test_after_youtube_research_the_card_names_the_youtube_tool(tmp_path):
    """The flow the "ask with a warning" decision was made for (S2 → S3): research first, then "Save
    those ideas". The card must be tainted AND name the YouTube tool that tainted the chat, not just say
    "untrusted", so the owner knows which outside text to be wary of before approving."""
    with app_client(tmp_path, fake_model(["Here are 3 desk-setup ideas."])) as (client, app):
        thread_id = send(client, "What's trending in desk-setup videos?")[0][1]["thread_id"]

        pending = approval_payload(send(client, "Save those ideas", thread_id))
        assert pending["tainted"] is True
        assert "query_youtube_trends" in pending["taint_sources"]
        assert saved_files(tmp_path / "ideas") == []  # nothing written before the click


def test_resume_with_the_wrong_id_is_409(tmp_path):
    """A stale card (an id that isn't the one actually pending) must be refused with 409, not silently
    accepted — approving the wrong request would run a tool the owner never actually clicked Approve
    on."""
    with app_client(tmp_path, fake_model(["Here are 3 desk-setup ideas."])) as (client, app):
        thread_id = send(client, "Give me 3 video ideas about desk setups")[0][1]["thread_id"]
        send(client, "Save those ideas", thread_id)  # now something IS pending, just not this id

        response = resume(client, thread_id, "not-the-real-id", approve=True)
        assert response.status_code == 409


def test_resume_with_nothing_pending_is_404(tmp_path):
    """Resuming a chat that never asked for approval is a clean 404, not a crash — there's nothing for
    the click to act on."""
    with app_client(tmp_path, fake_model(["Here are 3 desk-setup ideas."])) as (client, app):
        thread_id = send(client, "Give me 3 video ideas about desk setups")[0][1]["thread_id"]  # answered directly, no card
        response = resume(client, thread_id, "any-id", approve=True)
        assert response.status_code == 404


def test_a_new_message_auto_rejects_a_still_pending_request(tmp_path):
    """docs/contracts.md § 10: you can't leave a stale card that gets approved later after you've
    moved on to something else. A new /api/chat message while one is still pending resolves it as a
    reject first (its own trace line explains why), writes nothing, and then answers the new message
    normally — the chat keeps working, it just loses the stale card."""
    with app_client(tmp_path, fake_model(["Here are 3 desk-setup ideas.", "Hi there!"])) as (client, app):
        ideas_dir = tmp_path / "ideas"

        thread_id = send(client, "Give me 3 video ideas about desk setups")[0][1]["thread_id"]
        send(client, "Save those ideas", thread_id)  # left pending, never resumed

        events = send(client, "hi", thread_id)
        assert "error" not in event_names(events)
        cancel_lines = [d for n, d in events if n == "trace" and "cancelled by a new message" in d.get("detail", "")]
        assert len(cancel_lines) == 1
        assert saved_files(ideas_dir) == []  # the abandoned save_ideas request never ran
        assert pending_approval(client, app, thread_id) is None


def test_approved_args_are_exactly_what_was_shown_not_a_fresh_model_call(tmp_path):
    """The whole reason the approval node just reads state instead of re-running the worker's tool loop
    (docs/contracts.md § 10): resuming must never call the model a second time and risk saving
    different text than what was on the card. This model answers a second call with different args; if
    the approval node ever made one, the saved file would show it — proving by construction that it
    doesn't."""

    class ChangingArgsModel(FakeChatModel):
        """Every call to `_reply_with_tools` proposes save_ideas with *different* args, numbered by how
        many times it's been called, so a test can check exactly how many times (and with what
        arguments) the model was actually asked."""

        calls: list[str] = Field(default_factory=list)

        def _reply_with_tools(self, messages):
            version = f"version {len(self.calls) + 1}"
            self.calls.append(version)
            return AIMessage("", tool_calls=[{
                "name": "save_ideas", "args": {"ideas": version}, "id": f"call-{version}", "type": "tool_call",
            }])

    model = ChangingArgsModel()
    with app_client(tmp_path, model) as (client, app):
        ideas_dir = tmp_path / "ideas"

        events = send(client, "Save my ideas please")
        thread_id = events[0][1]["thread_id"]
        pending = approval_payload(events)
        assert pending["args"]["ideas"] == "version 1"
        assert len(model.calls) == 1  # the loop stopped at once; it never asked again before pausing

        resumed = parse_sse(resume(client, thread_id, pending["id"], approve=True).text)
        assert "Saved" in answer(resumed)
        assert len(model.calls) == 1  # still just the one call: the approval node never touches the model

        files = saved_files(ideas_dir)
        assert len(files) == 1
        saved_text = files[0].read_text()
        assert "version 1" in saved_text
        assert "version 2" not in saved_text


def test_run_approved_still_enforces_the_allow_list():
    """run_approved (tools/registry.py) is the one path a mutating tool actually executes — from the
    approval node, after your click — but it must still check the allow-list exactly like `call` does
    for a model's request: an agent that isn't allowed to use the tool is refused, not trusted just
    because *some* approval happened in this chat."""
    tools = ToolRegistry()
    tools.register("save", lambda text: "saved", tier="mutating", allowed_agents={"content_ideator"}, description="")

    with pytest.raises(ToolDenied, match="other_agent may not use save"):
        asyncio.run(tools.run_approved("other_agent", "save", {"text": "x"}))
