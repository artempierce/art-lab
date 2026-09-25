"""
tests/test_runs.py — Phase 13 tests (P13, docs/contracts.md § 15): every chat request gets one row
in the run log, with the right status, and the Runs page's endpoints can list and replay them.

Like test_api.py, these drive the real FastAPI app end to end with `TestClient`: real graph, real
guard, real tool gateway — only the model is the free `FakeChatModel` (or one of its small scripted
subclasses), and every store (chat, memory, ideas, runs) is a temp one, so no test ever touches the
owner's real data and every test here is instant and costs $0 (CLAUDE.md's cost rule).
"""

import json

from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding
from langgraph.checkpoint.memory import InMemorySaver

from artlab.api import create_app
from artlab.guards.input import MAX_INPUT_CHARS
from artlab.memory.store import MemoryStore
from artlab.model import fake_model
from artlab.rag.knowledge import KnowledgeBase
from artlab.runs.store import RunStore

# LeakyRespondModel is the scripted fake test_output_guard.py already built for the redaction path
# (an answer that quotes a whole sentence of RESPOND_PROMPT, uppercased): reused here rather than
# rewritten, so both test files agree on exactly what "a redacted answer" looks like.
from test_output_guard import LeakyRespondModel


def parse_sse(body: str) -> list[tuple[str, dict]]:
    """Split a full SSE response body into [(event name, data dict), ...], in order (test_api.py)."""
    events = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["event"], json.loads(fields["data"])))
    return events


def send(client: TestClient, message: str, thread_id: str | None = None) -> list[tuple[str, dict]]:
    """POST one chat message and return its parsed events. thread_id None starts a new chat (test_api.py)."""
    response = client.post("/api/chat", json={"message": message, "thread_id": thread_id})
    assert response.status_code == 200
    return parse_sse(response.text)


def resume(client: TestClient, thread_id: str, approval_id: str, approve: bool):
    """POST /api/chat/resume and return its parsed events (test_approvals.py)."""
    response = client.post("/api/chat/resume", json={"thread_id": thread_id, "id": approval_id, "approve": approve})
    assert response.status_code == 200
    return parse_sse(response.text)


def answer(events) -> str:
    """Join all `token` events back into the full answer text, as the browser does (test_api.py)."""
    return "".join(data["text"] for name, data in events if name == "token")


def start_trace_id(events) -> str:
    """The trace ID this request was given, from its own `start` event."""
    return next(data["trace_id"] for name, data in events if name == "start")


def approval_payload(events) -> dict:
    """The one `approval` event's data dict (test_approvals.py)."""
    approvals = [data for name, data in events if name == "approval"]
    assert len(approvals) == 1, f"expected exactly one approval event, got {len(approvals)}"
    return approvals[0]


def empty_kb(tmp_path) -> KnowledgeBase:
    """An empty knowledge base in a temp folder with instant fake embeddings (test_api.py)."""
    return KnowledgeBase(tmp_path / "chroma", embeddings=DeterministicFakeEmbedding(size=32), min_score=-1)


def empty_memory(tmp_path) -> MemoryStore:
    """A private, temp-folder memory store with instant fake embeddings (test_api.py)."""
    return MemoryStore(tmp_path / "chroma-memory", embeddings=DeterministicFakeEmbedding(size=32))


def make_app(tmp_path, model=None, **kwargs) -> TestClient:
    """A FastAPI app wired entirely to temp storage — including its own temp RunStore, so a test's
    recorded rows never land in the owner's real data/runs.db — with every real worker registered
    (an `ideas_dir` so save_ideas can run) unless the caller overrides `runs` or `ideas_dir` itself."""
    kwargs.setdefault("runs", RunStore(tmp_path / "runs.db"))
    kwargs.setdefault("ideas_dir", tmp_path / "ideas")
    return create_app(
        model=model or fake_model(), checkpointer=InMemorySaver(),
        knowledge=empty_kb(tmp_path), memory=empty_memory(tmp_path), **kwargs,
    )


def test_normal_chat_records_one_row_with_status_ok_and_totals(tmp_path):
    """A plain "hi" answers directly and finishes cleanly: exactly one row is recorded, keyed by the
    same trace ID the SSE stream reported, with status "ok" and every numeric field a real number
    (docs/contracts.md § 15's "one row per request", the "ok" default of the status rules)."""
    with TestClient(make_app(tmp_path, fake_model(["Hello from the fake model."]))) as client:
        events = send(client, "hi")
        trace_id = start_trace_id(events)

        rows = client.get("/api/runs").json()
        assert len(rows) == 1
        row = rows[0]
        assert row["trace_id"] == trace_id
        assert row["prompt"] == "hi"
        assert row["status"] == "ok"
        assert row["answered_by"] == "Arty"
        for field in ("input_tokens", "output_tokens", "ms", "steps"):
            assert isinstance(row[field], int) and row[field] >= 0
        assert isinstance(row["cost_usd"], float) and row["cost_usd"] >= 0


def test_blocked_message_records_status_blocked(tmp_path):
    """A message the input guard refuses outright (too long, docs/contracts.md § 15's "blocked")
    never reaches the model, but it still gets its own row, with status "blocked" — so a blocked
    request shows up on the Runs page just like any other."""
    with TestClient(make_app(tmp_path)) as client:
        events = send(client, "x" * (MAX_INPUT_CHARS + 1))
        trace_id = start_trace_id(events)

        row = client.get(f"/api/runs/{trace_id}").json()
        assert row["status"] == "blocked"


def test_approval_pause_and_its_resume_each_record_their_own_row(tmp_path):
    """Asking to save ideas pauses the run for your Approve/Reject click: that request gets its own
    row with status "approval" (docs/contracts.md § 15). Clicking Approve is itself a new request —
    POST /api/chat/resume — and records a second, separate row of its own, with status "ok" once the
    tool has actually run."""
    with TestClient(make_app(tmp_path, fake_model(["Here are 3 desk-setup ideas."]))) as client:
        first = send(client, "Give me 3 video ideas about desk setups")
        thread_id = first[0][1]["thread_id"]
        assert client.get(f"/api/runs/{start_trace_id(first)}").json()["status"] == "ok"

        pending_events = send(client, "Save those ideas", thread_id)
        pending = approval_payload(pending_events)
        assert client.get(f"/api/runs/{start_trace_id(pending_events)}").json()["status"] == "approval"

        resumed = resume(client, thread_id, pending["id"], approve=True)
        assert "Saved" in answer(resumed)
        assert client.get(f"/api/runs/{start_trace_id(resumed)}").json()["status"] == "ok"

        assert len(client.get("/api/runs").json()) == 3  # one row per request, not per chat


def test_output_guard_redaction_records_status_redacted(tmp_path):
    """When the output guard withholds a leaked-prompt answer (docs/contracts.md § 14), the request
    that produced it is recorded with status "redacted" (§ 15) — distinct from "blocked", since
    nothing here was refused outright, only cleaned up after the fact."""
    with TestClient(make_app(tmp_path, LeakyRespondModel())) as client:
        events = send(client, "hi")
        row = client.get(f"/api/runs/{start_trace_id(events)}").json()
        assert row["status"] == "redacted"


def test_runs_are_listed_newest_first_without_trace_lines(tmp_path):
    """GET /api/runs is the Runs page's table: newest request first, and light enough to list many
    rows because it never includes a row's trace lines (docs/contracts.md § 15) — those are only
    fetched once a row is clicked, via GET /api/runs/{trace_id}."""
    with TestClient(make_app(tmp_path)) as client:
        older = start_trace_id(send(client, "about cable trays"))
        newer = start_trace_id(send(client, "about desk lamps"))

        rows = client.get("/api/runs").json()
        assert [r["trace_id"] for r in rows] == [newer, older]
        assert all("trace" not in r and "trace_json" not in r for r in rows)


def test_run_detail_includes_its_trace_lines(tmp_path):
    """GET /api/runs/{trace_id} is what a clicked row loads: the same fields as the table plus a
    `trace` list, so the page can replay the run with TracePanel's RunBlock (docs/contracts.md § 15).
    404 if no row was ever recorded under that ID."""
    with TestClient(make_app(tmp_path)) as client:
        events = send(client, "hi")
        trace_id = start_trace_id(events)
        live_stages = [(d["stage"], d["status"]) for n, d in events if n == "trace"]

        row = client.get(f"/api/runs/{trace_id}").json()
        assert [(t["stage"], t["status"]) for t in row["trace"]] == live_stages

        assert client.get("/api/runs/does-not-exist").status_code == 404


def test_evals_latest_is_404_with_no_reports(tmp_path):
    """Before the evals runner has ever been run (or its reports folder doesn't even exist yet), GET
    /api/evals/latest is a clean 404 — the Runs page's fallback text, not a crash."""
    app = create_app(
        model=fake_model(), checkpointer=InMemorySaver(), knowledge=empty_kb(tmp_path), memory=empty_memory(tmp_path),
        evals_reports_dir=tmp_path / "reports",
    )
    with TestClient(app) as client:
        assert client.get("/api/evals/latest").status_code == 404


def test_evals_latest_returns_the_newest_report(tmp_path):
    """GET /api/evals/latest reads the newest report JSON by filename (Phase 11's
    `<YYYY-MM-DD-HHMM>.json` naming sorts the same way by name as by time) and returns its summary
    as is."""
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / "2026-01-01-0900.json").write_text(json.dumps({"started_at": "old", "cost_usd": 0.01}))
    (reports_dir / "2026-01-02-0900.json").write_text(json.dumps({"started_at": "new", "cost_usd": 0.02}))

    app = create_app(
        model=fake_model(), checkpointer=InMemorySaver(), knowledge=empty_kb(tmp_path), memory=empty_memory(tmp_path),
        evals_reports_dir=reports_dir,
    )
    with TestClient(app) as client:
        assert client.get("/api/evals/latest").json() == {"started_at": "new", "cost_usd": 0.02}


class BrokenRunStore(RunStore):
    """A run store whose `add()` always fails — proves a storage problem (a full disk, a locked
    file) can never break the chat itself (docs/contracts.md § 15): `stream_run` only logs it."""

    def add(self, run: dict) -> None:
        raise RuntimeError("disk is full (simulated)")


def test_a_broken_run_store_never_breaks_the_chat_stream(tmp_path):
    """Recording is wrapped in its own try/except (docs/contracts.md § 15): even when the run log
    itself is broken, the chat still streams its answer normally, with no `error` event."""
    with TestClient(make_app(tmp_path, fake_model(["Hello from the fake model."]), runs=BrokenRunStore(tmp_path / "runs.db"))) as client:
        events = send(client, "hi")
        assert "error" not in [name for name, _ in events]
        assert answer(events) == "Hello from the fake model."
        assert events[-1][0] == "done"
