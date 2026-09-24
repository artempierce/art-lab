"""
api.py — the HTTP API the web page talks to. Built with FastAPI.

Endpoints:

    POST /api/chat                 send one message; the reply streams back as server-sent events
    POST /api/chat/resume          answer a pending Approve/Reject card; streams the same way (Phase 6)
    GET  /api/threads              list all chats, newest first (the left sidebar)
    GET  /api/threads/{thread_id}  one chat's full history, with each answer's sources
    GET  /api/agents               the team and their tools, for the sidebar's Team panel (X3)

How POST /api/chat (and /api/chat/resume) stream. The response is *server-sent events* (SSE): a
long-lived HTTP response made of small text blocks, each one looking like

    event: token
    data: {"text": "Hello"}
    <blank line>

The browser reads them one by one as they arrive (frontend/src/api.ts → streamChat). Six event
types exist; `start` is always first, `done` always last, and `approval` (Phase 6, docs/contracts.md
§ 10) replaces `token` when the run stops to wait for your click instead of finishing:

    start      → trace ID and chat ID                                            (once)
    trace      → one line for the trace panel                                    (once per graph step)
    token      → a piece of the answer text                                      (many)
    approval   → a data-changing tool is waiting for Approve/Reject: {id, agent, tool, args,
                 tainted, taint_sources}                                         (instead of more tokens)
    done       → tokens used, cost, duration, the answer's sources                (once, at the end)
    error      → what went wrong                                        (instead of done, if something failed)

Both endpoints share one generator, `stream_run` (below), for this trace/token/approval/done/error
tail — the only difference between a first message and a resume is what starts the graph running
(a new `HumanMessage`, or `Command(resume=...)`), which the caller passes in as `run_input`.

Run the server:  cd backend && uv run uvicorn artlab.api:app --reload --port 8000
"""

import json
import time
import uuid
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from artlab.agents.capabilities import team
from artlab.agents.common import text_of
from artlab.agents.workers import WORKERS
from artlab.config import DB_PATH, IDEAS_DIR, REPO_ROOT
from artlab.graph import build_graph
from artlab.guards.classifier import InjectionClassifier, load_classifier
from artlab.model import cost_usd, make_model
from artlab.rag.knowledge import KnowledgeBase
from artlab.tools.catalog import build_tools

# Load settings (API keys, ARTLAB_FAKE_LLM, LangSmith) from the repo's .env into environment
# variables. LangChain and LangSmith read them from there automatically.
load_dotenv(REPO_ROOT / ".env")

# Nodes whose *streamed* model output is internal and must not appear in the chat: the supervisor's
# model call produces a routing decision (structured output), streamed as tool-call chunks, not an
# answer for you. This only hides those chunks — a whole message the node adds directly (e.g. the
# step-limit note, Phase 3) still reaches the browser; see the AIMessageChunk check below.
HIDDEN_NODES = {"supervisor"}


class ChatRequest(BaseModel):
    """The JSON body of POST /api/chat. FastAPI validates it and returns 422 if it doesn't fit.

    message    what you typed; must not be empty
    thread_id  which chat it belongs to; null starts a new chat (the server makes an ID)
    """

    message: str = Field(min_length=1)
    thread_id: str | None = None


class ResumeRequest(BaseModel):
    """The JSON body of POST /api/chat/resume (Phase 6, docs/contracts.md § 10).

    thread_id  which chat's pending approval this answers
    id         the pending request's own id (from its `approval` event) — must match, or 409
    approve    True to run the tool for real, False to skip it
    """

    thread_id: str
    id: str
    approve: bool


def sse(event: str, data: dict) -> str:
    """Format one server-sent event: an `event:` line, a `data:` line (JSON), and a blank line."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def stream_run(graph, run_input, config: dict, started: float):
    """Run the graph once and yield the SSE events for it: trace/token as they happen, then one of
    `approval`+`done`, plain `done`, or `error`. Shared by POST /api/chat (a new `HumanMessage`) and
    POST /api/chat/resume (`Command(resume=...)`, Phase 6, docs/contracts.md § 10) — both endpoints
    only differ in what starts the graph and in the `start` event they send first, so this is the one
    place that has to know the event shapes.

    Args:
        graph:   the compiled chat graph (app.state.graph)
        run_input: what to hand `graph.astream` — either `{"messages": [...]}` (a normal turn) or a
                 `Command(resume=...)` (continuing a paused one)
        config:  `{"configurable": {"thread_id": ...}}`, plus a run_id for a fresh run
        started: `time.perf_counter()` reading from just before this call, for the `done` event's `ms`
    """
    tokens_in = tokens_out = 0
    try:
        # Run the graph with two stream modes at once. Each item is (mode, chunk):
        #   "custom"   → a dict a node wrote with get_stream_writer()  → forward as `trace`
        #   "messages" → (message piece, metadata) as a model writes → forward as `token`
        # The checkpointer loads this chat's earlier messages first, so we only send the new ones. If
        # the run pauses at the `approval` node (interrupt()), astream just stops here, the same as if
        # the run had ended — that's why the pause is detected below, after this loop, not inside it.
        async for mode, chunk in graph.astream(run_input, config, stream_mode=["messages", "custom"]):
            if mode == "custom":
                # Keep a running token total for the `done` summary.
                tokens_in += chunk.get("input_tokens", 0)
                tokens_out += chunk.get("output_tokens", 0)
                yield sse("trace", chunk)
            else:
                message, meta = chunk
                # Forward only assistant text from answering nodes: streamed pieces ("AIMessageChunk")
                # and whole replies a node added directly ("ai", e.g. the guard's refusal). Only the
                # streamed chunks are hidden for HIDDEN_NODES — a whole message such a node adds
                # directly (e.g. the supervisor's step-limit note) is never a routing decision, so it
                # still reaches the browser.
                if message.type == "AIMessageChunk" and meta.get("langgraph_node") in HIDDEN_NODES:
                    continue
                if message.type in ("ai", "AIMessageChunk") and (text := text_of(message)):
                    yield sse("token", {"text": text})
    except Exception as exc:
        # Anything that goes wrong (missing API key, network error, …) becomes an `error`
        # event the page can show, instead of a silently broken stream.
        yield sse("error", {"message": f"{type(exc).__name__}: {exc}"})
        return

    done = {
        "input_tokens": tokens_in,
        "output_tokens": tokens_out,
        "cost_usd": cost_usd(tokens_in, tokens_out),
        "ms": round((time.perf_counter() - started) * 1000),
        "sources": [],
    }
    # `.next` names the node(s) the graph would run next; it's `("approval",)` only when that node
    # just called `interrupt()` and paused there (see the file header) — otherwise the run finished.
    state = await graph.aget_state(config)
    if state.next == ("approval",):
        yield sse("approval", state.values["pending_approval"])
        yield sse("done", done)
        return

    # The final answer is the last message; rag_agent attached its sources to it.
    last = state.values["messages"][-1]
    done["sources"] = last.additional_kwargs.get("sources", []) if last.type == "ai" else []
    yield sse("done", done)


def create_app(
    model: BaseChatModel | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    knowledge: KnowledgeBase | None = None,
    classifier: InjectionClassifier | None = None,
    ideas_dir: Path | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    Args:
        model:        chat model to use; None means "decide from .env" (see model.make_model)
        checkpointer: chat storage to use; None means "open data/artlab.db"
        knowledge:    knowledge base to search; None means "open data/chroma" (fill it with the ingest command)
        classifier:   the guard's layer-2 injection classifier; None means "off" (docs/contracts.md § 8).
                      Tests never pass one, so they stay fast and deterministic. The real server passes
                      `load_classifier()` (see the last line of this file), which is None until it's
                      been downloaded by hand.
        ideas_dir:    where `save_ideas` writes (Phase 6, docs/contracts.md § 10); None means the real
                      data/ideas folder. Tests pass a temp folder so a test run never touches it.

    The real server calls this with no arguments except the classifier (see the last line of this
    file). Tests pass a fake model, in-memory chat storage and a temporary knowledge base, so they
    run the whole API for free.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Runs once around the server's life: setup before `yield`, cleanup after.

        Opens the chat database (unless a checkpointer was passed in), opens the knowledge base, builds
        the tool registry and the graph, and stores them on `app.state` for the endpoints to use.
        `app.state.tools` is the same registry the graph's workers call through — GET /api/agents (X3)
        reads it, so the Team panel can never show a tool the workers don't actually have.
        `AsyncExitStack` closes the database connection cleanly when the server stops.
        """
        async with AsyncExitStack() as stack:
            saver = checkpointer
            if saver is None:
                DB_PATH.parent.mkdir(exist_ok=True)
                saver = await stack.enter_async_context(AsyncSqliteSaver.from_conn_string(str(DB_PATH)))
            tools = build_tools(knowledge or KnowledgeBase(), ideas_dir or IDEAS_DIR)
            app.state.checkpointer = saver
            app.state.tools = tools
            app.state.graph = build_graph(model or make_model(), saver, tools, classifier=classifier)
            yield

    app = FastAPI(title="Art Lab", lifespan=lifespan)

    @app.post("/api/chat")
    async def chat(req: ChatRequest):
        """Run the graph on one message and stream everything that happens back to the browser.

        Returns immediately with a streaming response; the `events()` generator below then produces
        the SSE events one at a time while the graph runs.

        Phase 6 (docs/contracts.md § 10): if this chat still has a pending approval from an earlier
        turn — you never clicked Approve or Reject on it, and now you've typed something else — it's
        auto-rejected first, silently, so it can never be approved later after you've moved on. A
        trace line says so, right after `start`, before the new message's own trace lines.
        """
        graph = app.state.graph

        # Two IDs per request:
        #   trace_id  identifies this one run. It's shown in the trace panel and is also the
        #             LangSmith run ID, so you can search LangSmith for exactly this message.
        #   thread_id identifies the chat. The checkpointer loads and saves history under it.
        trace_id = str(uuid.uuid4())
        thread_id = req.thread_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}, "run_id": trace_id, "run_name": "art-lab chat"}

        async def events():
            """Yield the SSE events for this run: start, an optional cancellation line, then the
            shared trace/token/done tail from `stream_run`."""
            yield sse("start", {"trace_id": trace_id, "thread_id": thread_id})
            started = time.perf_counter()

            state = await graph.aget_state(config)
            if state.next == ("approval",):
                pending_id = state.values["pending_approval"]["id"]
                # Actually resolve it — this clears pending_approval and saves a rejection reply to
                # history, exactly like a real Reject click — but its own trace/token events are
                # drained, not shown: the browser never asked for this card, so it only sees the one
                # line below explaining why it's gone.
                async for _mode, _chunk in graph.astream(
                    Command(resume={"id": pending_id, "approve": False}), config, stream_mode=["messages", "custom"]
                ):
                    pass
                yield sse("trace", {
                    "stage": "tool", "status": "ok",
                    "detail": "pending approval cancelled by a new message", "ms": 0,
                })

            async for event in stream_run(graph, {"messages": [HumanMessage(req.message)]}, config, started):
                yield event

        # StreamingResponse sends each string `events()` yields as soon as it's yielded.
        # "no-cache" stops proxies from buffering the stream.
        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.post("/api/chat/resume")
    async def resume(req: ResumeRequest):
        """Answer a pending Approve/Reject card and stream the continuation (Phase 6, docs/contracts.md
        § 10) — the same event shapes `chat` above produces, via the shared `stream_run`.

        404 if this chat has nothing pending (already resolved, or never asked); 409 if `req.id` names
        a different request than the one actually pending (a stale card, e.g. from a chat you've since
        moved past). Both checks run *before* the streaming response starts, so they come back as
        ordinary HTTP errors, not something the SSE stream has to explain.
        """
        graph = app.state.graph
        config = {"configurable": {"thread_id": req.thread_id}}

        state = await graph.aget_state(config)
        if state.next != ("approval",):
            raise HTTPException(404, "No approval is pending for this chat")
        pending = state.values.get("pending_approval")
        if not pending or pending["id"] != req.id:
            raise HTTPException(409, "That approval card is stale (already resolved, or from an older request)")

        async def events():
            yield sse("start", {"trace_id": str(uuid.uuid4()), "thread_id": req.thread_id})
            started = time.perf_counter()
            async for event in stream_run(
                graph, Command(resume={"id": req.id, "approve": req.approve}), config, started
            ):
                yield event

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.get("/api/threads")
    async def list_threads():
        """List every chat for the sidebar: [{thread_id, title, updated_at}], newest first.

        How: the checkpointer stores many snapshots per chat (one per graph step). We walk all
        of them, keep only the newest snapshot of each chat, and use that chat's first message
        you sent as its title (cut to 60 characters).

        Walking every snapshot is fine for one user's local chats. With Postgres (later) this
        becomes one SQL query.
        """
        latest: dict[str, dict] = {}
        async for cp in app.state.checkpointer.alist(None):
            thread_id = cp.config["configurable"]["thread_id"]
            ts = cp.checkpoint["ts"]  # when this snapshot was saved (ISO-8601 text, so it sorts correctly)
            if thread_id in latest and latest[thread_id]["updated_at"] >= ts:
                continue  # already have a newer snapshot of this chat
            messages = cp.checkpoint["channel_values"].get("messages", [])
            first = next((text_of(m) for m in messages if m.type == "human"), "New chat")
            latest[thread_id] = {"thread_id": thread_id, "title": first[:60], "updated_at": ts}
        return sorted(latest.values(), key=lambda t: t["updated_at"], reverse=True)

    @app.get("/api/agents")
    async def get_agents():
        """The team and their tools, for the sidebar's Team panel (X3): `agents.capabilities.team`
        built from the same `WORKERS` registry and tool registry the graph itself runs on, so this can
        never show a worker or a tool the app doesn't really have. See docs/contracts.md § 9."""
        return team(WORKERS, app.state.tools)

    @app.get("/api/threads/{thread_id}")
    async def get_thread(thread_id: str):
        """Return one chat's messages as [{role, content, sources}], or 404 if unknown.

        `aget_state` asks the checkpointer for this chat's latest saved state. `sources` is the list
        rag_agent attached to its answers ([] for everything else).
        """
        state = await app.state.graph.aget_state({"configurable": {"thread_id": thread_id}})
        messages = state.values.get("messages", [])
        if not messages:
            raise HTTPException(404, "No chat with that ID")
        return {
            "thread_id": thread_id,
            "messages": [
                {
                    "role": "user" if m.type == "human" else "assistant",
                    "content": text_of(m),
                    "sources": m.additional_kwargs.get("sources", []),
                }
                for m in messages
            ],
        }

    return app


# The app object uvicorn serves (`uvicorn artlab.api:app`): real model, SQLite and Chroma, from .env,
# plus the injection classifier if it's been downloaded (None, i.e. off, otherwise).
app = create_app(classifier=load_classifier())
