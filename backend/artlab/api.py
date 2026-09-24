"""
api.py — the HTTP API the web page talks to. Built with FastAPI.

Endpoints:

    POST /api/chat                 send one message; the reply streams back as server-sent events
    GET  /api/threads              list all chats, newest first (the left sidebar)
    GET  /api/threads/{thread_id}  one chat's full history, with each answer's sources
    GET  /api/agents               the team and their tools, for the sidebar's Team panel (X3)

How POST /api/chat streams. The response is *server-sent events* (SSE): a long-lived HTTP
response made of small text blocks, each one looking like

    event: token
    data: {"text": "Hello"}
    <blank line>

The browser reads them one by one as they arrive (frontend/src/api.ts → streamChat). Five
event types exist, always in this order:

    start   → trace ID and chat ID                           (once)
    trace   → one line for the trace panel                   (once per graph step)
    token   → a piece of the answer text                     (many)
    done    → tokens used, cost, duration, the answer's sources   (once, at the end)
    error   → what went wrong                                (instead of done, if something failed)

Run the server:  cd backend && uv run uvicorn artlab.api:app --reload --port 8000
"""

import json
import time
import uuid
from contextlib import AsyncExitStack, asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel, Field

from artlab.agents.capabilities import team
from artlab.agents.common import text_of
from artlab.agents.workers import WORKERS
from artlab.config import DB_PATH, REPO_ROOT
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


def sse(event: str, data: dict) -> str:
    """Format one server-sent event: an `event:` line, a `data:` line (JSON), and a blank line."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def create_app(
    model: BaseChatModel | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    knowledge: KnowledgeBase | None = None,
    classifier: InjectionClassifier | None = None,
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
            tools = build_tools(knowledge or KnowledgeBase())
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
            """Yield the SSE events for this run: start, then trace/token as they happen, then done or error."""
            yield sse("start", {"trace_id": trace_id, "thread_id": thread_id})
            started = time.perf_counter()
            tokens_in = tokens_out = 0

            try:
                # Run the graph with two stream modes at once. Each item is (mode, chunk):
                #   "custom"   → a dict a node wrote with get_stream_writer()  → forward as `trace`
                #   "messages" → (message piece, metadata) as a model writes → forward as `token`
                # The checkpointer loads this chat's earlier messages first, so we only send the new one.
                async for mode, chunk in graph.astream(
                    {"messages": [HumanMessage(req.message)]}, config, stream_mode=["messages", "custom"]
                ):
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

            # The final answer is the last message; rag_agent attached its sources to it.
            state = await graph.aget_state(config)
            last = state.values["messages"][-1]
            yield sse("done", {
                "input_tokens": tokens_in,
                "output_tokens": tokens_out,
                "cost_usd": cost_usd(tokens_in, tokens_out),
                "ms": round((time.perf_counter() - started) * 1000),
                "sources": last.additional_kwargs.get("sources", []) if last.type == "ai" else [],
            })

        # StreamingResponse sends each string `events()` yields as soon as it's yielded.
        # "no-cache" stops proxies from buffering the stream.
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
