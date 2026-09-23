"""HTTP API: stream a chat turn as server-sent events, list chats, load a chat's history.

Run it:  uv run uvicorn artlab.api:app --reload --port 8000
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
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel, Field

from artlab.graph import build_graph
from artlab.model import cost_usd, make_model

REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = REPO_ROOT / "data" / "artlab.db"

load_dotenv(REPO_ROOT / ".env")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    thread_id: str | None = None


def text_of(message: BaseMessage) -> str:
    """Message text, whether content is a plain string or a list of content blocks."""
    if isinstance(message.content, str):
        return message.content
    return "".join(b.get("text", "") for b in message.content if isinstance(b, dict))


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def create_app(model: BaseChatModel | None = None, checkpointer: BaseCheckpointSaver | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with AsyncExitStack() as stack:
            saver = checkpointer
            if saver is None:
                DB_PATH.parent.mkdir(exist_ok=True)
                saver = await stack.enter_async_context(AsyncSqliteSaver.from_conn_string(str(DB_PATH)))
            app.state.checkpointer = saver
            app.state.graph = build_graph(model or make_model(), saver)
            yield

    app = FastAPI(title="Art Lab", lifespan=lifespan)

    @app.post("/api/chat")
    async def chat(req: ChatRequest):
        graph = app.state.graph
        trace_id = str(uuid.uuid4())
        thread_id = req.thread_id or str(uuid.uuid4())
        # run_id makes the LangSmith trace ID equal to the one the trace panel shows.
        config = {"configurable": {"thread_id": thread_id}, "run_id": trace_id, "run_name": "art-lab chat"}

        async def events():
            yield sse("start", {"trace_id": trace_id, "thread_id": thread_id})
            started = time.perf_counter()
            tokens_in = tokens_out = 0
            try:
                async for mode, chunk in graph.astream(
                    {"messages": [HumanMessage(req.message)]}, config, stream_mode=["messages", "custom"]
                ):
                    if mode == "custom":
                        tokens_in += chunk.get("input_tokens", 0)
                        tokens_out += chunk.get("output_tokens", 0)
                        yield sse("trace", chunk)
                    else:
                        message, _meta = chunk
                        if message.type in ("ai", "AIMessageChunk") and (text := text_of(message)):
                            yield sse("token", {"text": text})
            except Exception as exc:
                yield sse("error", {"message": f"{type(exc).__name__}: {exc}"})
                return
            yield sse("done", {
                "input_tokens": tokens_in,
                "output_tokens": tokens_out,
                "cost_usd": cost_usd(tokens_in, tokens_out),
                "ms": round((time.perf_counter() - started) * 1000),
            })

        return StreamingResponse(
            events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )

    @app.get("/api/threads")
    async def list_threads():
        # Latest checkpoint per chat. Scans every checkpoint: fine for one user, revisit with Postgres.
        latest: dict[str, dict] = {}
        async for cp in app.state.checkpointer.alist(None):
            thread_id = cp.config["configurable"]["thread_id"]
            ts = cp.checkpoint["ts"]
            if thread_id in latest and latest[thread_id]["updated_at"] >= ts:
                continue
            messages = cp.checkpoint["channel_values"].get("messages", [])
            first = next((text_of(m) for m in messages if m.type == "human"), "New chat")
            latest[thread_id] = {"thread_id": thread_id, "title": first[:60], "updated_at": ts}
        return sorted(latest.values(), key=lambda t: t["updated_at"], reverse=True)

    @app.get("/api/threads/{thread_id}")
    async def get_thread(thread_id: str):
        state = await app.state.graph.aget_state({"configurable": {"thread_id": thread_id}})
        messages = state.values.get("messages", [])
        if not messages:
            raise HTTPException(404, "No chat with that ID")
        return {
            "thread_id": thread_id,
            "messages": [{"role": "user" if m.type == "human" else "assistant", "content": text_of(m)} for m in messages],
        }

    return app


app = create_app()
