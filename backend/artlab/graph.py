"""The agent graph. MVP shape: guard -> llm.

Every node reports what it did through LangGraph's custom stream (`get_stream_writer`).
Those reports are the lines you see in the web app's trace panel.
"""

import time
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, RemoveMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command

from artlab.model import model_name

MAX_INPUT_CHARS = 4000
SYSTEM_PROMPT = "You are Art Lab, a helpful assistant for a YouTube creator. Be concise."


def _ms_since(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


def build_graph(model: BaseChatModel, checkpointer: BaseCheckpointSaver):
    async def guard(state: MessagesState) -> Command[Literal["llm", "__end__"]]:
        start = time.perf_counter()
        write = get_stream_writer()
        message = state["messages"][-1]
        size = len(message.content)

        if size > MAX_INPUT_CHARS:
            reason = f"message is {size} chars; the limit is {MAX_INPUT_CHARS}"
            write({"stage": "guard", "status": "blocked", "detail": reason, "ms": _ms_since(start)})
            # Drop the blocked message so it never reaches the model on a later turn either.
            refusal = AIMessage(f"Blocked by the input guard: {reason}.")
            return Command(update={"messages": [RemoveMessage(id=message.id), refusal]}, goto=END)

        write({"stage": "guard", "status": "ok", "detail": f"pass · {size} chars", "ms": _ms_since(start)})
        return Command(goto="llm")

    async def llm(state: MessagesState):
        start = time.perf_counter()
        write = get_stream_writer()
        reply = await model.ainvoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])

        usage = reply.usage_metadata or {}
        tokens_in, tokens_out = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        write({
            "stage": "llm",
            "status": "ok",
            "detail": f"{model_name(model)} · {tokens_in} in / {tokens_out} out",
            "ms": _ms_since(start),
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
        })
        return {"messages": [reply]}

    return (
        StateGraph(MessagesState)
        .add_node("guard", guard)
        .add_node("llm", llm)
        .add_edge(START, "guard")
        .add_edge("llm", END)
        .compile(checkpointer=checkpointer)
    )
