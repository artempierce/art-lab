"""The agent graph. MVP shape: guard -> llm.

Every node reports what it did through LangGraph's custom stream (`get_stream_writer`).
Those reports are the lines you see in the web app's trace panel.
"""

import operator
import time
from typing import Annotated, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, RemoveMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command

from artlab.guards.input import check_input
from artlab.model import cost_usd, model_name

SYSTEM_PROMPT = "You are Art Lab, a helpful assistant for a YouTube creator. Be concise."


class ChatState(MessagesState):
    spent_usd: Annotated[float, operator.add]  # this chat's total model cost; each llm call adds to it


def _ms_since(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


def build_graph(model: BaseChatModel, checkpointer: BaseCheckpointSaver):
    async def guard(state: ChatState) -> Command[Literal["llm", "__end__"]]:
        start = time.perf_counter()
        write = get_stream_writer()
        message = state["messages"][-1]
        result = check_input(message.content, state.get("spent_usd", 0.0))

        if result.rule:
            write({"stage": "guard", "status": "blocked", "detail": f"{result.rule}: {result.reason}", "ms": _ms_since(start)})
            # Drop the blocked message so it never reaches the model on a later turn either.
            refusal = AIMessage(f"Blocked by the input guard ({result.rule}): {result.reason}.")
            return Command(update={"messages": [RemoveMessage(id=message.id), refusal]}, goto=END)

        write({"stage": "guard", "status": "ok", "detail": result.reason, "ms": _ms_since(start)})
        return Command(goto="llm")

    async def llm(state: ChatState):
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
        return {"messages": [reply], "spent_usd": cost_usd(tokens_in, tokens_out)}

    return (
        StateGraph(ChatState)
        .add_node("guard", guard)
        .add_node("llm", llm)
        .add_edge(START, "guard")
        .add_edge("llm", END)
        .compile(checkpointer=checkpointer)
    )
