"""
graph.py — the agent graph: the steps one chat message goes through, as a LangGraph.

What LangGraph is, in one paragraph: you describe your program as *nodes* (async functions)
connected by *edges* (which node runs next). Every node receives the shared *state* (here: the
chat's messages and how much it has spent), returns the parts it wants to change, and LangGraph
merges those changes in. A *checkpointer* saves the state after every node, so a chat can be
resumed later, even after a server restart.

The graph today (Phase 1):

    START ──▶ guard ──pass────▶ llm ──▶ END
                 └────blocked─────────▶ END   (reply with a refusal; the model is never called)

Later phases add nodes between guard and llm: a supervisor that picks an agent (Phase 2),
worker agents with tools (Phases 3–5), and so on. See docs/design.html → Build plan.

How the trace panel gets its lines: each node calls `get_stream_writer()` and writes a small
dict, e.g. {"stage": "guard", "status": "ok", "detail": "...", "ms": 1}. LangGraph delivers
those dicts on its "custom" stream; api.py forwards them to the browser as `trace` events.
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

# Sent to the model on every call, but never stored in the chat history. Keeping it out of
# state means we can change it at any time without rewriting old chats.
SYSTEM_PROMPT = "You are Art Lab, a helpful assistant for a YouTube creator. Be concise."


class ChatState(MessagesState):
    """Everything the graph remembers about one chat. Saved by the checkpointer after every node.

    messages   (inherited from MessagesState) the conversation so far. Its *reducer* is
               LangGraph's `add_messages`: when a node returns {"messages": [new]}, `new` is
               appended rather than replacing the list. It also understands
               `RemoveMessage(id=...)`, which deletes a message.
    spent_usd  total model cost of this chat so far. Its reducer is `operator.add`: when the llm
               node returns {"spent_usd": 0.002}, that amount is *added* to the running total.
               The input guard reads it to enforce the per-chat budget.
    """

    spent_usd: Annotated[float, operator.add]


def _ms_since(start: float) -> int:
    """Milliseconds elapsed since `start` (a `time.perf_counter()` reading). Used for trace timings."""
    return round((time.perf_counter() - start) * 1000)


def build_graph(model: BaseChatModel, checkpointer: BaseCheckpointSaver):
    """Assemble and compile the chat graph.

    Args:
        model:        the chat model the `llm` node calls. The real app passes Claude; tests
                      pass a fake one (see model.py).
        checkpointer: where state is saved between steps and between messages. The real app
                      uses SQLite (data/artlab.db); tests use an in-memory one.

    Returns:
        A compiled graph. api.py calls `graph.astream(...)` on it once per message.

    Why a function instead of a module-level graph: the model and the storage are *passed in*,
    so tests can swap both for free, fast versions without touching this file. The node
    functions are defined inside so they can use `model` directly (they are closures over it).
    """

    async def guard(state: ChatState) -> Command[Literal["llm", "__end__"]]:
        """Node 1 — run the input guard on the newest message, then either continue or stop.

        Returns a `Command`, which does two jobs in one return value: `update` changes the
        state, and `goto` picks the next node. That replaces a separate "conditional edge"
        function. The `Literal["llm", "__end__"]` type tells LangGraph (and graph drawings)
        which nodes this one can jump to.
        """
        start = time.perf_counter()
        write = get_stream_writer()  # sends trace lines to the "custom" stream

        # The message you just sent is always the last one in the history.
        message = state["messages"][-1]
        result = check_input(message.content, state.get("spent_usd", 0.0))

        if result.rule:
            # Blocked. Tell the trace panel which rule fired and why.
            write({"stage": "guard", "status": "blocked", "detail": f"{result.rule}: {result.reason}", "ms": _ms_since(start)})
            # Delete the blocked message from history (RemoveMessage) so it can never reach the
            # model on a later turn either, add a refusal as the assistant's reply, and end the run.
            refusal = AIMessage(f"Blocked by the input guard ({result.rule}): {result.reason}.")
            return Command(update={"messages": [RemoveMessage(id=message.id), refusal]}, goto=END)

        # Passed. Report it (with budget used) and hand over to the llm node.
        write({"stage": "guard", "status": "ok", "detail": result.reason, "ms": _ms_since(start)})
        return Command(goto="llm")

    async def llm(state: ChatState):
        """Node 2 — ask the model for a reply to the conversation so far.

        Steps:
          1. Send the system prompt plus the whole chat history to the model.
          2. Read how many tokens that used (the model reports it on the reply).
          3. Write a trace line: model name, tokens in/out, time taken.
          4. Return the reply (appended to messages) and its cost (added to spent_usd).

        Streaming note: we call `ainvoke` (wait for the full reply), yet the browser still sees
        the answer appear word by word. That's because api.py runs the graph with LangGraph's
        "messages" stream mode, which captures every token the model produces inside any node
        and streams it out as it arrives. The node code doesn't need to know.
        """
        start = time.perf_counter()
        write = get_stream_writer()

        # 1. System prompt first, then the conversation. The system prompt isn't in state; see SYSTEM_PROMPT.
        reply = await model.ainvoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])

        # 2. Real Claude fills `usage_metadata`; the fake model leaves it empty, which counts as 0 tokens.
        usage = reply.usage_metadata or {}
        tokens_in, tokens_out = usage.get("input_tokens", 0), usage.get("output_tokens", 0)

        # 3. The token counts ride along in the trace event so api.py can total the run's cost.
        write({
            "stage": "llm",
            "status": "ok",
            "detail": f"{model_name(model)} · {tokens_in} in / {tokens_out} out",
            "ms": _ms_since(start),
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
        })

        # 4. The reducers do the merging: messages are appended, spent_usd is added to the total.
        return {"messages": [reply], "spent_usd": cost_usd(tokens_in, tokens_out)}

    # Wire the nodes together. `guard` has no outgoing edge here because its Command's `goto`
    # decides where to go. Compiling with a checkpointer turns on saving: each run is saved
    # under the `thread_id` that api.py passes in the config (one thread = one chat).
    return (
        StateGraph(ChatState)
        .add_node("guard", guard)
        .add_node("llm", llm)
        .add_edge(START, "guard")
        .add_edge("llm", END)
        .compile(checkpointer=checkpointer)
    )
