"""
agents/respond.py — node 3a of the graph: Arty answering directly.

Runs when the supervisor routes a message to itself instead of a worker: small talk, general
knowledge, writing or brainstorming help. It just asks the model, with no tools and no knowledge-base
lookup — that's what tells it apart from rag_agent.
"""

import time

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langgraph.config import get_stream_writer

from artlab.agents.common import ms_since, tokens_used
from artlab.agents.state import ChatState
from artlab.model import cost_usd, model_name
from artlab.tools.registry import ToolRegistry

RESPOND_PROMPT = """You are Arty, Art Lab's assistant for a YouTube creator's studio: a cheerful little retro computer
who loves art and good ideas. Be warm, practical and concise."""


def make_node(model: BaseChatModel, tools: ToolRegistry):
    """Build the `respond` node, closing over the model and the tool registry.

    `tools` used to be unused here (this node never calls a tool itself). X3 uses it to describe every
    OTHER worker's tools, so Arty can answer "what can you do" from the same data the UI's Team panel
    shows (docs/contracts.md § 9) — built once here, not per message, the same way rag_agent builds its
    `rephraser` once.

    Lazy import, not a top-level one: `agents/workers.py` imports this module (`respond.make_node` is
    one of its `WorkerSpec`s) while it's still building its `WORKERS` tuple, so a module-level
    `from artlab.agents.workers import WORKERS` here would try to read `WORKERS` before workers.py has
    finished defining it — a circular import. Importing it inside this function instead is safe: by
    the time anything actually *calls* `make_node` (graph.py, building the graph), workers.py has
    already finished running.
    """
    from artlab.agents.capabilities import describe_team
    from artlab.agents.workers import WORKERS

    system_prompt = (
        f"{RESPOND_PROMPT}\n\n"
        "Here is the whole team and every tool each one can use:\n"
        f"{describe_team(WORKERS, tools)}\n\n"
        'Use this list, and only this list, to answer questions like "what can you do", "what tools do '
        'you have" or "what does <agent> do" - never invent a tool or a worker that isn\'t in it.'
    )

    async def respond(state: ChatState):
        """Node 3a — answer directly, from the conversation so far.

        Steps:
          1. Send the system prompt (the respond prompt plus the team text above) and the whole chat
             history to the model.
          2. Write a trace line: model name, tokens in/out, time taken.
          3. Return the reply (appended to messages), its cost, and `answered_by` so the supervisor finishes.

        Streaming note: we call `ainvoke` (wait for the full reply), yet the browser sees the answer
        appear word by word. api.py runs the graph with LangGraph's "messages" stream mode, which
        captures every token a model produces inside any node and streams it out as it arrives.
        """
        start = time.perf_counter()
        write = get_stream_writer()

        reply = await model.ainvoke([SystemMessage(system_prompt), *state["messages"]])

        tokens_in, tokens_out = tokens_used(reply)
        write({
            "stage": "arty", "status": "ok", "detail": f"answer · {model_name(model)} · {tokens_in} in / {tokens_out} out",
            "ms": ms_since(start), "input_tokens": tokens_in, "output_tokens": tokens_out,
        })
        return {"messages": [reply], "spent_usd": cost_usd(tokens_in, tokens_out), "answered_by": "respond"}

    return respond
