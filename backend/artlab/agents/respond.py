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
    """Build the `respond` node, closing over the model. `tools` is unused today — kept so every
    worker node has the same factory signature (`make_node(model, tools)`)."""

    async def respond(state: ChatState):
        """Node 3a — answer directly, from the conversation so far.

        Steps:
          1. Send the respond prompt plus the whole chat history to the model.
          2. Write a trace line: model name, tokens in/out, time taken.
          3. Return the reply (appended to messages), its cost, and `answered_by` so the supervisor finishes.

        Streaming note: we call `ainvoke` (wait for the full reply), yet the browser sees the answer
        appear word by word. api.py runs the graph with LangGraph's "messages" stream mode, which
        captures every token a model produces inside any node and streams it out as it arrives.
        """
        start = time.perf_counter()
        write = get_stream_writer()

        reply = await model.ainvoke([SystemMessage(RESPOND_PROMPT), *state["messages"]])

        tokens_in, tokens_out = tokens_used(reply)
        write({
            "stage": "arty", "status": "ok", "detail": f"answer · {model_name(model)} · {tokens_in} in / {tokens_out} out",
            "ms": ms_since(start), "input_tokens": tokens_in, "output_tokens": tokens_out,
        })
        return {"messages": [reply], "spent_usd": cost_usd(tokens_in, tokens_out), "answered_by": "respond"}

    return respond
