"""
agents/guard.py — node 1 of the graph: the input guard.

Runs first on every turn, before any model sees the message. It checks the newest message against
the input guard rules (artlab.guards.input) and either blocks the turn with a fixed refusal, or lets
it through to the supervisor. Blocking here means a bad message never costs a model call.
"""

import time
from typing import Literal

from langchain_core.messages import AIMessage, RemoveMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END
from langgraph.types import Command

from artlab.agents.common import ms_since
from artlab.agents.state import ChatState
from artlab.guards.input import check_input


def make_node():
    """Build the `guard` node. No dependencies to close over (the guard doesn't call a model)."""

    async def guard(state: ChatState) -> Command[Literal["supervisor", "__end__"]]:
        """Node 1 — run the input guard on the newest message, then either continue or stop.

        Returns a `Command`, which does two jobs in one return value: `update` changes the state, and
        `goto` picks the next node. The `Literal[...]` type tells LangGraph (and graph drawings) which
        nodes this one can jump to.
        """
        start = time.perf_counter()
        write = get_stream_writer()  # sends trace lines to the "custom" stream

        # The message you just sent is always the last one in the history.
        message = state["messages"][-1]
        result = check_input(message.content, state.get("spent_usd", 0.0))

        if result.rule:
            # Blocked. Tell the trace panel which rule fired and why.
            write({"stage": "guard", "status": "blocked", "detail": f"{result.rule}: {result.reason}", "ms": ms_since(start)})
            # Delete the blocked message from history (RemoveMessage) so it can never reach the model
            # on a later turn either, add a refusal as the assistant's reply, and end the run.
            refusal = AIMessage(f"Blocked by the input guard ({result.rule}): {result.reason}.")
            return Command(update={"messages": [RemoveMessage(id=message.id), refusal], "answered_by": ""}, goto=END)

        # Passed. Report it (with budget used), clear last turn's routing notes and step count, and hand over.
        write({"stage": "guard", "status": "ok", "detail": result.reason, "ms": ms_since(start)})
        return Command(update={"task": "", "answered_by": "", "steps": 0, "handoff": ""}, goto="supervisor")

    return guard
