"""
agents/state.py — the shape of one chat, as the graph passes it from node to node.

Every node in the graph (guard, supervisor, respond, rag_agent) receives and returns pieces of
`ChatState`; LangGraph merges the pieces in using each field's *reducer* (see the class docstring).
The checkpointer saves this state after every node, so a chat can be resumed later.
"""

import operator
from typing import Annotated

from langgraph.graph import MessagesState


def add_unique(existing: list[str], new: list[str]) -> list[str]:
    """The reducer for `taint_sources`: append names from `new` that aren't already in `existing`,
    keeping the order they were first seen in. Two different turns (or two tool calls in one turn) can
    taint a chat for the same reason — e.g. `search_knowledge` twice — and the approval card should
    list each source once, not once per occurrence.

    Example: add_unique(["fetch_comments"], ["fetch_comments", "search_knowledge"])
             == ["fetch_comments", "search_knowledge"]
    """
    result = list(existing)
    for name in new:
        if name not in result:
            result.append(name)
    return result


class ChatState(MessagesState):
    """Everything the graph remembers about one chat. Saved by the checkpointer after every node.

    messages         (inherited from MessagesState) the conversation so far. Its *reducer* is
                     LangGraph's `add_messages`: when a node returns {"messages": [new]}, `new` is
                     appended rather than replacing the list. It also understands `RemoveMessage(id=...)`,
                     which deletes one.
    spent_usd        total model cost of this chat. Its reducer is `operator.add`: when a node returns
                     {"spent_usd": 0.002}, that amount is *added*. The input guard enforces the budget on it.
    task             this turn's question, rewritten by the supervisor to stand alone, handed to the worker.
    answered_by      which worker answered this turn ("" until one has). Tells the supervisor to finish.
    steps            how many times the supervisor has sent work to a worker this turn (the guard resets
                     it to 0). The circuit breaker stops the turn at MAX_STEPS. Counted once per dispatch.
    handoff          a worker's request that another worker continue ("" = no). The guard resets it.
    tainted          True once untrusted content (tool results, documents…) has entered this chat. Its
                     reducer is `operator.or_`: any node returning {"tainted": True} sets it, and False
                     never clears it, so the whole chat stays tainted. A tainted chat's data-changing
                     tools need your Approve click (Phase 6) instead of running outright.
    pending_approval a data-changing tool call waiting for your click, or `None` when nothing waits:
                     `{id, agent, tool, args, tainted, taint_sources}`. Its reducer is the plain
                     "last write wins" LangGraph gives every field with no `Annotated` reducer — a
                     worker sets it when a tool needs approval, and the approval node clears it back to
                     `None` once you've clicked. docs/contracts.md § 10.
    taint_sources    where the chat's `tainted` flag came from, e.g. "fetch_comments", "search_knowledge",
                     "guard classifier" — shown on the approval card so you know what to check before
                     approving. Its reducer is `add_unique` (above): every source is added once, in the
                     order it first appeared, and (like `tainted`) it's never removed.

    The full contract for these fields: docs/contracts.md §§ 1, 10.
    """

    spent_usd: Annotated[float, operator.add]
    task: str
    answered_by: str
    steps: int
    handoff: str
    tainted: Annotated[bool, operator.or_]
    pending_approval: dict | None
    taint_sources: Annotated[list[str], add_unique]
