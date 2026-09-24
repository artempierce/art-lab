"""
agents/state.py — the shape of one chat, as the graph passes it from node to node.

Every node in the graph (guard, supervisor, respond, rag_agent) receives and returns pieces of
`ChatState`; LangGraph merges the pieces in using each field's *reducer* (see the class docstring).
The checkpointer saves this state after every node, so a chat can be resumed later.
"""

import operator
from typing import Annotated

from langgraph.graph import MessagesState


class ChatState(MessagesState):
    """Everything the graph remembers about one chat. Saved by the checkpointer after every node.

    messages     (inherited from MessagesState) the conversation so far. Its *reducer* is LangGraph's
                 `add_messages`: when a node returns {"messages": [new]}, `new` is appended rather than
                 replacing the list. It also understands `RemoveMessage(id=...)`, which deletes one.
    spent_usd    total model cost of this chat. Its reducer is `operator.add`: when a node returns
                 {"spent_usd": 0.002}, that amount is *added*. The input guard enforces the budget on it.
    task         this turn's question, rewritten by the supervisor to stand alone, handed to the worker.
    answered_by  which worker answered this turn ("" until one has). Tells the supervisor to finish.
    """

    spent_usd: Annotated[float, operator.add]
    task: str
    answered_by: str
