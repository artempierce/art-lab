"""
agents/workers.py — the worker registry: every node the supervisor can route to (T1, Phase 3).

This is the ONLY place a route is added. A `WorkerSpec` names a worker, gives the one-line
description the supervisor's prompt shows so the model knows when to pick it, and holds the node
factory that builds it. `graph.py` turns each entry into a graph node and wires it back to the
supervisor; `supervisor.py` builds its `RouteDecision` schema (the "next" field's allowed values)
from the same list. Add a new worker here — a Phase 5 (T10) job — and both pick it up with no
other change. Contract: docs/contracts.md § 3.
"""

from dataclasses import dataclass
from typing import Awaitable, Callable

from langchain_core.language_models import BaseChatModel

from artlab.agents import rag_agent, respond
from artlab.agents.state import ChatState
from artlab.tools.registry import ToolRegistry

# A worker node: takes the chat state, returns the state update it wants to make (docs/contracts.md § 2).
Node = Callable[[ChatState], Awaitable[dict]]


@dataclass(frozen=True)
class WorkerSpec:
    """One worker the supervisor can route to.

    name         the node name, and the route value the supervisor's `RouteDecision.next` picks
                 (e.g. "rag_agent")
    description  one line the supervisor's prompt shows, explaining when to route here
    make_node    the worker's node factory: make_node(model, tools) -> Node (docs/contracts.md § 2)
    """

    name: str
    description: str
    make_node: Callable[[BaseChatModel, ToolRegistry], Node]


# Every worker the supervisor can route to today. Descriptions are the same wording the old, fixed
# SUPERVISOR_PROMPT used, so routing behaviour doesn't change just because the prompt is now built
# from this list. Phase 5 (T10) appends new workers here — nowhere else.
WORKERS: tuple[WorkerSpec, ...] = (
    WorkerSpec(
        name="respond",
        description=(
            "everything else, which you answer yourself - small talk, general knowledge, writing or "
            "brainstorming help."
        ),
        make_node=respond.make_node,
    ),
    WorkerSpec(
        name="rag_agent",
        description=(
            "questions about the studio's own policies, rules, processes, checklists, requirements or "
            "project documents - anything that needs the studio's knowledge base."
        ),
        make_node=rag_agent.make_node,
    ),
)
