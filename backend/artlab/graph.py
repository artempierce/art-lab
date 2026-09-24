"""
graph.py — the agent graph: the steps one chat message goes through, as a LangGraph.

What LangGraph is, in one paragraph: you describe your program as *nodes* (async functions)
connected by *edges* (which node runs next). Every node receives the shared *state* (here: the
chat's messages, what it has spent, and this turn's routing notes), returns the parts it wants to
change, and LangGraph merges those changes in. A *checkpointer* saves the state after every node, so
a chat can be resumed later, even after a server restart.

The graph today (Phase 3 — supervisor v2):

    START ──▶ guard ──blocked──────────────────────────────────────────▶ END   (refusal, no model call)
                │
               pass
                ▼
            supervisor ──"respond"────▶ respond ─────┐   answers directly (small talk, general help)
       (main agent, Arty)  ──"rag_agent"─▶ rag_agent ─┤   searches the knowledge base, answers with citations
                │  ▲                                  │
                │  └──────────── a worker answered ◀──┘   (or asks to hand off to another worker)
                ├── finish ────────────────────────────────────────────▶ END
                └── step limit reached (5 dispatches) ──────────────────▶ END  (with a note in the answer)

The supervisor is the "main agent", and it has a name: **Arty** (a little retro computer with a face
and a beret in the web UI, frontend/src/components/Arty.tsx). Arty decides who answers each message,
and workers report back to him — either finishing, or asking (`handoff`) that another worker
continue. Node names stay technical (`supervisor`, `respond`); the trace panel shows both as "arty".

Who Arty can route to is no longer fixed here: `agents/workers.py` holds the worker registry
(`WORKERS`, a tuple of `WorkerSpec`), and `build_graph` turns each entry into a node plus an edge
back to the supervisor. Adding a worker is a one-file change (`workers.py`); this module and
`supervisor.py` build themselves from whatever registry they're given. A circuit breaker
(`supervisor.MAX_STEPS`) stops a handoff loop between workers after 5 dispatches, so the graph can't
run forever — see `agents/supervisor.py` and docs/contracts.md § 3.

How the trace panel gets its lines: each node calls `get_stream_writer()` and writes a small dict,
e.g. {"stage": "guard", "status": "ok", "detail": "...", "ms": 1}. LangGraph delivers those dicts on
its "custom" stream; api.py forwards them to the browser as `trace` events.

Where the nodes themselves live: each one is its own module under `agents/` (guard.py,
supervisor.py, respond.py, rag_agent.py), so that different tickets can work on different nodes in
parallel without touching a shared file. This module only draws the graph's shape: it builds each
node from its module's `make_node` factory (or, for workers, `WorkerSpec.make_node`) and wires them
together.
"""

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from artlab.agents import guard, supervisor
from artlab.agents.state import ChatState
from artlab.agents.workers import WORKERS, WorkerSpec
from artlab.tools.registry import ToolRegistry


def build_graph(
    model: BaseChatModel,
    checkpointer: BaseCheckpointSaver,
    tools: ToolRegistry,
    workers: tuple[WorkerSpec, ...] = WORKERS,
):
    """Assemble and compile the chat graph.

    Args:
        model:        the chat model every node uses. The real app passes Claude; tests pass the fake.
        checkpointer: where state is saved between steps and between messages. The real app uses
                      SQLite (data/artlab.db); tests use an in-memory one.
        tools:        the tool registry (tools/catalog.py). Workers search through it.
        workers:      the worker registry (agents/workers.py). Defaults to every real worker; tests
                      can pass their own stub `WorkerSpec`s instead.

    Returns:
        A compiled graph. api.py calls `graph.astream(...)` on it once per message.

    Why a function instead of a module-level graph: the model, storage, tools and workers are all
    *passed in*, so tests can swap them for free, fast versions. Each node is built by its module's
    `make_node` factory, which closes over the arguments that node needs (the same closures the
    nodes used when they lived inline here).
    """
    names = tuple(worker.name for worker in workers)

    # Wire the nodes together. `guard` and `supervisor` have no fixed outgoing edges: their Commands'
    # `goto` decides. `destinations` doesn't change that — it only tells LangGraph (for graph
    # drawings) which nodes the supervisor *can* jump to: any worker, or the end. Workers always
    # report back to the supervisor. Compiling with a checkpointer turns on saving: each run is saved
    # under the `thread_id` api.py passes in (one thread = one chat).
    graph = StateGraph(ChatState)
    graph.add_node("guard", guard.make_node())
    graph.add_node("supervisor", supervisor.make_node(model, workers), destinations=(*names, END))
    for worker in workers:
        graph.add_node(worker.name, worker.make_node(model, tools))
        graph.add_edge(worker.name, "supervisor")
    graph.add_edge(START, "guard")
    return graph.compile(checkpointer=checkpointer)
