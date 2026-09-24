"""
graph.py — the agent graph: the steps one chat message goes through, as a LangGraph.

What LangGraph is, in one paragraph: you describe your program as *nodes* (async functions)
connected by *edges* (which node runs next). Every node receives the shared *state* (here: the
chat's messages, what it has spent, and this turn's routing notes), returns the parts it wants to
change, and LangGraph merges those changes in. A *checkpointer* saves the state after every node, so
a chat can be resumed later, even after a server restart.

The graph today (Phase 2):

    START ──▶ guard ──blocked──────────────────────────────────────────▶ END   (refusal, no model call)
                │
               pass
                ▼
            supervisor ──"respond"────▶ respond ─────┐   answers directly (small talk, general help)
            (main agent)                              │
                │  ▲    ──"rag_agent"──▶ rag_agent ───┤   searches the knowledge base, answers with citations
                │  └──────── a worker answered ◀──────┘
                └── finish ─────────────────────────────────────────────▶ END

The supervisor is the "main agent", and it has a name: **Arty** (a little retro computer with a face and a beret in the web UI,
frontend/src/components/Arty.tsx). Arty decides who answers each message — himself, via the `respond`
node, or rag_agent — and workers report back to him. Node names stay technical (`supervisor`,
`respond`); the trace panel shows both as "arty". Phase 3 adds more workers and a step limit for when
workers start handing work to each other.

How the trace panel gets its lines: each node calls `get_stream_writer()` and writes a small dict,
e.g. {"stage": "guard", "status": "ok", "detail": "...", "ms": 1}. LangGraph delivers those dicts on
its "custom" stream; api.py forwards them to the browser as `trace` events.

Where the nodes themselves live: each one is now its own module under `agents/` (guard.py,
supervisor.py, respond.py, rag_agent.py), so that different tickets can work on different nodes in
parallel without touching a shared file. This module only draws the graph's shape: it builds each
node from its module's `make_node` factory and wires them together with the same edges as before.
"""

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, StateGraph

from artlab.agents import guard, rag_agent, respond, supervisor
from artlab.agents.state import ChatState
from artlab.tools.registry import ToolRegistry


def build_graph(model: BaseChatModel, checkpointer: BaseCheckpointSaver, tools: ToolRegistry):
    """Assemble and compile the chat graph.

    Args:
        model:        the chat model every node uses. The real app passes Claude; tests pass the fake.
        checkpointer: where state is saved between steps and between messages. The real app uses
                      SQLite (data/artlab.db); tests use an in-memory one.
        tools:        the tool registry (tools/catalog.py). rag_agent searches through it.

    Returns:
        A compiled graph. api.py calls `graph.astream(...)` on it once per message.

    Why a function instead of a module-level graph: the model, storage and tools are *passed in*, so
    tests can swap them for free, fast versions. Each node is built by its module's `make_node`
    factory, which closes over the arguments that node needs (the same closures the nodes used when
    they lived inline here).
    """
    # Wire the nodes together. `guard` and `supervisor` have no fixed outgoing edges: their Commands'
    # `goto` decides. Workers always report back to the supervisor. Compiling with a checkpointer turns
    # on saving: each run is saved under the `thread_id` api.py passes in (one thread = one chat).
    return (
        StateGraph(ChatState)
        .add_node("guard", guard.make_node())
        .add_node("supervisor", supervisor.make_node(model))
        .add_node("respond", respond.make_node(model, tools))
        .add_node("rag_agent", rag_agent.make_node(model, tools))
        .add_edge(START, "guard")
        .add_edge("respond", "supervisor")
        .add_edge("rag_agent", "supervisor")
        .compile(checkpointer=checkpointer)
    )
