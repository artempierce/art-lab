"""
graph.py — the agent graph: the steps one chat message goes through, as a LangGraph.

What LangGraph is, in one paragraph: you describe your program as *nodes* (async functions)
connected by *edges* (which node runs next). Every node receives the shared *state* (here: the
chat's messages, what it has spent, and this turn's routing notes), returns the parts it wants to
change, and LangGraph merges those changes in. A *checkpointer* saves the state after every node, so
a chat can be resumed later, even after a server restart.

What `interrupt()` and `Command(resume=...)` are (first used here, by the `approval` node, Phase 6 —
docs/contracts.md § 10): `interrupt(payload)` pauses the *node* it's called in and saves the graph's
state right there — `payload` is whatever you want the caller to see while it's waiting (here: the
tool the owner is being asked to approve). `astream`/`ainvoke` simply stop, as if the run had ended.
Later, calling `astream(Command(resume=some_value), config, ...)` on the *same thread* picks the run
back up — but by **re-running the paused node from its very first line**, not from the `interrupt()`
call onward. This time, though, `interrupt(payload)` doesn't pause again: it returns `some_value`
straight away, and the node continues from there. That's why the `approval` node below reads
`pending_approval` and calls `interrupt()` as its very first action, before anything else — a second
run of the same node must never repeat a side effect (like running the tool) that only belongs after
the resume.

The graph today (Phase 10 — output guard and caps):

    START ──▶ guard ──blocked──────────────────────────────────────────▶ END   (refusal, no model call)
                │
               pass
                ▼
            summarize   (only once the chat passes 30 messages: folds the old ones into one summary)
                ▼
             recall ──▶ supervisor ──"respond"────▶ respond ─────┐   answers directly (small talk, general help)
     (loads state["memory"]) (main agent, Arty)  ──"rag_agent"─▶ rag_agent ─┤   searches the knowledge base, cites sources
                                 │  ▲                                  │
                                 │  └──────────── a worker answered ◀──┘   (or asks to hand off to another worker)
                                 ├── a tool needs your click ──▶ approval ──▶ END  (interrupt(); resumes via /api/chat/resume)
                                 ├── finish ──▶ remember ──▶ output_guard ──▶ END   (saves facts from YOUR message, then
                                 │                                                  checks the answer, docs/contracts.md §§ 11, 14)
                                 └── step limit reached (5 dispatches) ──▶ output_guard ──▶ END  (with a note in the answer)

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

Where the nodes themselves live: each one is its own module under `agents/` (guard.py, recall.py,
supervisor.py, respond.py, rag_agent.py, remember.py…), so that different tickets can work on
different nodes in parallel without touching a shared file. This module only draws the graph's shape:
it builds each node from its module's `make_node` factory (or, for workers, `WorkerSpec.make_node`)
and wires them together.

`recall` and `remember` (Phase 7, docs/contracts.md § 11) are the owner's long-term memory: `recall`
loads your saved facts into state before the supervisor decides anything, and `remember` — reached
only once a worker has answered and the turn is otherwise done — looks at the message you just typed
for anything durable worth saving. Neither is a worker: like `approval`, they're never in `WORKERS`,
so the model can't route to them, only the graph's own wiring sends a turn there.

`output_guard` (Phase 10, docs/contracts.md § 14) is the last stop before END on every path except
`approval`'s own — both the ordinary "finish" branch (after `remember`) and the step-limit stop paths
in `agents/supervisor.py` land here. It checks every answer this turn added, redacts a leaked prompt or
a stray internal tag, and rewrites the saved message so the chat history is also clean, not just the
live view api.py already retracted with a `replace` event. Like `recall`/`remember`/`approval`, it's
never a worker, and it needs no model — see `guards/output.py`.
"""

import time

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from artlab.agents import guard, recall, remember, summarize, supervisor
from artlab.agents.common import ms_since
from artlab.agents.state import ChatState
from artlab.agents.workers import WORKERS, WorkerSpec
from artlab.guards import output
from artlab.guards.classifier import InjectionClassifier
from artlab.memory.store import MemoryStore
from artlab.tools.registry import ToolRegistry


def _make_approval_node(tools: ToolRegistry):
    """Build the `approval` node (Phase 6, docs/contracts.md § 10) — not a worker: it's never in
    `WORKERS`, so it's never a route the supervisor's `RouteDecision` can pick; only the supervisor's
    own code sends a turn here, when `pending_approval` is set.

    Closes over `tools` so it can run the approved call for real (`tools.run_approved` — the one path
    a mutating tool can actually execute, per tools/registry.py).
    """

    async def approval(state: ChatState) -> dict:
        """Pause for your Approve/Reject click, then run (or skip) the tool exactly as recorded.

        Why this is its own node, not a pause inside a worker's tool loop: see the file header for
        what `interrupt()`/`Command(resume=...)` do. Pausing inside a worker's loop would mean a
        resume re-runs *that* loop from the top, calling the model again — for possibly different
        arguments than the ones on the card you approved. This node's whole job is to read the
        request a worker already recorded (`state["pending_approval"]`) and act on it, so re-running
        it from the top on resume is harmless: nothing happens before the `interrupt()` call below.

        Steps:
          1. `interrupt(payload)` — first thing this function does. The first time through, this
             pauses the run right here (LangGraph saves the state; api.py notices via
             `aget_state(config).next` and sends the browser an `approval` event with the same
             payload). On resume, it returns instead of pausing — see the file header.
          2. `decision` is whatever `Command(resume=...)` was called with: `{"id", "approve"}`. A
             resume whose `id` doesn't match this request is treated as a reject — a stale card from
             an old turn, or (api.py) a new message auto-cancelling this one.
          3. Approve → run the tool for real, through `run_approved` (the only way a mutating tool
             ever runs — no model call can reach it). Reject → a fixed "nothing was saved" reply.
          4. Clear `pending_approval` and let the graph's `approval → END` edge finish the turn.
        """
        pending = state["pending_approval"]
        decision = interrupt({
            "id": pending["id"], "agent": pending["agent"], "tool": pending["tool"],
            "args": pending["args"], "tainted": pending["tainted"], "taint_sources": pending["taint_sources"],
        })

        write = get_stream_writer()
        start = time.perf_counter()
        approved = bool(decision.get("approve")) and decision.get("id") == pending["id"]

        if approved:
            result = await tools.run_approved(pending["agent"], pending["tool"], pending["args"])
            write({
                "stage": "tool", "status": "ok" if result.ok else "error",
                "detail": f"{pending['tool']} · approved · {'ok' if result.ok else 'failed'}", "ms": ms_since(start),
            })
            reply = AIMessage(result.text)
        else:
            write({"stage": "tool", "status": "ok", "detail": f"{pending['tool']} · rejected", "ms": ms_since(start)})
            reply = AIMessage("OK, nothing was saved.")

        return {"messages": [reply], "pending_approval": None}

    return approval


def build_graph(
    model: BaseChatModel,
    checkpointer: BaseCheckpointSaver,
    tools: ToolRegistry,
    memory: MemoryStore,
    workers: tuple[WorkerSpec, ...] = WORKERS,
    classifier: InjectionClassifier | None = None,
):
    """Assemble and compile the chat graph.

    Args:
        model:        the chat model every node uses. The real app passes Claude; tests pass the fake.
        checkpointer: where state is saved between steps and between messages. The real app uses
                      SQLite (data/artlab.db); tests use an in-memory one.
        tools:        the tool registry (tools/catalog.py). Workers search through it.
        memory:       the long-term memory store (Phase 7, docs/contracts.md § 11) `recall` and
                      `remember` use. The real app opens data/chroma/; tests pass a temp one — always
                      isolated, the same way `tools`/`checkpointer` never default to something real.
        workers:      the worker registry (agents/workers.py). Defaults to every real worker; tests
                      can pass their own stub `WorkerSpec`s instead.
        classifier:   the guard's layer-2 injection classifier (docs/contracts.md § 8). None (the
                      default) means "off" — passed straight through to `guard.make_node`.

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
    # drawings) which nodes the supervisor *can* jump to: any worker, "approval", "remember", or
    # "output_guard" (the step-limit stop paths, docs/contracts.md § 14) — `END` stays too, for the
    # "unknown handoff target" error path (§ 3), which never needs the output guard because it never
    # adds an answer. Workers always report back to the supervisor. `approval`, `remember` and
    # `output_guard` are not workers — none is ever in `workers`, so none is a route the model can
    # pick, only something the supervisor's own code (or a fixed edge, below) sends a turn to.
    # `recall` sits between the guard and the supervisor on every turn, so it has one fixed edge, not a
    # `Command`. Compiling with a checkpointer turns on saving: each run is saved under the `thread_id`
    # api.py passes in (one thread = one chat), which is also what makes `interrupt()`/resume possible
    # — see the file header.
    graph = StateGraph(ChatState)
    graph.add_node("guard", guard.make_node(classifier))
    graph.add_node("summarize", summarize.make_node(model))  # a pass-through until the chat is long
    graph.add_node("recall", recall.make_node(memory))
    graph.add_edge("summarize", "recall")
    graph.add_node(
        "supervisor", supervisor.make_node(model, workers),
        destinations=(*names, "approval", "remember", "output_guard", END),
    )
    graph.add_node("remember", remember.make_node(model, memory))
    graph.add_node("output_guard", output.make_node())
    graph.add_node("approval", _make_approval_node(tools))
    graph.add_edge("recall", "supervisor")
    graph.add_edge("remember", "output_guard")
    graph.add_edge("output_guard", END)
    graph.add_edge("approval", END)
    for worker in workers:
        graph.add_node(worker.name, worker.make_node(model, tools))
        graph.add_edge(worker.name, "supervisor")
    graph.add_edge(START, "guard")
    return graph.compile(checkpointer=checkpointer)
