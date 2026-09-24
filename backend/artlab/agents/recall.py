"""
agents/recall.py — node 2 of the graph: loads your saved facts before the supervisor runs
(Phase 7, M1). Runs right after the guard, on every turn — including a flagged one: recalling what
you've already told Art Lab isn't a security decision, only *extracting new* facts is (see
agents/remember.py, and docs/contracts.md § 11's safety rule).

Puts the facts into state["memory"]; the supervisor and respond then show them to the model as one
more system message, wrapped as untrusted (docs/contracts.md § 11) — this node itself never touches a
model, so it costs $0.
"""

import time

from langgraph.config import get_stream_writer

from artlab.agents.common import ms_since, text_of
from artlab.agents.state import ChatState
from artlab.memory.store import MemoryStore


def make_node(store: MemoryStore):
    """Build the `recall` node, closing over the memory store."""

    async def recall(state: ChatState) -> dict:
        """Node 2 — load the facts most relevant to what you just said into state["memory"].

        Steps:
          1. Use the message you just sent (the last one in history — guard runs first and never
             appends anything on the path that reaches here) as the recall query.
          2. `MemoryStore.recall` does the rest: with few enough facts stored it just returns all of
             them, otherwise the ones closest in meaning to the query (memory/store.py).
          3. Trace: how many facts came back, so you can watch memory build up as you chat.
        """
        start = time.perf_counter()
        write = get_stream_writer()

        query = text_of(state["messages"][-1])
        facts = store.recall(query)

        detail = f"recalled {len(facts)} fact{'s' if len(facts) != 1 else ''}" if facts else "no facts yet"
        write({"stage": "memory", "status": "ok", "detail": detail, "ms": ms_since(start)})
        return {"memory": facts}

    return recall
