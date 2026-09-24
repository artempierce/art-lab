"""
memory/ — long-term memory (Phase 7, M1): facts about the owner that outlive a single chat.

Where this sits in the message flow: `agents/recall.py` reads from here at the start of every turn
(state["memory"]), and `agents/remember.py` writes to it at the end of one, once the supervisor
decides the turn is done. `store.py` is the one file in here so far — the Chroma-backed
`MemoryStore`. See docs/contracts.md § 11 for the full spec.
"""
