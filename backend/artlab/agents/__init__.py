"""
agents/ — the graph's nodes: one module per step a chat message goes through.

    state.py       ChatState — everything the graph remembers about one chat
    common.py      small helpers several nodes share (text_of, ms_since, tokens_used)
    guard.py       the input guard — blocks a message before any model sees it
    supervisor.py  Arty, the main agent — decides who answers and when the turn is done
    respond.py     answers directly, from the conversation so far
    rag_agent.py   answers from the knowledge base, citing sources

graph.py wires these into the actual LangGraph shape; it doesn't define node bodies itself.
"""
