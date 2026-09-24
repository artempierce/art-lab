"""
tests/test_capabilities.py — X3 capabilities tests: the "team and tools" list is built only from
`workers.WORKERS` and the tool registry (docs/contracts.md § 9), so a new tool or worker shows up in
it without any change to `capabilities.py`. Also checks Arty's own prompt carries the same text, and
that GET /api/agents serves the list to the frontend's Team panel.
"""

import asyncio

from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from artlab.agents.capabilities import ARTY_DESCRIPTION, ARTY_NAME, describe_team, team
from artlab.agents.workers import WORKERS, WorkerSpec
from artlab.api import create_app
from artlab.graph import build_graph
from artlab.model import FakeChatModel, fake_model
from artlab.rag.knowledge import KnowledgeBase
from artlab.tools.catalog import build_tools
from artlab.tools.registry import ToolRegistry


def empty_kb(tmp_path) -> KnowledgeBase:
    """An empty knowledge base with instant fake embeddings — just enough to build the real tool
    registry (`build_tools`), the same helper `test_api.py` uses."""
    return KnowledgeBase(tmp_path / "chroma", embeddings=DeterministicFakeEmbedding(size=32), min_score=-1)


def test_team_lists_arty_first_then_rag_agent_with_its_read_only_tool(tmp_path):
    """Protects the shape docs/contracts.md § 9 promises the UI and Arty's prompt: Arty is always
    first, has no tools ("respond" isn't listed separately — it's Arty answering himself), and
    rag_agent follows with its one real tool, correctly labelled read-only."""
    tools = build_tools(empty_kb(tmp_path))

    entries = team(WORKERS, tools)

    assert entries[0] == {"name": ARTY_NAME, "description": ARTY_DESCRIPTION, "tools": []}
    assert all(e["name"] != "respond" for e in entries)  # folded into Arty, not listed again

    rag_entry = next(e for e in entries if e["name"] == "rag_agent")
    assert rag_entry["tools"] == [{
        "name": "search_knowledge",
        "tier": "read-only",
        "description": (
            "Search the studio's knowledge base (policies, processes, project docs) for passages "
            "relevant to a question."
        ),
    }]


def test_a_tool_registered_for_a_new_agent_shows_up_with_no_code_change():
    """`team` reads `tools.tools_for(worker.name)` for whatever workers it's given, so a brand-new
    agent with a brand-new tool appears just from registering it — no change to capabilities.py."""
    tools = ToolRegistry()
    tools.register(
        "draft_thumbnail", lambda brief: brief, tier="mutating",
        allowed_agents={"thumbnail_artist"}, description="Draft a thumbnail from a brief.",
    )
    new_worker = WorkerSpec(
        name="thumbnail_artist", description="makes thumbnails", make_node=lambda model, tools: None
    )

    entries = team((*WORKERS, new_worker), tools)

    artist_entry = next(e for e in entries if e["name"] == "thumbnail_artist")
    assert artist_entry["tools"] == [{
        "name": "draft_thumbnail", "tier": "changes data", "description": "Draft a thumbnail from a brief.",
    }]


def test_respond_system_prompt_contains_the_team_text():
    """Arty's own system prompt must carry the exact text `describe_team` builds, so "what can you do"
    / "what tools do you have" answers come from the real registries, never an invented list.

    Runs the real graph (routed to only the "respond" worker, so the fake supervisor can't send it
    anywhere else) rather than calling the node directly — `get_stream_writer()` inside it needs a
    real LangGraph runnable context, which only running the graph provides."""

    class SpyModel(FakeChatModel):
        """The fake model, but it records the system message (the first one) of every prompt."""

        prompts: list[str] = Field(default_factory=list)

        def _reply(self, messages):
            self.prompts.append(str(messages[0].content))
            return super()._reply(messages)

    spy = SpyModel()
    tools = ToolRegistry()  # empty registry: describe_team should still list every non-respond worker
    respond_spec = next(w for w in WORKERS if w.name == "respond")
    graph = build_graph(spy, InMemorySaver(), tools, workers=(respond_spec,))

    asyncio.run(graph.ainvoke({"messages": [HumanMessage("what can you do?")]}, {"configurable": {"thread_id": "t"}}))

    assert describe_team(WORKERS, tools) in spy.prompts[-1]


def test_get_agents_returns_the_team(tmp_path):
    """GET /api/agents (the frontend's Team panel) returns the same list `team` builds, sourced from
    the real tool registry the app builds in its lifespan and stores on app.state.tools."""
    app = create_app(model=fake_model(), checkpointer=InMemorySaver(), knowledge=empty_kb(tmp_path))
    with TestClient(app) as client:
        response = client.get("/api/agents")

    assert response.status_code == 200
    body = response.json()
    assert body[0]["name"] == ARTY_NAME
    assert any(e["name"] == "rag_agent" and e["tools"] for e in body)
