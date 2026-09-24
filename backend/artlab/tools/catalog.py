"""
tools/catalog.py — every tool in the app, its risk tier, and which agents may use it. One place to look.

    tool                  tier        allowed agents      what it does
    search_knowledge      read-only   rag_agent           top matching chunks from the knowledge base
    query_youtube_trends  read-only   youtube_researcher  fake trending-videos search (stub, T2)
    fetch_comments        read-only   youtube_researcher  fake comments for a video (stub, T2)

Later phases add tools here (save_ideas, load_skill, …), each with its own allow-list.
"""

from artlab.rag.knowledge import KnowledgeBase
from artlab.tools.registry import ToolRegistry
from artlab.tools.stubs import fetch_comments, query_youtube_trends


def build_tools(knowledge: KnowledgeBase) -> ToolRegistry:
    """Create the registry with every tool, wired to the given knowledge base."""
    tools = ToolRegistry()
    tools.register(
        "search_knowledge",
        knowledge.search,
        tier="read_only",
        allowed_agents={"rag_agent"},
        description="Search the studio's knowledge base (policies, processes, project docs) for passages relevant to a question.",
    )
    # Stubs (T2): free, deterministic fakes for the Phase 5 youtube_researcher worker.
    # See tools/stubs.py for why they're stubs, and why one fetch_comments entry is poisoned.
    tools.register(
        "query_youtube_trends",
        query_youtube_trends,
        tier="read_only",
        allowed_agents={"youtube_researcher"},
        description="Look up trending videos in a niche (title, views, growth, channel).",
    )
    tools.register(
        "fetch_comments",
        fetch_comments,
        tier="read_only",
        allowed_agents={"youtube_researcher"},
        description="Fetch the comments on a video.",
    )
    return tools
