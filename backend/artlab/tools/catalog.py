"""
tools/catalog.py — every tool in the app, its risk tier, and which agents may use it. One place to look.

    tool                  tier        allowed agents      what it does
    search_knowledge      read-only   rag_agent           top matching chunks from the knowledge base
    query_youtube_trends  read-only   youtube_researcher  fake trending-videos search (stub, T2)
    fetch_comments        read-only   youtube_researcher  fake comments for a video (stub, T2)
    save_ideas            mutating    content_ideator     write the ideas you just gave to a dated file (Phase 6)

Later phases add more tools here (load_skill, …), each with its own allow-list.
"""

from pathlib import Path

from artlab.config import IDEAS_DIR
from artlab.rag.knowledge import KnowledgeBase
from artlab.tools.ideas import make_save_ideas
from artlab.tools.registry import ToolRegistry
from artlab.tools.stubs import fetch_comments, query_youtube_trends


def build_tools(knowledge: KnowledgeBase, ideas_dir: Path = IDEAS_DIR) -> ToolRegistry:
    """Create the registry with every tool, wired to the given knowledge base.

    `ideas_dir` is where `save_ideas` writes (docs/contracts.md § 10). The real app never passes it —
    the default is the real data/ideas folder — but tests point it at a temp folder (via
    `api.create_app(ideas_dir=...)`) so running the suite never touches the repo's own data.
    """
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
    # Phase 6 (docs/contracts.md § 10): content_ideator's first data-changing tool. Its mutating tier
    # means the gateway never lets a model run it directly (ApprovalRequired) — only the graph's
    # `approval` node can, after your click. untrusted_output=False: the confirmation text it returns
    # is our own code's, not outside data, so it isn't wrapped.
    tools.register(
        "save_ideas",
        make_save_ideas(ideas_dir),
        tier="mutating",
        allowed_agents={"content_ideator"},
        description="Save the video ideas you just gave to a dated file in the studio's ideas folder.",
        untrusted_output=False,
    )
    return tools
