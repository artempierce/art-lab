"""
tools/catalog.py — every tool in the app, its risk tier, and which agents may use it. One place to look.

    tool               tier        allowed agents   what it does
    search_knowledge   read-only   rag_agent        top matching chunks from the knowledge base

Later phases add tools here (YouTube stubs, save_ideas, load_skill, …), each with its own allow-list.
"""

from artlab.rag.knowledge import KnowledgeBase
from artlab.tools.registry import ToolRegistry


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
    return tools
