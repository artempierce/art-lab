"""
tools/catalog.py — every tool in the app, its risk tier, and which agents may use it. One place to look.

    tool                   tier        allowed agents      what it does
    search_knowledge       read-only   rag_agent           top matching chunks from the knowledge base
    query_youtube_trends   read-only   youtube_researcher  fake trending-videos search (stub, T2)
    fetch_comments         read-only   youtube_researcher  fake comments for a video (stub, T2)
    save_ideas             mutating    content_ideator     write the ideas you just gave to a dated file (Phase 6)
    load_skill             read-only   youtube_researcher, content_ideator, english_coach
                                        pull a whole skill's text into this turn, by name (Phase 8)
    ask_youtube_researcher read-only   content_ideator     hand youtube_researcher a question and read
                                                             back its answer (Phase 9b, § 13)

Later phases add more tools here, each with its own allow-list.
"""

from pathlib import Path

from langchain_core.language_models import BaseChatModel

from artlab.agents.youtube_researcher import PROMPT as YOUTUBE_RESEARCHER_PROMPT
from artlab.config import IDEAS_DIR
from artlab.model import fake_model
from artlab.rag.knowledge import KnowledgeBase
from artlab.skills.loader import Skill, load_skills
from artlab.tools.agents import make_agent_tool
from artlab.tools.ideas import make_save_ideas
from artlab.tools.registry import ToolRegistry
from artlab.tools.skills import make_load_skill
from artlab.tools.stubs import fetch_comments, query_youtube_trends


def build_tools(
    knowledge: KnowledgeBase,
    ideas_dir: Path = IDEAS_DIR,
    skills: dict[str, Skill] | None = None,
    model: BaseChatModel | None = None,
) -> ToolRegistry:
    """Create the registry with every tool, wired to the given knowledge base.

    `ideas_dir` is where `save_ideas` writes (docs/contracts.md § 10). The real app never passes it —
    the default is the real data/ideas folder — but tests point it at a temp folder (via
    `api.create_app(ideas_dir=...)`) so running the suite never touches the repo's own data.

    `skills` is what `load_skill` can hand back (Phase 8, docs/contracts.md § 12); `None` (every real
    call site) reads the repo's own `backend/skills/` once, via `load_skills()`. Unlike `ideas_dir`,
    tests use this same real folder too — there's nothing to fake about our own checked-in markdown.

    `model` is what `ask_youtube_researcher`'s nested turn runs on (Phase 9b, docs/contracts.md § 13).
    The real app passes its one shared chat model (api.py) — the same model every worker's own turn
    already runs on, so asking youtube_researcher a question costs and behaves exactly like it
    answering its own turn. `None` (every test that doesn't care about agent-calling-agent) falls back
    to the free `fake_model()` — never a real, paid call just from building the registry.
    """
    skills = load_skills() if skills is None else skills
    model = fake_model() if model is None else model
    tools = ToolRegistry(skills=skills)
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
    # Phase 8 (docs/contracts.md § 12): every skill-reading worker's window into backend/skills/. Its
    # output is our own reviewed repo text, not outside data, so untrusted_output=False — loading a
    # skill never taints the chat.
    tools.register(
        "load_skill",
        make_load_skill(skills),
        tier="read_only",
        allowed_agents={"youtube_researcher", "content_ideator", "english_coach"},
        description="Load a skill's full instructions by name, when the task needs know-how beyond what's already in your prompt.",
        untrusted_output=False,
    )
    # Phase 9b (docs/contracts.md § 13): "agent calls agent" — content_ideator can hand
    # youtube_researcher a question mid-turn and read back its answer, the same way it reads back any
    # other tool's result. read_only: asking a question never changes anything on its own (whatever
    # youtube_researcher itself does inside its own turn is still checked against *its* allow-list).
    # untrusted_output=True (the default, named here for clarity): the researcher's answer is built
    # from YouTube data, so it's exactly as untrusted as query_youtube_trends or fetch_comments
    # themselves. timeout_s=60: this tool runs a whole nested turn — its own model calls and up to
    # MAX_TOOL_CALLS tool calls — which needs far longer than an ordinary tool's 10s default.
    tools.register(
        "ask_youtube_researcher",
        make_agent_tool(model, tools, "youtube_researcher", YOUTUBE_RESEARCHER_PROMPT),
        tier="read_only",
        allowed_agents={"content_ideator"},
        description="Ask youtube_researcher a question (e.g. about trends or viewer comments) and get its research answer back.",
        untrusted_output=True,
        timeout_s=60.0,
    )
    return tools
