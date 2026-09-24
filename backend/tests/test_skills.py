"""
tests/test_skills.py — Phase 8 tests (S1, docs/contracts.md § 12): skills loaded on demand. An agent
sees only a skill's name and one-line description until it calls `load_skill`; the full text only
enters its own turn then, and never taints the chat — a skill is our own reviewed repo text, not
outside data. That's the difference between an agent and a skill (design book: "just know-how → skill").

Covers every layer: the loader (skills/loader.py) reading backend/skills/ and skipping a folder whose
front matter lies about its own name, the load_skill tool itself (tools/skills.py), the skills index
`run_tool_loop` adds to an allowed agent's system prompt (and withholds from one without the tool), and
the two S9 end-to-end scenarios through the real FastAPI app — one where the fake model reaches for the
hook-formulas skill, one where it doesn't need to. Like test_phase5_routes.py / test_memory.py, the
end-to-end tests drive the real app with `TestClient`: real graph, real tool gateway, real (repo) skills
— only the model is the free `FakeChatModel`, so every test here is instant and costs $0 (CLAUDE.md's
cost rule).
"""

import asyncio
import json

from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from pydantic import Field

from artlab.agents import content_ideator, english_coach
from artlab.agents.state import ChatState
from artlab.agents.tool_loop import run_tool_loop
from artlab.api import create_app
from artlab.memory.store import MemoryStore
from artlab.model import FakeChatModel, fake_model
from artlab.rag.knowledge import KnowledgeBase
from artlab.skills.loader import load_skills
from artlab.tools.catalog import build_tools
from artlab.tools.registry import ToolRegistry
from artlab.tools.skills import make_load_skill


# ===== helpers (the same shapes test_approvals.py / test_phase5_routes.py / test_memory.py use) =====

def parse_sse(body: str) -> list[tuple[str, dict]]:
    """Split a full SSE response body into [(event name, data dict), ...], in order (test_api.py)."""
    events = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["event"], json.loads(fields["data"])))
    return events


def send(client: TestClient, message: str, thread_id: str | None = None) -> list[tuple[str, dict]]:
    """POST one chat message and return its parsed events. thread_id None starts a new chat."""
    response = client.post("/api/chat", json={"message": message, "thread_id": thread_id})
    assert response.status_code == 200
    return parse_sse(response.text)


def answer(events) -> str:
    """Join all `token` events back into the full answer text, as the browser does."""
    return "".join(data["text"] for name, data in events if name == "token")


def trace_lines(events) -> list[dict]:
    """The `trace` events' data dicts, in order — one per graph step."""
    return [data for name, data in events if name == "trace"]


def empty_kb(tmp_path) -> KnowledgeBase:
    """An empty knowledge base in a temp folder with instant fake embeddings (test_api.py)."""
    return KnowledgeBase(tmp_path / "chroma", embeddings=DeterministicFakeEmbedding(size=32), min_score=-1)


def empty_memory(tmp_path) -> MemoryStore:
    """A private, temp-folder memory store with instant fake embeddings (Phase 7, M1) — every real
    graph needs one now that recall/remember are permanent nodes."""
    return MemoryStore(tmp_path / "chroma-memory", embeddings=DeterministicFakeEmbedding(size=32))


# ===== the loader =====

def test_loader_reads_all_three_real_skills():
    """docs/contracts.md § 12: `load_skills()` with no argument reads the repo's own backend/skills/
    and must find exactly the three skills the app ships with — nothing missing, nothing extra."""
    skills = load_skills()
    assert set(skills) == {"hook-formulas", "retention-analysis", "style-guide"}


def test_a_folder_whose_front_matter_name_does_not_match_is_skipped(tmp_path):
    """The loader's own integrity check (docs/contracts.md § 12): a SKILL.md whose front-matter `name`
    doesn't match the folder it's in is skipped entirely — not loaded under either name. A mismatch
    usually means a copied or renamed folder nobody updated, and loading it anyway would make
    `load_skill` reachable by a name nothing on disk is really called."""
    good = tmp_path / "good-skill"
    good.mkdir()
    (good / "SKILL.md").write_text("---\nname: good-skill\ndescription: a fine skill\n---\nBody text.")

    bad = tmp_path / "renamed-folder"
    bad.mkdir()
    (bad / "SKILL.md").write_text("---\nname: old-name\ndescription: a stale skill\n---\nBody text.")

    skills = load_skills(tmp_path)

    assert set(skills) == {"good-skill"}
    assert "old-name" not in skills and "renamed-folder" not in skills


# ===== the load_skill tool =====

def test_load_skill_returns_the_body():
    """tools/skills.py: `load_skill("hook-formulas")` must return that file's actual markdown body —
    proof the loader's parsed body, not just its front matter, reaches the tool."""
    load_skill = make_load_skill(load_skills())
    body = load_skill("hook-formulas")

    assert "Formula 1: The Problem Hook" in body
    assert "---" not in body  # the front matter delimiters never leak into the body


def test_load_skill_unknown_name_lists_whats_available():
    """docs/contracts.md § 12's exact contract: an unknown name gets "No skill named '<name>'.
    Available: ..." naming every loaded skill, so a model that mistypes a name can self-correct next
    turn instead of just failing silently."""
    load_skill = make_load_skill(load_skills())
    text = load_skill("nope")

    assert text == "No skill named 'nope'. Available: hook-formulas, retention-analysis, style-guide."


def test_loading_a_skill_does_not_taint_the_chat(tmp_path):
    """docs/contracts.md § 12: load_skill is registered with untrusted_output=False — a skill is our
    own reviewed repo text, not outside data, so running it must never mark a chat's tool result as
    untrusted the way an outside tool (search_knowledge, query_youtube_trends...) would."""
    tools = build_tools(empty_kb(tmp_path), ideas_dir=tmp_path / "ideas")
    result = asyncio.run(tools.call("content_ideator", "load_skill", name="hook-formulas"))

    assert result.ok is True
    assert result.untrusted is False


# ===== the skills index in the system prompt =====

class RecordingModel(FakeChatModel):
    """The fake model, but it records the full message list of every call, so a test can check the
    system prompt's exact text without a real API call — the same trick test_content_ideator.py's and
    test_english_coach.py's SpyFakeModel use. `bind_tools` returns a *copy* of the model (model.py), so
    `calls` — kept on the instance a test holds onto — stays the same list object across the copy."""

    calls: list[list] = Field(default_factory=list)

    def _reply(self, messages):
        self.calls.append(list(messages))
        return super()._reply(messages)


def _system_prompt(model: RecordingModel, tools: ToolRegistry, agent: str, task: str) -> str:
    """Run `run_tool_loop` once, inside the smallest graph that can give `get_stream_writer()` a
    context (the same pattern test_tool_loop.py's run_loop uses), and return the system message text
    it built — so a test can check the skills index is (or isn't) part of it."""

    async def node(state: ChatState) -> dict:
        r = await run_tool_loop(model, tools, agent, "You are a test worker.", state["task"], tainted_in=False)
        return {"messages": [r.reply], "spent_usd": r.spent_usd, "answered_by": agent}

    async def _run():
        graph = (
            StateGraph(ChatState)
            .add_node(agent, node)
            .add_edge(START, agent)
            .compile(checkpointer=InMemorySaver())
        )
        await graph.ainvoke({"messages": [HumanMessage(task)], "task": task}, {"configurable": {"thread_id": "t"}})

    asyncio.run(_run())
    system = next(m for m in model.calls[0] if isinstance(m, SystemMessage))
    return str(system.content)


def test_skills_index_appears_for_an_agent_allowed_to_use_load_skill():
    """docs/contracts.md § 12: an agent whose tools include load_skill gets the skills index appended
    to its system prompt — every loaded skill's name and description — so it knows what it could load
    before deciding whether a task needs it."""
    tools = ToolRegistry(skills=load_skills())
    tools.register(
        "load_skill", make_load_skill(tools.skills), tier="read_only",
        allowed_agents={"content_ideator"}, description="Load a skill.", untrusted_output=False,
    )

    prompt = _system_prompt(RecordingModel(), tools, "content_ideator", "hello")

    assert "Skills you can load with load_skill" in prompt
    assert "hook-formulas:" in prompt


def test_skills_index_does_not_appear_for_an_agent_without_load_skill():
    """The other half of the same rule: an agent load_skill isn't registered for (e.g. rag_agent) never
    sees the index at all, not even an empty one — it has no way to act on it."""
    tools = ToolRegistry(skills=load_skills())
    tools.register(
        "load_skill", make_load_skill(tools.skills), tier="read_only",
        allowed_agents={"content_ideator"}, description="Load a skill.", untrusted_output=False,
    )

    prompt = _system_prompt(RecordingModel(), tools, "rag_agent", "hello")

    assert "load_skill" not in prompt


# ===== S9 end to end =====

def test_hooks_request_loads_the_hook_formulas_skill(tmp_path):
    """S9: "Write catchy hooks for these 3 ideas" reaches content_ideator, whose fake model recognises
    "hooks" (FAKE_SKILL_HINTS, model.py) and calls load_skill — the trace carries a "skill" stage line
    naming the skill it loaded, and the turn still finishes with a real answer."""
    app = create_app(
        model=fake_model(["Here are the hooks."]), checkpointer=InMemorySaver(),
        knowledge=empty_kb(tmp_path), ideas_dir=tmp_path / "ideas", memory=empty_memory(tmp_path),
    )
    with TestClient(app) as client:
        events = send(client, "Write catchy hooks for these 3 ideas")
        lines = trace_lines(events)

    assert any(d["stage"] == "arty" and d["detail"].startswith("→ content_ideator") for d in lines)
    skill_lines = [d for d in lines if d["stage"] == "skill"]
    assert len(skill_lines) == 1
    assert skill_lines[0]["detail"] == "loaded hook-formulas"
    assert answer(events)  # a real answer streamed


def test_plain_ideas_request_loads_no_skill(tmp_path):
    """The index is offered, but the fake doesn't need it here: "Give me 3 video ideas about desk
    setups" doesn't mention hooks, so FAKE_SKILL_HINTS matches nothing and content_ideator answers
    directly, with no "skill" trace line at all."""
    app = create_app(
        model=fake_model(["Here are 3 desk-setup ideas."]), checkpointer=InMemorySaver(),
        knowledge=empty_kb(tmp_path), ideas_dir=tmp_path / "ideas", memory=empty_memory(tmp_path),
    )
    with TestClient(app) as client:
        events = send(client, "Give me 3 video ideas about desk setups")
        lines = trace_lines(events)

    assert not [d for d in lines if d["stage"] == "skill"]
    assert answer(events) == "Here are 3 desk-setup ideas."


# ===== leaner prompts =====

def test_content_ideator_prompt_no_longer_inlines_hook_patterns_and_mentions_load_skill():
    """docs/contracts.md § 12: the hook patterns Phase 5 folded directly into PROMPT as a stand-in are
    gone now that hook-formulas is a real, loadable skill — and PROMPT tells the model load_skill
    exists, so it knows to reach for it instead of guessing."""
    assert "The Problem Hook" not in content_ideator.PROMPT
    assert "The Counter-Intuitive Flip" not in content_ideator.PROMPT
    assert "load_skill" in content_ideator.PROMPT


def test_english_coach_prompt_no_longer_inlines_style_rules_and_mentions_load_skill():
    """Same rule for english_coach: the detailed grammar/clarity/tone rule list Phase 5 folded in as a
    stand-in for style-guide is gone, and PROMPT tells the model load_skill exists."""
    assert "subject-verb agreement" not in english_coach.PROMPT
    assert "it's = it is, its = possessive" not in english_coach.PROMPT
    assert "load_skill" in english_coach.PROMPT
