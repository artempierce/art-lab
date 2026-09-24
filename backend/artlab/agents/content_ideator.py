"""
agents/content_ideator.py — node 3c of the graph: the studio's idea specialist.

Runs when the supervisor routes a message here: requests for video ideas, hooks or outlines. Built on
the model-driven tool loop (agents/tool_loop.py, docs/contracts.md § 9): the model gets whatever tools
the registry lists for "content_ideator" and decides for itself whether to use them, up to the loop's
3-call cap — this file never names a tool, so it doesn't need to change when tools are added.

In Phase 5 the registry had no tools for this agent, so a turn was exactly one model call —
`run_tool_loop`'s own behaviour for a tool-less worker (§ 9). Phase 6 registers its first tool,
`save_ideas` (tools/catalog.py) — a mutating tool, so the model can *ask* for it but it never just
runs: `run_tool_loop` catches the gateway's `ApprovalRequired` and this worker passes the resulting
`pending_approval` request on, unchanged, in its own state update. Phase 8 adds a second tool,
`load_skill`, so it can pull in a whole skill (like backend/skills/hook-formulas/SKILL.md) instead of
relying on the handful of hook patterns folded into PROMPT below — this file won't need to change
for that either.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from artlab.agents.state import ChatState
from artlab.agents.tool_loop import run_tool_loop
from artlab.tools.registry import ToolRegistry

# The node name, and the route value the supervisor's RouteDecision.next picks (docs/contracts.md § 2).
NAME = "content_ideator"

# Phase 6 (docs/contracts.md § 10): how much of the chat's own history this worker sees besides its own
# task, so "save those ideas" can find the ideas it wrote last turn. Small on purpose — this is recent
# back-and-forth, not the model's whole memory; a real Claude call still pays for every token it reads.
HISTORY_MESSAGES = 4

# The studio's idea specialist. Four patterns from backend/skills/hook-formulas/SKILL.md are folded in
# directly here, because Phase 5 has no way to load a skill yet (that's `load_skill`, Phase 8) — this is
# a stand-in for that file, not a copy of it.
PROMPT = """You are content_ideator, Art Lab's video idea specialist for a YouTube creator's studio.

Give EXACTLY 3 video ideas. For each one, give:
  - a title (at most 60 characters — the studio's brand rule)
  - a hook: what the video says or shows in the first 5 seconds to stop someone from scrolling past
  - a 3-point outline: the three beats the video hits, in order

Rules:
- No clickbait: every hook must be something the finished video actually delivers. Never promise more
  than the idea can pay off.
- If the request includes research (comments, trends, a past video's numbers), build the ideas on it —
  don't ignore it and fall back to generic suggestions.
- Be concise and specific; "make a video about X" is not an idea.

Reach for these hook patterns when they fit (there are more, but these four cover most requests):
  - The Problem Hook: name a pain point the audience has, then hint at the fix.
  - The Counter-Intuitive Flip: contradict a common belief in the niche, then promise the real story.
  - The List Hook: promise a specific, bounded list and name the first item right away.
  - The Direct Question Hook: ask the one question that nails the viewer's exact situation."""


def make_node(model: BaseChatModel, tools: ToolRegistry):
    """Build the `content_ideator` node (docs/contracts.md §§ 2, 9's worker template).

    Just closes over `model` and `tools` and hands them to `run_tool_loop` on every turn — the loop
    does everything (binding whatever tools the registry lists for this agent, the 3-call cap, taint
    tracking); this function only supplies NAME and PROMPT.
    """

    async def content_ideator(state: ChatState) -> dict:
        """Node 3c — answer with 3 ideas, through the tool loop.

        Steps:
          1. Take the last HISTORY_MESSAGES messages *before* this turn's own question as `history`
             (docs/contracts.md § 10): `state["messages"][-1]` is always the raw message you just sent
             — `state["task"]` is the supervisor's standalone rewrite of that same message, so passing
             both would show the model its own question twice.
          2. Run the tool loop with this worker's prompt, the turn's standalone question (`task`), and
             that history — plus the chat's incoming `tainted`/`taint_sources`, so a request to save
             right after tainted research still needs approval and still shows why.
          3. Return the reply and its cost, and set `answered_by` so the supervisor knows this step
             finished (docs/contracts.md § 2).
          4. A tool waiting for your click (`result.pending`)? Pass it on as `pending_approval`
             unchanged — the supervisor routes straight to the approval node next (§ 10).
          5. If the loop used (or already carried) any untrusted tool result, mark the chat tainted and
             say where from (§§ 1, 10) — `run_tool_loop` never touches `ChatState` itself, so every
             Phase 5 worker adds this one line the same way.
        """
        history: list[BaseMessage] = state["messages"][:-1][-HISTORY_MESSAGES:]
        result = await run_tool_loop(
            model, tools, NAME, PROMPT, state["task"],
            tainted_in=state.get("tainted", False),
            taint_sources_in=state.get("taint_sources", []),
            history=history,
        )
        update = {"messages": [result.reply], "spent_usd": result.spent_usd, "answered_by": NAME}
        if result.pending:
            update["pending_approval"] = result.pending
        if result.tainted:
            update["tainted"] = True
            update["taint_sources"] = result.taint_sources
        return update

    return content_ideator
