"""
agents/capabilities.py — the "team and tools" list (X3): who's on the team and what each one can use,
built only from `workers.WORKERS` and the tool registry, so it can never drift from what the code
actually does. Two things read this same data: Arty's own prompt (`respond.py`, via `describe_team`),
so "what tools do you have" is answered from the real list, and `GET /api/agents` (`api.py`), which the
frontend's Team panel (`Sidebar.tsx` → `TeamList.tsx`) renders. Contract: docs/contracts.md § 9.
"""

from artlab.agents.workers import WorkerSpec
from artlab.tools.registry import Tool, ToolRegistry

# Arty is the supervisor plus the "respond" worker (docs/contracts.md § 9): the model routes each
# message, and when it keeps the message for itself, that's "respond" answering. So Arty gets one
# fixed entry here, standing in for both, and "respond" is skipped when we walk WORKERS below —
# listing it separately would show the same agent twice.
ARTY_NAME = "Arty"
ARTY_DESCRIPTION = (
    "the main agent: routes each message to the right specialist, and answers everything else himself "
    "(small talk, general knowledge, writing or brainstorming help) - no tools."
)

# The registry's internal tier names ("read_only", "mutating") are for code; these are what a person
# (or Arty, in a sentence) reads.
TIER_LABELS = {"read_only": "read-only", "mutating": "changes data"}


def _tool_entry(tool: Tool) -> dict:
    """One tool's entry in the team list: its name, tier label and description."""
    return {"name": tool.name, "tier": TIER_LABELS[tool.tier], "description": tool.description}


def team(workers: tuple[WorkerSpec, ...], tools: ToolRegistry) -> list[dict]:
    """The whole team, Arty first: `[{name, description, tools: [{name, tier, description}]}]`.

    Each worker's tools come straight from `tools.tools_for(worker.name)` — the same call the model's
    own tool loop uses (docs/contracts.md § 9) — so a new tool registered for an existing or brand-new
    worker shows up here with no change to this file: register it in `tools/catalog.py`, add the
    worker to `WORKERS` if it's new, and both `describe_team` and `GET /api/agents` pick it up.
    """
    entries = [{"name": ARTY_NAME, "description": ARTY_DESCRIPTION, "tools": []}]
    for worker in workers:
        if worker.name == "respond":
            continue  # respond is Arty answering himself — already covered by the entry above
        entries.append({
            "name": worker.name,
            "description": worker.description,
            "tools": [_tool_entry(tool) for tool in tools.tools_for(worker.name)],
        })
    return entries


def describe_team(workers: tuple[WorkerSpec, ...], tools: ToolRegistry) -> str:
    """The same data as `team`, rendered as short plain text for a prompt — one line per agent:

        Arty - the main agent: ... - no tools
        rag_agent - questions about ... - tools: search_knowledge (read-only): Search the studio's...

    `respond.py` puts this in Arty's system prompt so he can answer "what can you do", "what tools do
    you have" or "what does <agent> do" from this list only, never by inventing a tool or a worker.
    """
    lines = []
    for entry in team(workers, tools):
        if entry["tools"]:
            tool_text = "; ".join(f"{t['name']} ({t['tier']}): {t['description']}" for t in entry["tools"])
            lines.append(f"{entry['name']} - {entry['description']} - tools: {tool_text}")
        else:
            lines.append(f"{entry['name']} - {entry['description']} - no tools")
    return "\n".join(lines)
