"""
tools/registry.py — the tool registry: every tool call goes through here, and gets checked first.

This is the minimal version of the manifest's "tool execution gateway" (docs/architecture.md, layer 4).
Tenet 2 there: a model only *asks* for an action; our code decides whether it happens.

`ToolRegistry.call(agent, tool_name, **args)` checks, in order:

    1. the tool exists
    2. the calling agent is on the tool's allow-list      e.g. search_knowledge: rag_agent only
    3. the tool is read-only                              tools that change data ("mutating") need your
                                                          Approve/Reject click, which arrives in Phase 6;
                                                          until then they are refused outright
Only then does the tool run.

A tool's risk tier and allow-list are fixed when it is registered. The caller can't pass in a lower
tier — that was design bug #2 in the original manifest, where the caller chose the tier itself.

Phase 4 grows this into the full gateway: stub tools, retries, failure handling.
"""

from dataclasses import dataclass
from typing import Any, Callable, Literal

Tier = Literal["read_only", "mutating"]


class ToolDenied(PermissionError):
    """The registry refused a tool call. The message says why, in words an agent (or you) can act on."""


@dataclass(frozen=True)
class Tool:
    """One registered tool.

    fn              the Python function that does the work
    tier            "read_only" (looks things up) or "mutating" (changes data or sends things)
    allowed_agents  names of the agents that may call it
    description     one line, for people and for model prompts
    """

    name: str
    fn: Callable[..., Any]
    tier: Tier
    allowed_agents: frozenset[str]
    description: str


class ToolRegistry:
    """Holds the app's tools and enforces who may call which."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, name: str, fn: Callable[..., Any], tier: Tier, allowed_agents: set[str], description: str) -> None:
        """Add a tool. Its tier and allow-list can't be changed by callers later."""
        self._tools[name] = Tool(name, fn, tier, frozenset(allowed_agents), description)

    def call(self, agent: str, name: str, **args: Any) -> Any:
        """Run tool `name` on behalf of `agent`, after the three checks. Raises ToolDenied if any fails."""
        tool = self._tools.get(name)
        if tool is None:
            raise ToolDenied(f"unknown tool {name!r}")
        if agent not in tool.allowed_agents:
            raise ToolDenied(f"{agent} may not use {name} (allowed: {', '.join(sorted(tool.allowed_agents))})")
        if tool.tier == "mutating":
            raise ToolDenied(f"{name} changes data and needs your approval, which arrives in Phase 6")
        return tool.fn(**args)
