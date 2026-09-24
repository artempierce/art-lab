"""
tools/registry.py — the tool gateway: every tool call goes through here, gets checked first, and its
result gets wrapped on the way back.

This is the manifest's "tool execution gateway" (docs/architecture.md, layer 4), growing phase by phase.
Tenet 2 there: a model only *asks* for an action; our code decides whether it happens.

`await tools.call(agent, tool_name, tainted=..., **args)` does, in order:

    1. check   the tool exists
               the calling agent is on the tool's allow-list        e.g. search_knowledge: rag_agent only
               the tool is read-only                                tools that change data ("mutating") need
                                                                    your Approve/Reject click (Phase 6); until
                                                                    then they are refused outright, and in a
                                                                    tainted chat they always will be
    2. run     the tool, in a worker thread (tools are ordinary blocking functions)
    3. render  its return value as `Piece`s (tools/untrusted.py)
    4. wrap    every piece in <untrusted_retrieval> if the tool's output is untrusted (the default)

and returns a `ToolResult`: the raw value for our code, and the wrapped text for a model.

Why the gateway wraps, not each tool: the trust boundary is a property of the *system*
(docs/execution-plan.md § 4a). If every tool wrapped its own output, one forgotten call would let outside
text reach a model looking like instructions. Here, a new tool is wrapped without doing anything.

A tool's risk tier, allow-list and untrusted flag are fixed when it is registered. The caller can't pass
in a lower tier — that was design bug #2 in the original manifest, where the caller chose the tier itself.

Phase 4 (ticket T2) adds: stub tools, a per-tool timeout, one retry, failures returned as a failed
ToolResult instead of an exception, and an injection scan of every result on arrival.
Contract: docs/contracts.md § 4.
"""

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Callable, Literal

from artlab.tools.untrusted import Piece, wrap_untrusted

Tier = Literal["read_only", "mutating"]


class ToolDenied(PermissionError):
    """The registry refused a tool call. The message says why, in words an agent (or you) can act on."""


@dataclass(frozen=True)
class Tool:
    """One registered tool.

    fn                the Python function that does the work
    tier              "read_only" (looks things up) or "mutating" (changes data or sends things)
    allowed_agents    names of the agents that may call it
    description       one line, for people and for model prompts
    untrusted_output  True if what it returns comes from outside Art Lab (documents, web pages, YouTube…),
                      so the gateway wraps it and the chat becomes tainted. True unless we say otherwise.
    """

    name: str
    fn: Callable[..., Any]
    tier: Tier
    allowed_agents: frozenset[str]
    description: str
    untrusted_output: bool = True


@dataclass(frozen=True)
class ToolResult:
    """What a tool call gives back.

    tool       which tool ran
    ok         False when the tool failed (from Phase 4: after its one retry)
    data       the tool's own return value, for *our* code (e.g. a SearchResult's hits and citations);
               None if the tool failed. Never put this in a prompt: use `text`.
    text       what a model may see: the pieces, each wrapped as untrusted if the tool's output is untrusted
    untrusted  the tool's registered untrusted_output. A worker that uses this result returns
               {"tainted": True} so the whole chat is marked (docs/contracts.md § 1).
    error      what went wrong, when ok is False                                      (Phase 4)
    attempts   how many tries it took: 2 when the first one failed                     (Phase 4)
    flagged    the arrival scan found injection patterns in the text                   (Phase 4)
    """

    tool: str
    ok: bool
    data: Any
    text: str
    untrusted: bool
    error: str | None = None
    attempts: int = 1
    flagged: bool = False


def to_pieces(tool_name: str, value: Any) -> list[Piece]:
    """Turn a tool's return value into pieces of text.

    Three cases, checked in order:
      * the value knows how to split itself (it has a `.pieces()` method, like SearchResult) → use that
      * plain text → one piece, labelled with the tool's name as its source
      * anything else (dicts, lists…) → one piece of JSON
    """
    if hasattr(value, "pieces"):
        return value.pieces()
    if isinstance(value, str):
        return [Piece(source=tool_name, text=value)]
    return [Piece(source=tool_name, text=json.dumps(value, default=str, ensure_ascii=False))]


def render_pieces(pieces: list[Piece], untrusted: bool) -> str:
    """Join pieces into the text a model sees, wrapping each one if the output is untrusted.

    Example, one untrusted piece labelled "[1] knowledge/brand-voice.md › Titles":

        [1] knowledge/brand-voice.md › Titles
        <untrusted_retrieval source="knowledge/brand-voice.md">
        Titles are at most 60 characters…
        </untrusted_retrieval>

    The label sits outside the wrapper: it's written by our code, and the model needs it to cite [1].
    """
    blocks = []
    for piece in pieces:
        body = wrap_untrusted(piece.text, piece.source) if untrusted else piece.text
        blocks.append(f"{piece.label}\n{body}" if piece.label else body)
    return "\n\n".join(blocks)


class ToolRegistry:
    """Holds the app's tools, enforces who may call which, and wraps what they return."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(
        self,
        name: str,
        fn: Callable[..., Any],
        tier: Tier,
        allowed_agents: set[str],
        description: str,
        *,
        untrusted_output: bool = True,
    ) -> None:
        """Add a tool. Its tier, allow-list and untrusted flag can't be changed by callers later."""
        self._tools[name] = Tool(name, fn, tier, frozenset(allowed_agents), description, untrusted_output)

    def check(self, agent: str, name: str, tainted: bool = False) -> Tool:
        """Step 1: return the tool if `agent` may run it now, else raise ToolDenied saying why.

        A refusal is a decision, not a failure: it is never retried.
        """
        tool = self._tools.get(name)
        if tool is None:
            raise ToolDenied(f"unknown tool {name!r}")
        if agent not in tool.allowed_agents:
            raise ToolDenied(f"{agent} may not use {name} (allowed: {', '.join(sorted(tool.allowed_agents))})")
        if tool.tier == "mutating":
            if tainted:
                # The lethal-trifecta rule (docs/execution-plan.md § 4a): once outside text is in the chat,
                # nothing may change or send data, whatever the model asks.
                raise ToolDenied(f"{name} changes data, and this chat has read untrusted content, so it is refused")
            raise ToolDenied(f"{name} changes data and needs your approval, which arrives in Phase 6")
        return tool

    async def call(self, agent: str, name: str, *, tainted: bool = False, **args: Any) -> ToolResult:
        """Run tool `name` on behalf of `agent` (steps 1–4 in the file header).

        Args:
            agent:    the calling agent's name, checked against the tool's allow-list
            name:     the tool to run
            tainted:  the chat's `tainted` flag; a tainted chat can't run mutating tools
            **args:   the tool's own arguments

        Raises ToolDenied if the checks fail. Example:
            result = await tools.call("rag_agent", "search_knowledge", query="sponsor rules")
            result.data.hits   → the hits, for our code        result.text → wrapped, for the model
        """
        # 1. Check.
        tool = self.check(agent, name, tainted)

        # 2. Run. `asyncio.to_thread` runs the blocking function in a worker thread, so the server stays
        #    free for other requests meanwhile (the knowledge search, for one, briefly uses the CPU).
        value = await asyncio.to_thread(tool.fn, **args)

        # 3 + 4. Render and wrap.
        text = render_pieces(to_pieces(name, value), tool.untrusted_output)
        return ToolResult(tool=name, ok=True, data=value, text=text, untrusted=tool.untrusted_output)
