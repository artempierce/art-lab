"""
tools/registry.py — the tool gateway: every tool call goes through here, gets checked first, and its
result gets wrapped on the way back.

This is the manifest's "tool execution gateway" (docs/architecture.md, layer 4), growing phase by phase.
Tenet 2 there: a model only *asks* for an action; our code decides whether it happens.

`await tools.call(agent, tool_name, tainted=..., **args)` does, in order:

    1. check    the tool exists
                the calling agent is on the tool's allow-list        e.g. search_knowledge: rag_agent only
                the tool is read-only                                tools that change data ("mutating") need
                                                                     your Approve/Reject click (Phase 6); until
                                                                     then they are refused outright, and in a
                                                                     tainted chat they always will be
    2. run      the tool, in a worker thread, with a per-tool timeout; any exception or timeout from the
                tool gets one retry, and a second failure comes back as a failed `ToolResult` instead of
                raising
    3. render   its return value as `Piece`s (tools/untrusted.py)
    4. scan     every untrusted piece for known injection phrasings on arrival (guards/input.py); a match
                sets `flagged=True` — the text is still returned, flagging reports, it doesn't delete
    5. wrap     every piece in <untrusted_retrieval> if the tool's output is untrusted (the default)

and returns a `ToolResult`: the raw value for our code, and the wrapped text for a model.

Why the gateway wraps, not each tool: the trust boundary is a property of the *system*
(docs/execution-plan.md § 4a). If every tool wrapped its own output, one forgotten call would let outside
text reach a model looking like instructions. Here, a new tool is wrapped without doing anything.

A tool's risk tier, allow-list and untrusted flag are fixed when it is registered. The caller can't pass
in a lower tier — that was design bug #2 in the original manifest, where the caller chose the tier itself.

Contract: docs/contracts.md § 4.
"""

import asyncio
import inspect
import json
from dataclasses import dataclass
from typing import Any, Callable, Literal

from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import create_model

from artlab.guards.input import find_injection
from artlab.rag.ingest import DOCUMENT_SKIP_RULES
from artlab.tools.untrusted import Piece, escape_tags, wrap_untrusted

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
    timeout_s         how long `call` waits before giving up on one try (default 10s)             (T2)
    """

    name: str
    fn: Callable[..., Any]
    tier: Tier
    allowed_agents: frozenset[str]
    description: str
    untrusted_output: bool = True
    timeout_s: float = 10.0


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


def scan_pieces(pieces: list[Piece]) -> bool:
    """The arrival scan (T2): True if any piece's text matches a known injection phrasing.

    Runs the same rules the input guard runs on chat messages (guards/input.py), skipping
    "fake-tags" — the same skip document ingest uses (rag/ingest.py), and for the same reason: the
    gateway always escapes our tag names before wrapping (tools/untrusted.py), so a piece containing
    "<untrusted_retrieval>" can't break out and doesn't need flagging just for that.

    A match only sets `ToolResult.flagged`; the text is still wrapped and returned untouched —
    flagging reports a suspicious result, it doesn't delete it.
    """
    return any(find_injection(piece.text, skip=DOCUMENT_SKIP_RULES) for piece in pieces)


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
        timeout_s: float = 10.0,
    ) -> None:
        """Add a tool. Its tier, allow-list, untrusted flag and timeout can't be changed by callers later."""
        self._tools[name] = Tool(name, fn, tier, frozenset(allowed_agents), description, untrusted_output, timeout_s)

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

    def tools_for(self, agent: str) -> list[Tool]:
        """The tools `agent` may call, in the order they were registered (docs/contracts.md § 9).

        Used to build the model's tool definitions (`specs_for`, below) and, later, by
        `agents/capabilities.py` (X3) to list an agent's tools for the UI's Team panel.
        """
        return [tool for tool in self._tools.values() if agent in tool.allowed_agents]

    def specs_for(self, agent: str) -> list[dict]:
        """`tools_for(agent)` as tool definitions for `model.bind_tools(...)` (docs/contracts.md § 9):
        one dict per tool, in the JSON-schema shape `langchain_core` expects from every model provider.

        The model is only ever shown a tool's *required* parameters: any parameter with a default
        (e.g. `search(query, k=4)`'s `k`) is left out of the schema entirely, so the model can't ask
        for it — our own code always supplies it, keeping control of limits like how many hits come
        back. Built the same way LangChain builds `with_structured_output`'s schema: one Pydantic field
        per required parameter (`pydantic.create_model`, using the function's own type hints), handed
        to `convert_to_openai_tool`. The tool's *name* and *description* then come from the
        registration, not from `create_model`'s anonymous class.

        Example: `KnowledgeBase.search(self, query: str, k: int = TOP_K)` is registered as
        "search_knowledge" — the spec that comes back is named "search_knowledge" and its schema has
        only `query`.
        """
        specs = []
        for tool in self.tools_for(agent):
            required_fields: dict[str, Any] = {}
            for param_name, param in inspect.signature(tool.fn).parameters.items():
                if param_name == "self" or param.default is not inspect.Parameter.empty:
                    continue  # defaulted parameters are hidden from the model (see the docstring above)
                annotation = param.annotation if param.annotation is not inspect.Parameter.empty else str
                required_fields[param_name] = (annotation, ...)
            schema = create_model(tool.name, **required_fields)
            spec = convert_to_openai_tool(schema)
            spec["function"]["name"] = tool.name
            spec["function"]["description"] = tool.description
            specs.append(spec)
        return specs

    async def call(self, agent: str, name: str, *, tainted: bool = False, **args: Any) -> ToolResult:
        """Run tool `name` on behalf of `agent` (steps 1–5 in the file header).

        Args:
            agent:    the calling agent's name, checked against the tool's allow-list
            name:     the tool to run
            tainted:  the chat's `tainted` flag; a tainted chat can't run mutating tools
            **args:   the tool's own arguments

        Raises ToolDenied if the checks fail — a refusal is a decision, not a failure, so it is never
        retried (see `check`). Anything the tool itself raises, or a timeout, gets one retry (T2); if
        the second try also fails, this returns a failed `ToolResult` instead of raising. Example:
            result = await tools.call("rag_agent", "search_knowledge", query="sponsor rules")
            result.data.hits   → the hits, for our code        result.text → wrapped, for the model
        """
        # 1. Check.
        tool = self.check(agent, name, tainted)

        # 2. Run, with one retry (T2). `asyncio.to_thread` runs the blocking function in a worker
        #    thread, so the server stays free for other requests meanwhile (the knowledge search, for
        #    one, briefly uses the CPU). `asyncio.wait_for` gives up waiting after `timeout_s` — but the
        #    thread itself keeps running in the background; Python has no way to kill it. We just stop
        #    waiting for it and move on, which is safe here because tools don't touch shared state.
        error: str | None = None
        for attempt in (1, 2):
            try:
                value = await asyncio.wait_for(asyncio.to_thread(tool.fn, **args), timeout=tool.timeout_s)
            except asyncio.TimeoutError:
                error = f"timeout after {tool.timeout_s}s"
                continue
            except Exception as exc:  # any tool failure (a 500, bad data…) — never crashes the caller
                message = str(exc)[:200]
                error = f"{type(exc).__name__}: {message}"
                continue

            # 3. Render.
            pieces = to_pieces(name, value)
            # 4. Scan (T2): only untrusted output is checked — our own data doesn't need it.
            flagged = tool.untrusted_output and scan_pieces(pieces)
            # 5. Wrap.
            text = render_pieces(pieces, tool.untrusted_output)
            return ToolResult(
                tool=name, ok=True, data=value, text=text, untrusted=tool.untrusted_output,
                attempts=attempt, flagged=flagged,
            )

        # Both tries failed: a failed ToolResult instead of raising, so a worker can keep going.
        # An exception message can carry outside text (an error page echoing the request, say), so for an
        # untrusted tool the message itself is wrapped like any other result; only our own prefix stays outside.
        text = f"[tool {name} failed]\n" + (wrap_untrusted(error, name) if tool.untrusted_output else escape_tags(error))
        return ToolResult(
            tool=name, ok=False, data=None, text=text, untrusted=tool.untrusted_output,
            error=error, attempts=2,
        )
