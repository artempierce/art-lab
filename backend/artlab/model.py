"""
model.py — chooses which chat model the agents talk to, and what it costs.

There are two options, and the rest of the app can't tell them apart:

  * Real Claude (`claude-haiku-4-5`), through LangChain's `ChatAnthropic` wrapper. Costs money.
  * `FakeChatModel`, a free stand-in with predictable behaviour. The tests use it, and you can click
    through the whole UI with it by setting ARTLAB_FAKE_LLM=1 in .env.

Both are LangChain chat models with the same methods (`ainvoke`, `with_structured_output`, …), so
the graph, the API and the web page run exactly the same code path either way. With the fake, the
knowledge-base search is still completely real (local embeddings) — only the *writing* is faked.

This file also owns the price table and the cost formula, so every dollar figure in the app
(trace panel footer, per-chat budget) comes from one place.
"""

import itertools
import json
import os
import re
from typing import Any, get_args

import pydantic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field, PrivateAttr

# The cheapest current Claude model. Change it here and everything follows.
MODEL_ID = "claude-haiku-4-5"

# Anthropic list price in US dollars per 1,000,000 tokens: (input tokens, output tokens).
PRICE_PER_MTOK = (1.00, 5.00)

# What the fake model says, so it's obvious in the UI that no real call happened.
FAKE_REPLY = "This is the fake model talking. No API call was made, so this reply cost $0."

# Words that make the fake supervisor send a question to rag_agent (a crude stand-in for Claude's judgement).
KNOWLEDGE_HINTS = re.compile(
    r"\b(our|policy|policies|rules?|guidelines?|requirements?|checklist|process|sponsor\w*|expenses?|"
    r"moderation|upload\w*|calendar|approv\w*|brand|studio|architecture|personas?)\b",
    re.IGNORECASE,
)

# T10 (Phase 5): the same crude trick, one hint per new worker. Checked in this order, before
# KNOWLEDGE_HINTS, in `_pick_route` below.
ENGLISH_COACH_HINTS = re.compile(
    r"\b(grammar|polish|proofread|rephrase|spelling)\b|\bfix (?:the|this) (?:text|sentence|wording)\b",
    re.IGNORECASE,
)
YOUTUBE_RESEARCHER_HINTS = re.compile(
    r"\b(trend|trending|comments?|viewers?|competitors?)\b|what'?s popular",
    re.IGNORECASE,
)
CONTENT_IDEATOR_HINTS = re.compile(r"\b(ideas?|hooks?|outline|brainstorm)\b", re.IGNORECASE)

# Phase 7 (M1, docs/contracts.md § 11): the fake `remember` extractor's whole "understanding" of a
# message — a stand-in for a real model reading the sentence and deciding what's durable. Catches
# "my niche is budget desk gear" -> key "niche", value "budget desk gear"; a message with no "my X is
# Y" shape yields no facts at all, same as a real model finding nothing worth remembering.
FACTS_PATTERN = re.compile(r"\bmy (\w+(?: \w+)?) is ([^.,!?]+)", re.IGNORECASE)

# Phase 6 (docs/contracts.md § 10): which of `run_tool_loop`'s tools the fake model only calls when the
# task actually mentions it — a tool with no entry here keeps § 9's old "always call it" behaviour.
# Without this, content_ideator's save_ideas would get "called" on every single turn, including "give
# me 3 ideas", which never asks to save anything. Phase 8 (§ 12) made `_reply_with_tools` walk every
# bound tool in order instead of only ever looking at the first one, so a worker whose first tool's
# hint doesn't match (save_ideas) can still fall through to a later one that does (load_skill, below).
FAKE_TOOL_HINTS: dict[str, re.Pattern] = {"save_ideas": re.compile(r"\bsave\b", re.IGNORECASE)}

# Phase 8 (docs/contracts.md § 12): a crude stand-in for the model recognising which skill a task
# needs, since the fake can't read a loaded skill's own description the way a real model would. The
# fake's load_skill call only fires when one of these patterns matches the task; the first match's
# skill name becomes the `name` argument — no match means "don't load anything", the same as any other
# FAKE_TOOL_HINTS miss. Only hook-formulas is exercised by the app's own scenarios (S9) today.
FAKE_SKILL_HINTS: tuple[tuple[str, re.Pattern], ...] = (
    ("hook-formulas", re.compile(r"\bhooks?\b", re.IGNORECASE)),
)

# The fake router's whole decision table, checked top to bottom (docs/contracts.md § 9, T10): the
# first pattern that matches the question *and* whose route is actually allowed by the schema wins.
# `_pick_route` falls through to "respond" if nothing matches (or nothing matching is allowed).
ROUTE_HINTS: tuple[tuple[str, re.Pattern, str], ...] = (
    ("english_coach", ENGLISH_COACH_HINTS, "fake router: mentions grammar, polish or rephrasing"),
    ("youtube_researcher", YOUTUBE_RESEARCHER_HINTS, "fake router: mentions trends, comments or viewers"),
    ("content_ideator", CONTENT_IDEATOR_HINTS, "fake router: mentions ideas, hooks or an outline"),
    ("rag_agent", KNOWLEDGE_HINTS, "fake router: mentions the studio's docs"),
)


def _allowed_routes(schema: Any) -> set[str] | None:
    """The RouteDecision schema's allowed `next` values, read from whatever shape
    `with_structured_output` handed to `bind_tools` (model.py's own docstring explains why there are
    two): a pydantic class (what `agents/supervisor.py` passes today) or a plain JSON-schema dict (what
    some providers convert it to first). Returns None if the shape isn't recognised, so an unfamiliar
    schema doesn't silently block every route.

    This is what keeps the fake honest: it must never pick a route the schema itself would reject, even
    when a test builds a graph with only some of the real workers (docs/contracts.md § 3's `workers=`).
    """
    if isinstance(schema, type) and issubclass(schema, pydantic.BaseModel):
        field = schema.model_fields.get("next")
        return set(get_args(field.annotation)) if field is not None else None
    if isinstance(schema, dict):
        function = schema.get("function", schema)
        properties = function.get("parameters", function).get("properties", {})
        enum = properties.get("next", {}).get("enum")
        return set(enum) if enum else None
    return None


def _pick_route(question: str, allowed: set[str] | None) -> tuple[str, str]:
    """Choose a route and its reason for the fake supervisor (see `ROUTE_HINTS`).

    The first hint whose pattern matches `question` AND whose route is in `allowed` wins; a matching
    hint whose route isn't allowed is skipped, not taken (so it can fall through to the next hint, and
    ultimately to "respond") — that's what test_phase5_routes.py's schema test checks: a graph built
    with only `respond` registered must still land on "respond" for a grammar request, never on a
    route "RouteDecision" would refuse. `allowed=None` (an unrecognised schema) allows every hint.
    """
    for route, pattern, reason in ROUTE_HINTS:
        if pattern.search(question) and (allowed is None or route in allowed):
            return route, reason
    return "respond", "fake router: general question"


def make_model() -> BaseChatModel:
    """Return the model the app should use, based on the ARTLAB_FAKE_LLM setting.

    Called once, when the API server starts (see `api.create_app`).

      ARTLAB_FAKE_LLM=1   -> the free fake model
      anything else       -> real Claude; `ChatAnthropic` reads ANTHROPIC_API_KEY from the
                             environment, which api.py loads from the .env file
    """
    if os.getenv("ARTLAB_FAKE_LLM") == "1":
        return fake_model()

    # Imported here rather than at the top so fake mode and the tests never load the Anthropic
    # client at all.
    from langchain_anthropic import ChatAnthropic

    # max_tokens is a ceiling on how long one answer may be. It is not a cost: you only pay for
    # tokens actually generated.
    return ChatAnthropic(model=MODEL_ID, max_tokens=16000)


class FakeChatModel(BaseChatModel):
    """A free, predictable stand-in for Claude. It plays five roles:

      supervisor       when asked for a RouteDecision (structured output), it picks a route from
                       `ROUTE_HINTS` — a crude keyword check, in order: english_coach, youtube_researcher,
                       content_ideator, then the original KNOWLEDGE_HINTS → rag_agent, else "respond"
                       (`_pick_route`). It never picks a route the bound schema doesn't actually allow
                       (`_allowed_routes`), so tests that register only some workers still get a real one.
      rag_agent        when its prompt contains <untrusted_retrieval> sources, it quotes the start of
                       source [1] and cites it — so you can see real retrieval results in the UI
      rag_agent retry  when asked for a Rephrase (structured output, after a first search found
                       nothing), it rewords the question by appending " policy" — deterministic, and
                       different enough from the original to plausibly match a second time
      remember         when asked for Facts (structured output, docs/contracts.md § 11), it runs
                       FACTS_PATTERN over the message: a match gives exactly one fact (key = the words
                       before "is", spaces turned to underscores; value = the rest, stripped); no
                       match gives an empty list, same as a real model finding nothing to remember
      tool loop        when bound to a *real* tool list (docs/contracts.md § 9, run_tool_loop): it
                       walks the bound tools in order and calls the first one the task actually wants
                       (FAKE_TOOL_HINTS' plain keyword match, or FAKE_SKILL_HINTS for load_skill,
                       §§ 10, 12), then answers quoting its result — see `_reply_with_tools`. Nothing
                       wants the task → it answers directly, like a tool-less worker
      respond          otherwise, it answers with the next of `replies`, in a loop

    How structured output works (and why `bind_tools` is here): LangChain's `with_structured_output(Schema)`
    turns the schema into a *tool* the model is forced to call, then reads the tool call's arguments
    back as a Schema object. Real Claude does that natively; this fake does it by remembering which
    tool it was bound to (`bind_tools`) and returning a tool call for it. `run_tool_loop` binds a plain
    *list* of tools instead (no forced choice), so `bind_tools` tells the two apart by the keyword
    arguments `with_structured_output` passes along (checked against the installed langchain-core,
    1.6.4): `tool_choice="any"` and `ls_structured_output_format=...`, present only for structured output.
    """

    replies: list[str] = Field(default_factory=lambda: [FAKE_REPLY])
    tool_name: str | None = None  # set by bind_tools: the structured-output schema we must "fill in"
    tool_schema: Any = None  # set by bind_tools: the schema itself, so _pick_route can read its `next`
    tool_specs: list | None = None  # set by bind_tools: a real tool list, from run_tool_loop (§ 9)
    _turn: Any = PrivateAttr(default_factory=itertools.count)  # which reply comes next

    @property
    def _llm_type(self) -> str:
        return "fake"

    def bind_tools(self, tools: list, **kwargs: Any) -> "FakeChatModel":
        """Return a copy bound to `tools`, playing whichever tool-calling role fits how it was called
        (see the class docstring): structured output forces one schema (`tool_name`, `tool_schema`); a
        plain tool list from `run_tool_loop` is remembered whole (`tool_specs`), so `_reply_with_tools`
        can call one."""
        if "ls_structured_output_format" in kwargs:
            tool = tools[0]
            name = tool.__name__ if isinstance(tool, type) else tool["name"]
            return self.model_copy(update={"tool_name": name, "tool_schema": tool})
        return self.model_copy(update={"tool_specs": tools})

    def _reply_with_tools(self, messages: list[BaseMessage]) -> AIMessage:
        """The fake's `run_tool_loop` behaviour (docs/contracts.md §§ 9, 12), for a real tool list.

        If the last message is already a `ToolMessage` (a tool has just answered), quote the start of
        it as the final answer — proving a real tool result reached the model, for $0. Otherwise walk
        `tool_specs` in order and call the first one this task actually wants:

          * `load_skill` (Phase 8, § 12) is special: its argument is filled from FAKE_SKILL_HINTS, not
            the task text — the fake can't read a skill's own description the way a real model would,
            so this stands in for "the model recognised this task needs the hook-formulas skill". No
            pattern matches → this tool doesn't want the task either, same as a hint miss below.
          * any other tool named in FAKE_TOOL_HINTS only wants the task when its pattern matches it
            (docs/contracts.md § 10) — a miss moves on to the *next* spec instead of giving up, which is
            what lets a worker with two hinted tools (content_ideator's save_ideas and load_skill) still
            reach whichever one actually fits.
          * a tool with no hint at all is called unconditionally, filling every required string
            argument with the task text — the fake doesn't know what a good argument looks like, but
            this is deterministic and drives `run_tool_loop` through one real tool call before it
            answers, the same as every tool did before FAKE_TOOL_HINTS existed.

        No spec left that wants this task → this behaves exactly as if no tools were bound, and the
        model just answers with `replies`.
        """
        if messages and isinstance(messages[-1], ToolMessage):
            start = str(messages[-1].content)[:200]
            return AIMessage(f"(Fake model, no API call.) The tool said: {start}")

        task = next(m.content for m in reversed(messages) if isinstance(m, HumanMessage))
        for spec in self.tool_specs:
            function = spec["function"]
            name = function["name"]
            if name == "load_skill":
                skill = next((s for s, pattern in FAKE_SKILL_HINTS if pattern.search(task)), None)
                if skill is None:
                    continue
                args = {"name": skill}
            else:
                hint = FAKE_TOOL_HINTS.get(name)
                if hint is not None and not hint.search(task):
                    continue
                params = function["parameters"]
                args = {n: task for n in params.get("required", []) if params["properties"][n].get("type") == "string"}
            return AIMessage("", tool_calls=[{"name": name, "args": args, "id": "fake-call", "type": "tool_call"}])
        return AIMessage(self.replies[next(self._turn) % len(self.replies)])

    def _reply(self, messages: list[BaseMessage]) -> AIMessage:
        """Decide what to say, based on which role we're playing (see the class docstring)."""
        if self.tool_name == "RouteDecision":
            question = next(m.content for m in reversed(messages) if isinstance(m, HumanMessage))
            route, reason = _pick_route(question, _allowed_routes(self.tool_schema))
            args = {"next": route, "reason": reason, "question": question}
            return AIMessage("", tool_calls=[{"name": self.tool_name, "args": args, "id": "fake-call", "type": "tool_call"}])
        if self.tool_name == "Rephrase":
            question = next(m.content for m in reversed(messages) if isinstance(m, HumanMessage))
            args = {"query": f"{question} policy"}
            return AIMessage("", tool_calls=[{"name": self.tool_name, "args": args, "id": "fake-call", "type": "tool_call"}])
        if self.tool_name == "Facts":
            text = next(m.content for m in reversed(messages) if isinstance(m, HumanMessage))
            match = FACTS_PATTERN.search(text)
            facts = [{"key": match.group(1).strip().lower().replace(" ", "_"), "value": match.group(2).strip()}] if match else []
            args = {"facts": facts}
            return AIMessage("", tool_calls=[{"name": self.tool_name, "args": args, "id": "fake-call", "type": "tool_call"}])
        if self.tool_name:
            raise ValueError(f"the fake model can't fill in {self.tool_name}")
        if self.tool_specs:
            return self._reply_with_tools(messages)

        prompt = str(messages[-1].content)
        source = re.search(r"<untrusted_retrieval[^>]*>\n(.*?)\n</untrusted_retrieval>", prompt, re.DOTALL)
        if source:
            body = source.group(1).split("\n\n", 1)[-1]  # drop the "Title › Heading" line
            return AIMessage(f"(Fake model, no API call.) The knowledge base says: {body[:220].strip()} [1]")
        return AIMessage(self.replies[next(self._turn) % len(self.replies)])

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        """Answer in one piece (used when nothing is streaming)."""
        return ChatResult(generations=[ChatGeneration(message=self._reply(messages))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        """Answer word by word, like a real model streams tokens, so the UI shows text arriving."""
        reply = self._reply(messages)
        if reply.tool_calls:
            call = reply.tool_calls[0]
            yield ChatGenerationChunk(message=AIMessageChunk("", tool_call_chunks=[
                {"name": call["name"], "args": json.dumps(call["args"]), "id": call["id"], "index": 0}
            ]))
            return
        for piece in re.findall(r"\S+\s*|\s+", reply.content):  # words with their spacing, so they rejoin exactly
            chunk = ChatGenerationChunk(message=AIMessageChunk(piece))
            if run_manager:
                run_manager.on_llm_new_token(piece, chunk=chunk)
            yield chunk


def fake_model(replies: list[str] = [FAKE_REPLY]) -> FakeChatModel:
    """A FakeChatModel whose "respond" answers are `replies`, in a loop.

    Example: fake_model(["Hi", "Bye"]) answers "Hi", then "Bye", then "Hi" again, and so on.
    """
    return FakeChatModel(replies=list(replies))


def model_name(model: BaseChatModel) -> str:
    """Human-readable model name for the trace panel.

    `ChatAnthropic` stores its model ID in `.model`; the fake model has no such attribute,
    so it shows as "fake model".
    """
    return getattr(model, "model", "fake model")


def cost_usd(input_tokens: int, output_tokens: int) -> float:
    """Convert token counts into US dollars using PRICE_PER_MTOK.

    Example: 1,200 input + 300 output tokens
             = (1,200 × $1 + 300 × $5) / 1,000,000 = $0.0027
    """
    price_in, price_out = PRICE_PER_MTOK
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000
