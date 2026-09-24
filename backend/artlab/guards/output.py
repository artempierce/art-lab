"""
guards/output.py — the output guard: deterministic checks on what a worker just answered, run once
per turn, after the model has already finished (docs/contracts.md § 14, ticket G1).

Where it runs: the `output_guard` node in graph.py, wired `remember → output_guard → END`. The
step-limit stop paths in `agents/supervisor.py` (the circuit breaker, § 3) go here too instead of
straight to END, so an answer that was cut short by the step limit still gets checked. The `approval`
node's own reply (§ 10) never comes through here — it's our own fixed text, not a model answer.

**"Stream, then retract" (the Phase 10 decision):** the browser already saw every token as it
streamed, before this node ever runs. If this guard finds something wrong, it can't stop that from
having been shown live — what it *can* do is fix the message that gets **saved** (so reopening the
chat later shows the clean version) and tell the browser to swap the bubble's text for the clean one
right now (api.py's `replace` SSE event, sent only when this node found something to fix).

Two things this guard looks for in a worker's answer, checked in this order by `check_output()`:

  1. **A leaked prompt.** Every prompt Art Lab sends a model (Arty's routing and respond prompts,
     rag_agent's, the three Phase 5 workers', the tool loop's TOOL_RULES, and the extraction/summary
     prompts) is OUR OWN internal instruction text — never meant to reach the chat. If an answer
     repeats a long enough chunk of one, word for word (a prompt-injection trick that tricked the
     model into reciting its instructions, or the model just echoing them by accident), that's a
     leak: the whole message is withheld, so nothing of it reaches you.
  2. **A stray internal tag.** `<untrusted_retrieval…>`, `</untrusted_retrieval>`, `<system>` or
     `<assistant>` inside an answer (e.g. the model quoting a tool result a little too literally)
     would confuse the *next* turn's own wrapping (tools/untrusted.py) if it were saved as-is. Only
     the tag itself is removed — the surrounding text is kept.

A third check, the **ideator schema warning**, is warn-only and never redacts anything: content_ideator
promises exactly 3 ideas (agents/content_ideator.py's PROMPT), so an answer with fewer than 3
numbered/bulleted lines gets a note in the trace, in case the studio wants to look at it — it might
still be a perfectly good answer in an unusual shape.
"""

import re
import time
from dataclasses import dataclass

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.config import get_stream_writer

from artlab.agents import content_ideator, english_coach, remember, respond, summarize, supervisor, youtube_researcher
from artlab.agents.common import ms_since, text_of
from artlab.agents.rag_agent import RAG_PROMPT, REPHRASE_PROMPT
from artlab.agents.state import ChatState
from artlab.agents.tool_loop import TOOL_RULES

# What a leaked-prompt hit replaces the WHOLE message with — nothing of the original answer is kept,
# since we can no longer trust any of it once part of it turns out to be our own instructions.
WITHHELD_TEXT = "[Withheld by the output guard: the answer repeated Arty's internal instructions.]"

# A sentence shorter than this is common in an ordinary reply ("Be concise.", "Got it!") and would
# false-positive constantly if it counted as a "leaked prompt" match. 40 characters is long enough
# that an accidental match is implausible — see docs/contracts.md § 14.
MIN_LEAK_CHARS = 40

# Every prompt constant Art Lab sends a model, collected once here so `prompt_sentences()` (below) never
# has to know each module's own constant names again. Adding a new prompt means adding it to this tuple
# — `tests/test_output_guard.py` checks that every module in here contributes at least one sentence, so
# forgetting one fails a test instead of silently leaving a hole in the guard.
_PROMPT_TEXTS: tuple[str, ...] = (
    supervisor.PROMPT_HEADER,
    supervisor.PROMPT_FOOTER,
    respond.RESPOND_PROMPT,
    RAG_PROMPT,
    REPHRASE_PROMPT,
    youtube_researcher.PROMPT,
    content_ideator.PROMPT,
    english_coach.PROMPT,
    TOOL_RULES,
    remember.PROMPT,
    summarize.SUMMARY_PROMPT,
)

# Splits a prompt (or an answer being checked) into sentences on '.', '!', '?' or a newline — a prompt
# constant is written as plain English with line breaks for readability, not one run-on sentence, so
# newlines have to count as breaks too, or "the first line" would never reach MIN_LEAK_CHARS on its own.
_SENTENCE_SPLIT = re.compile(r"[.!?\n]+")

# Our own tag names, opening or closing, with any attributes and any spacing — the same set
# tools/untrusted.py escapes when wrapping, but here they're stripped outright, not escaped: an answer
# is text for you to read, not something that will be re-wrapped, so there's nothing left to protect by
# keeping the tag's literal characters around.
_TAG = re.compile(r"</?\s*(?:untrusted_retrieval|system|assistant)\b[^>]*>", re.IGNORECASE)

# content_ideator's PROMPT promises exactly 3 ideas, each presumably introduced by a number or a
# bullet. Fewer than this many such lines is a warning, not a redaction (see the module docstring).
MIN_IDEAS = 3
_LIST_LINE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.MULTILINE)


def _normalise(text: str) -> str:
    """Lowercase and collapse all whitespace (including newlines) to single spaces, so two sentences
    that differ only in case or line-wrapping compare equal — docs/contracts.md § 14's "handle an
    answer that quotes a prompt sentence with different line wrapping"."""
    return re.sub(r"\s+", " ", text).strip().lower()


def prompt_sentences() -> tuple[str, ...]:
    """Every sentence of at least MIN_LEAK_CHARS characters from every prompt constant in
    `_PROMPT_TEXTS`, normalised (§14's rule 1) and de-duplicated, keeping first-seen order.

    Built fresh each call — there are only a few dozen short prompt constants, so this costs nothing
    worth caching, and a test can call it directly to check every module is represented.
    """
    seen: dict[str, None] = {}
    for prompt in _PROMPT_TEXTS:
        for sentence in _SENTENCE_SPLIT.split(prompt):
            normalised = _normalise(sentence)
            if len(normalised) >= MIN_LEAK_CHARS:
                seen[normalised] = None
    return tuple(seen)


@dataclass(frozen=True)
class OutputVerdict:
    """What `check_output()` decided about one answer.

    ok             False if anything was found (a leak or a stray tag) — never set by the schema
                   warning, which is checked separately and never redacts (see the module docstring).
    reason         None when ok; otherwise "leaked prompt" or "internal tags removed".
    redacted_text  the text to actually save and show: unchanged when ok, `WITHHELD_TEXT` for a leak,
                   or the answer with the tag(s) removed.
    """

    ok: bool
    reason: str | None
    redacted_text: str


def check_output(text: str) -> OutputVerdict:
    """Run the two redacting checks from docs/contracts.md § 14, in order, on one answer's text.

    1. Leaked prompt: does `text`, normalised, contain any normalised sentence from `prompt_sentences()`
       as a substring? A hit withholds the whole message — once part of it is our own instructions, none
       of the rest can be trusted to be a real answer either.
    2. Internal tags: strip `_TAG` matches. Only checked if step 1 found nothing, so a message that's
       already being withheld doesn't also get a (pointless) tag-stripped version.

    Neither hit → `OutputVerdict(ok=True, reason=None, redacted_text=text)`: the answer stands as is.
    """
    normalised = _normalise(text)
    for sentence in prompt_sentences():
        if sentence in normalised:
            return OutputVerdict(ok=False, reason="leaked prompt", redacted_text=WITHHELD_TEXT)

    stripped = _TAG.sub("", text)
    if stripped != text:
        return OutputVerdict(ok=False, reason="internal tags removed", redacted_text=stripped)

    return OutputVerdict(ok=True, reason=None, redacted_text=text)


def _ideator_schema_warning(text: str) -> str | None:
    """None if `text` looks like content_ideator's required 3 ideas (at least MIN_IDEAS lines starting
    with a number or a bullet); otherwise the trace warning to show — never used to change the text."""
    if len(_LIST_LINE.findall(text)) >= MIN_IDEAS:
        return None
    return "schema warning (expected 3 ideas)"


def make_node():
    """Build the `output_guard` node. No arguments: every check here is plain text matching against
    our own prompt constants, so — unlike every other node — it needs no model, no tools and no store.
    """

    async def output_guard(state: ChatState) -> dict:
        """Node — check every answer this turn added, redact what's wrong, and write one trace line.

        Steps:
          1. Find this turn's own answers: every `AIMessage` after the last `HumanMessage` in
             `state["messages"]` — the same "what did THIS turn add" rule `agents/remember.py` uses to
             find your own message, mirrored here for the model's replies instead.
          2. Run `check_output()` on each one. Any that fail are copied with the redacted text, keeping
             the SAME `id` — `add_messages` (the `messages` reducer) then updates that message in place
             rather than appending a new one, so the saved chat shows the redacted version, not both.
          3. If the worker that answered was content_ideator, also run the (non-redacting) schema
             check on its last answer's final text — after step 2's redaction, if any, since a
             withheld message obviously "looks like" nothing close to 3 ideas either.
          4. One trace line, stage "guard": "output ok" when nothing fired, otherwise the reasons and
             the warning joined together. `status` is "blocked" only for an actual redaction — the
             schema warning alone never blocks. api.py reads this exact detail text (it starts with
             "output") to decide whether to send the browser a `replace` event.
        """
        write = get_stream_writer()
        start = time.perf_counter()
        messages = state["messages"]

        last_human = next(
            (i for i in range(len(messages) - 1, -1, -1) if isinstance(messages[i], HumanMessage)), -1
        )
        turn_ai = [m for m in messages[last_human + 1 :] if isinstance(m, AIMessage)]

        changed: list[AIMessage] = []
        reasons: list[str] = []
        last_text = ""
        for message in turn_ai:
            verdict = check_output(text_of(message))
            last_text = verdict.redacted_text
            if not verdict.ok:
                reasons.append(verdict.reason)
                changed.append(message.model_copy(update={"content": verdict.redacted_text}))

        warning = None
        if turn_ai and state.get("answered_by") == "content_ideator":
            warning = _ideator_schema_warning(last_text)

        parts = []
        if "leaked prompt" in reasons:
            parts.append("redacted (leaked prompt)")
        if "internal tags removed" in reasons:
            parts.append("internal tags removed")
        if warning:
            parts.append(warning)
        detail = "output: " + " · ".join(parts) if parts else "output ok"

        write({"stage": "guard", "status": "blocked" if reasons else "ok", "detail": detail, "ms": ms_since(start)})
        return {"messages": changed} if changed else {}

    return output_guard
