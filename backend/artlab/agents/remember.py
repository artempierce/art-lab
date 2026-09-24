"""
agents/remember.py — the last node of a turn: saves any durable facts your message carried
(Phase 7, M1). Runs only once the supervisor decides the turn is done (graph.py: supervisor's "done"
branch goes here before END); the approval and step-limit paths skip straight to END instead — there's
nothing new to learn from either of those.

The safety rule (docs/contracts.md § 11, "the heart of it"): this node sees ONLY the human message you
just typed — the last `HumanMessage` in `state["messages"]` — never an assistant reply, a tool result
or a document. Those can all carry outside text (a poisoned web page, a planted comment), and if
extraction read them, "remember" would become a way to plant a fake memory that every future chat
trusts. Reading just your own words is what makes that impossible.
"""

import time

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from pydantic import BaseModel, Field

from artlab.agents.common import ms_since, text_of, tokens_used
from artlab.agents.state import ChatState
from artlab.memory.store import MemoryStore
from artlab.model import cost_usd

# Shorter than this, a message is never worth a model call ("hi", "ok", "thanks!") — there's nothing
# durable to find, so skip extraction rather than pay for an answer we already know will be empty.
MIN_MESSAGE_CHARS = 15

# The cap from docs/contracts.md § 11: however many facts the model returns, only the first 3 are
# saved. Enforced here in code, not left to the prompt (the same tenet as tools/registry.py hiding a
# tool's defaulted parameters from the model).
MAX_FACTS = 3

PROMPT = """Extract durable facts about the user and their channel from their message below: things like
their niche, audience, tone or posting schedule that stay true across many future chats.

Never extract a request, a question or a one-off task — only facts that describe the user or their
channel. At most 3 facts. If there's nothing durable in the message, return an empty list."""


class Fact(BaseModel):
    """One durable fact, as the model reports it. `key` is normalised (lowercased, capped) again by
    `MemoryStore.save`, so the model doesn't have to get the formatting exactly right."""

    key: str = Field(description="a short name for the fact, e.g. 'niche' or 'posting_schedule'")
    value: str = Field(description="the fact's value, e.g. 'budget desk gear'")


class Facts(BaseModel):
    """The whole extraction result: at most 3 facts, or none. Filled in by the model as *structured
    output* — the same mechanism `agents/supervisor.py`'s `RouteDecision` uses."""

    facts: list[Fact] = Field(default_factory=list, description="at most 3 durable facts found in the message")


def make_node(model: BaseChatModel, store: MemoryStore):
    """Build the `remember` node, closing over the model (for extraction) and the memory store.

    extractor: the model, but forced to answer by filling in `Facts` — built once here, not per call,
    the same way supervisor.py builds its `router`.
    """
    extractor = model.with_structured_output(Facts, include_raw=True)

    async def remember(state: ChatState, config: RunnableConfig) -> dict:
        """Node — extract and save this turn's durable facts, if there are any to look for.

        Steps:
          1. Find the last `HumanMessage` — the message you actually typed this turn. `state["task"]`
             is the supervisor's rewritten standalone question, not your own words, so this reads
             `state["messages"]` directly instead.
          2. Skip, with no model call: the guard flagged this turn (`turn_flagged`), or the message is
             shorter than MIN_MESSAGE_CHARS.
          3. Otherwise, ask the model for `Facts` and save up to MAX_FACTS of them — `MemoryStore.save`
             upserts by key, so re-stating a fact (a new niche, say) just replaces the old value.
          4. Trace: what was saved, "nothing to remember", or which skip rule fired. The model call's
             cost (real Claude; $0 for the fake) is added to `spent_usd` either way it ran.
        """
        write = get_stream_writer()
        start = time.perf_counter()

        message = next(m for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
        text = text_of(message)

        if state.get("turn_flagged"):
            write({"stage": "memory", "status": "ok", "detail": "skipped (flagged message)", "ms": ms_since(start)})
            return {}

        if len(text) < MIN_MESSAGE_CHARS:
            write({"stage": "memory", "status": "ok", "detail": "skipped (too short)", "ms": ms_since(start)})
            return {}

        out = await extractor.ainvoke([SystemMessage(PROMPT), HumanMessage(text)])
        tokens_in, tokens_out = tokens_used(out["raw"])
        spent = cost_usd(tokens_in, tokens_out)
        facts = out["parsed"].facts[:MAX_FACTS] if out["parsed"] else []

        if not facts:
            write({
                "stage": "memory", "status": "ok", "detail": "nothing to remember",
                "ms": ms_since(start), "input_tokens": tokens_in, "output_tokens": tokens_out,
            })
            return {"spent_usd": spent}

        thread_id = config["configurable"]["thread_id"]
        for fact in facts:
            store.save(fact.key, fact.value, thread_id)

        saved = ", ".join(f"{fact.key} = {fact.value}" for fact in facts)
        write({
            "stage": "memory", "status": "ok", "detail": f"saved {saved}",
            "ms": ms_since(start), "input_tokens": tokens_in, "output_tokens": tokens_out,
        })
        return {"spent_usd": spent}

    return remember
