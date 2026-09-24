"""
agents/guard.py — node 1 of the graph: the input guard.

Runs first on every turn, before any model sees the message. Two layers, in order:

  1. the regex rules (artlab.guards.input.check_input) — cheap, deterministic; a match blocks the
     turn outright with a fixed refusal, so a bad message never costs a model call.
  2. the local injection classifier (artlab.guards.classifier), only once layer 1 has passed. Its
     policy is "reduce privileges", not "block" (docs/contracts.md § 8): a flagged message is still
     answered, but the chat is marked tainted, so data-changing tools are refused in it from then on
     — the same `tainted` flag the tool gateway sets when untrusted retrieval enters a chat.
"""

import asyncio
import time
from typing import Literal

from langchain_core.messages import AIMessage, RemoveMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END
from langgraph.types import Command

from artlab.agents.common import ms_since
from artlab.agents.state import ChatState
from artlab.guards.classifier import THRESHOLD, InjectionClassifier
from artlab.guards.input import check_input


def make_node(classifier: InjectionClassifier | None = None):
    """Build the `guard` node.

    Args:
        classifier: the layer-2 injection classifier to run once the regex layer passes. None (the
                     default) means "off" — not downloaded yet, or a test wants the fast,
                     deterministic path with no classifier at all (docs/contracts.md § 8).
    """

    async def guard(state: ChatState) -> Command[Literal["recall", "__end__"]]:
        """Node 1 — run the input guard on the newest message, then either continue or stop.

        Steps:
          1. the regex check (`check_input`). A match ends the turn right here — the classifier
             never runs on a message layer 1 already blocked.
          2. if layer 1 passed and we have a classifier, score the message. A high score, or the
             classifier raising, taints the chat instead of blocking it (fail-safe: an error must
             never be read as "trusted") but lets the run continue either way — on to `recall`
             (Phase 7, docs/contracts.md § 11), never straight to the supervisor any more.

        `turn_flagged` records the classifier's verdict for this turn alone (unlike `tainted`, which
        stays true for the whole chat): `remember` reads it to skip extracting a memory from a message
        the classifier didn't trust.

        Returns a `Command`, which does two jobs in one return value: `update` changes the state, and
        `goto` picks the next node. The `Literal[...]` type tells LangGraph (and graph drawings) which
        nodes this one can jump to.
        """
        start = time.perf_counter()
        write = get_stream_writer()  # sends trace lines to the "custom" stream

        # The message you just sent is always the last one in the history.
        message = state["messages"][-1]
        result = check_input(message.content, state.get("spent_usd", 0.0))

        if result.rule:
            # Blocked. Tell the trace panel which rule fired and why.
            write({"stage": "guard", "status": "blocked", "detail": f"{result.rule}: {result.reason}", "ms": ms_since(start)})
            # Delete the blocked message from history (RemoveMessage) so it can never reach the model
            # on a later turn either, add a refusal as the assistant's reply, and end the run.
            refusal = AIMessage(f"Blocked by the input guard ({result.rule}): {result.reason}.")
            return Command(update={"messages": [RemoveMessage(id=message.id), refusal], "answered_by": ""}, goto=END)

        # Passed layer 1. Clear last turn's routing notes and step count either way; the classifier
        # (layer 2) only decides whether `tainted` (and `turn_flagged`, its per-turn twin) is also set.
        update = {"task": "", "answered_by": "", "steps": 0, "handoff": "", "turn_flagged": False}
        detail = result.reason  # today's detail text; the classifier appends its own part to it

        if classifier is None:
            detail += " · classifier off"
        else:
            try:
                # score() runs the ONNX model — CPU work — so it runs off the event loop.
                score = await asyncio.to_thread(classifier.score, message.content)
            except Exception as exc:
                detail += f" · classifier failed ({type(exc).__name__}: {exc}) → chat tainted, data-changing tools locked"
                write({"stage": "guard", "status": "flagged", "detail": detail, "ms": ms_since(start)})
                # "guard classifier" (docs/contracts.md § 10) is the source name a later approval card
                # shows, so you know a flagged message — not a tool result — is why it's asking.
                return Command(
                    update={**update, "tainted": True, "taint_sources": ["guard classifier"], "turn_flagged": True},
                    goto="recall",
                )

            if score >= THRESHOLD:
                detail += f" · classifier {score:.2f} ≥ {THRESHOLD:.2f} → chat tainted, data-changing tools locked"
                write({"stage": "guard", "status": "flagged", "detail": detail, "ms": ms_since(start)})
                return Command(
                    update={**update, "tainted": True, "taint_sources": ["guard classifier"], "turn_flagged": True},
                    goto="recall",
                )

            detail += f" · classifier {score:.2f}"

        write({"stage": "guard", "status": "ok", "detail": detail, "ms": ms_since(start)})
        return Command(update=update, goto="recall")

    return guard
