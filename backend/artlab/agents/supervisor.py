"""
agents/supervisor.py — node 2 of the graph: Arty, the main agent.

Runs after the guard on every turn. Its job is to decide who answers a new message — itself
(the `respond` node) or a worker (`rag_agent`) — and to notice once a worker has answered, so the
turn can finish. It's the only node that asks the model to fill in structured output (`RouteDecision`).
"""

import time
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END
from langgraph.types import Command
from pydantic import BaseModel, Field

from artlab.agents.common import ms_since, text_of, tokens_used
from artlab.agents.state import ChatState
from artlab.model import cost_usd

# ── Prompts ────────────────────────────────────────────────────────────────────────────────────────
# Sent to the model on each call, never stored in the chat history, so they can change any time
# without rewriting old chats.

SUPERVISOR_PROMPT = """You are Arty, the main agent of Art Lab, an assistant for a YouTube creator's studio.
For the user's latest message, decide who should answer it:
- "rag_agent": questions about the studio's own policies, rules, processes, checklists, requirements or
  project documents - anything that needs the studio's knowledge base.
- "respond": everything else, which you answer yourself - small talk, general knowledge, writing or
  brainstorming help.
Also rewrite the latest message as a standalone question (resolve words like "it" or "that" from the
conversation), and give a one-sentence reason for your choice."""

# How the trace panel names who answered: Arty himself (the `respond` node) or a worker agent.
ANSWERED_BY = {"respond": "Arty", "rag_agent": "rag_agent"}


class RouteDecision(BaseModel):
    """The supervisor's decision for one message. The model fills this in as *structured output*:
    LangChain shows it these fields as a form the model must complete, and Pydantic checks the result
    (e.g. `next` can only be one of the two names — anything else is rejected)."""

    next: Literal["respond", "rag_agent"] = Field(
        description="rag_agent for questions about the studio's own policies, processes, requirements or docs; respond for everything else."
    )
    reason: str = Field(description="One short sentence explaining the choice. Shown in the trace panel.")
    question: str = Field(description="The user's latest message rewritten as a standalone question.")


def make_node(model: BaseChatModel):
    """Build the `supervisor` node, closing over the model and the structured-output router built from it.

    router: the supervisor's model, but it must answer by filling in a RouteDecision.
    include_raw=True also returns the raw reply, which carries the token counts we need for cost.
    """
    router = model.with_structured_output(RouteDecision, include_raw=True)

    async def supervisor(state: ChatState) -> Command[Literal["respond", "rag_agent", "__end__"]]:
        """Node 2 — the main agent: decide who answers, then finish once someone has.

        Two situations:
          A. A worker already answered this turn (`answered_by` is set): finish. No model call —
             knowing "we're done" doesn't need an LLM (tenet 1: deterministic code steers the model).
          B. Nobody has answered yet: ask the model for a RouteDecision, then go to that worker,
             handing it the standalone question in `task`.
        If the model's output doesn't fit RouteDecision, fall back to "respond" instead of failing the turn.
        """
        start = time.perf_counter()
        write = get_stream_writer()

        # A. Done?
        if state.get("answered_by"):
            who = ANSWERED_BY.get(state["answered_by"], state["answered_by"])
            write({"stage": "arty", "status": "ok", "detail": f"done · answered by {who}", "ms": ms_since(start)})
            return Command(goto=END)

        # B. Route. `out` is {"parsed": RouteDecision | None, "raw": the reply, "parsing_error": ...}.
        out = await router.ainvoke([SystemMessage(SUPERVISOR_PROMPT), *state["messages"]])
        decision: RouteDecision | None = out["parsed"]
        if decision is None:
            decision = RouteDecision(next="respond", reason="routing output was invalid; answering directly", question=text_of(state["messages"][-1]))

        tokens_in, tokens_out = tokens_used(out["raw"])
        choice = "answering myself" if decision.next == "respond" else f"→ {decision.next}"
        write({
            "stage": "arty", "status": "ok", "detail": f"{choice} · {decision.reason}",
            "ms": ms_since(start), "input_tokens": tokens_in, "output_tokens": tokens_out,
        })
        return Command(update={"task": decision.question, "spent_usd": cost_usd(tokens_in, tokens_out)}, goto=decision.next)

    return supervisor
