"""
agents/supervisor.py — node 2 of the graph: Arty, the main agent (v2, T1 Phase 3).

Runs after the guard on every turn. Its job is to decide who answers a new message (a worker from
the registry in `agents/workers.py`), to notice once a worker has answered so the turn can finish,
to follow a worker's request that another worker continue (a "handoff"), and to stop the turn if
that keeps happening too long (the step-limit circuit breaker). It's the only node that asks the
model to fill in structured output (`RouteDecision`). Contract: docs/contracts.md § 3.
"""

import time
from typing import Literal

import pydantic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END
from langgraph.types import Command
from pydantic import Field

from artlab.agents.common import ms_since, text_of, tokens_used
from artlab.agents.state import ChatState
from artlab.agents.workers import WorkerSpec
from artlab.model import cost_usd

# ── Prompt ─────────────────────────────────────────────────────────────────────────────────────────
# Sent to the model on each call, never stored in the chat history, so it can change any time without
# rewriting old chats. The persona and closing instructions are fixed; only the bullet list of workers
# (built in `_prompt`) changes with the registry, so a new worker needs no prompt rewrite here.
PROMPT_HEADER = """You are Arty, the main agent of Art Lab, an assistant for a YouTube creator's studio.
For the user's latest message, decide who should answer it:"""
PROMPT_FOOTER = """Also rewrite the latest message as a standalone question (resolve words like "it" or "that" from the
conversation), and give a one-sentence reason for your choice."""

# The circuit breaker: at most this many worker dispatches per turn (one route or handoff each), so
# two workers handing off to each other can't run forever. docs/contracts.md § 3.
MAX_STEPS = 5

# What the chat sees when the breaker trips, appended after whatever answer already exists.
STEP_LIMIT_NOTICE = f"Stopped at the step limit ({MAX_STEPS} steps). The answer above is the best result so far."


def _prompt(workers: tuple[WorkerSpec, ...]) -> str:
    """Build the routing prompt: fixed persona and instructions, one bullet per registered worker."""
    bullets = "\n".join(f'- "{worker.name}": {worker.description}' for worker in workers)
    return f"{PROMPT_HEADER}\n{bullets}\n{PROMPT_FOOTER}"


def _answered_by_name(name: str) -> str:
    """How the trace panel names who answered: Arty himself (the `respond` node) or a worker's own name."""
    return "Arty" if name == "respond" else name


def make_node(model: BaseChatModel, workers: tuple[WorkerSpec, ...]):
    """Build the `supervisor` node from the worker registry.

    `RouteDecision` is built fresh from `workers` with `pydantic.create_model`, so its `next` field
    only accepts registered worker names — but the class keeps the name "RouteDecision", because the
    fake model (`model.py`) keys on that name to know it's being asked to route (docs/contracts.md § 3).

    router: the supervisor's model, but it must answer by filling in a RouteDecision.
    include_raw=True also returns the raw reply, which carries the token counts we need for cost.
    """
    names = tuple(worker.name for worker in workers)
    by_name = {worker.name: worker for worker in workers}
    RouteDecision = pydantic.create_model(
        "RouteDecision",
        next=(Literal[tuple(names)], Field(description=f"Which worker should answer: {', '.join(names)}.")),
        reason=(str, Field(description="One short sentence explaining the choice. Shown in the trace panel.")),
        question=(str, Field(description="The user's latest message rewritten as a standalone question.")),
    )
    router = model.with_structured_output(RouteDecision, include_raw=True)
    prompt = _prompt(workers)

    async def route(messages: list) -> tuple[object, int, int]:
        """Ask the model for a RouteDecision, retrying once if the output doesn't parse.

        If it's still invalid after the retry, fall back to "respond" instead of failing the turn —
        the fallback's own `reason` says so, which lands in the trace line the caller writes.
        Returns (decision, input_tokens, output_tokens), counting BOTH calls when it retried: an
        invalid answer is still billed, so it must count towards the cost and the chat's budget.
        """
        out = await router.ainvoke([SystemMessage(prompt), *messages])
        tokens_in, tokens_out = tokens_used(out["raw"])
        if out["parsed"] is None:
            out = await router.ainvoke([SystemMessage(prompt), *messages])  # one retry (§ 3)
            retry_in, retry_out = tokens_used(out["raw"])
            tokens_in, tokens_out = tokens_in + retry_in, tokens_out + retry_out
        if out["parsed"] is not None:
            return out["parsed"], tokens_in, tokens_out
        fallback = RouteDecision(
            next="respond",
            reason="routing output was invalid twice; answering myself",
            question=text_of(messages[-1]),
        )
        return fallback, tokens_in, tokens_out

    async def supervisor(state: ChatState) -> Command:
        """Node 2 — decide who answers, follow handoffs, and stop at the step limit.

        Checked in order (docs/contracts.md § 3); each dispatch below (routing or a handoff) counts
        once towards the step limit, and its trace line ends with " · step n/{MAX_STEPS}":
          A. A worker answered and isn't asking for another one to continue: finish. No model call —
             knowing "we're done" doesn't need an LLM (tenet 1: deterministic code steers the model).
          B. A worker wants another worker to continue (`handoff`): send it there. An unknown worker
             name ends the turn with an error instead. If the step limit is already reached, stop
             instead of dispatching.
          C. Nobody has answered yet this turn: ask the model for a RouteDecision (with the retry-once
             rule above), then dispatch to that worker. In practice this can never hit the step limit
             itself — the guard resets `steps` to 0 at the start of every turn, and it's only reached
             through B afterwards — but the same check guards it for safety.

        `Command`'s return type isn't parameterised with a `Literal` of worker names here, because
        that set is only known once `workers` is passed in at graph-build time; graph.py declares the
        possible destinations to LangGraph itself, via `add_node(..., destinations=...)`, for the
        graph drawings.
        """
        start = time.perf_counter()
        write = get_stream_writer()
        steps = state.get("steps", 0)

        # A. Done — a worker answered and isn't handing off.
        if state.get("answered_by") and not state.get("handoff"):
            who = _answered_by_name(state["answered_by"])
            write({"stage": "arty", "status": "ok", "detail": f"done · answered by {who}", "ms": ms_since(start)})
            return Command(goto=END)

        # B. A handoff to follow.
        handoff = state.get("handoff", "")
        if handoff:
            if handoff not in by_name:
                write({
                    "stage": "arty", "status": "error",
                    "detail": f"unknown handoff target '{handoff}'", "ms": ms_since(start),
                })
                return Command(goto=END)
            if steps >= MAX_STEPS:
                write({
                    "stage": "arty", "status": "stopped",
                    "detail": f"stopped · step limit reached ({MAX_STEPS} steps)", "ms": ms_since(start),
                })
                return Command(update={"messages": [AIMessage(STEP_LIMIT_NOTICE)]}, goto=END)
            write({
                "stage": "arty", "status": "ok",
                "detail": f"→ {handoff} · handoff · step {steps + 1}/{MAX_STEPS}", "ms": ms_since(start),
            })
            return Command(update={"steps": steps + 1, "answered_by": "", "handoff": ""}, goto=handoff)

        # C. Nobody has answered yet: route (unless the step limit was already reached).
        if steps >= MAX_STEPS:
            write({
                "stage": "arty", "status": "stopped",
                "detail": f"stopped · step limit reached ({MAX_STEPS} steps)", "ms": ms_since(start),
            })
            return Command(update={"messages": [AIMessage(STEP_LIMIT_NOTICE)]}, goto=END)

        decision, tokens_in, tokens_out = await route(state["messages"])
        step_n = steps + 1
        choice = "answering myself" if decision.next == "respond" else f"→ {decision.next}"
        write({
            "stage": "arty", "status": "ok", "detail": f"{choice} · {decision.reason} · step {step_n}/{MAX_STEPS}",
            "ms": ms_since(start), "input_tokens": tokens_in, "output_tokens": tokens_out,
        })
        return Command(
            update={"task": decision.question, "spent_usd": cost_usd(tokens_in, tokens_out), "steps": step_n},
            goto=decision.next,
        )

    return supervisor
