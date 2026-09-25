"""
evals/judge.py — the LLM judge that grades a worker's or rag_agent's answer against golden `checks`
(Phase 11, docs/contracts.md § 15).

Used by `evals/run.py` for both the rag suite (one check: "every claim is supported by the cited
sources") and the three worker suites (one check per line in `evals/<worker>_golden.yaml`). The judge
is the same model the app answers with (`model.py`'s `MODEL_ID`), forced to fill in `Verdicts` —
structured output, so grading is a plain list of pass/fail/reason instead of free text to parse.

The judge's own reply is our own model talking *about* the (possibly untrusted) answer under test —
report data only, the way a trace line is; it's never fed back into an agent, so it needs no untrusted
wrapping (CLAUDE.md's security rule is about data agents read, not about a report a person reads).
"""

import pydantic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from artlab.agents.common import tokens_used

# The judge's instructions (docs/contracts.md § 15: "grade only what's asked; when unsure, fail").
# Sent once as a system message, same as every other structured-output caller in this app.
JUDGE_PROMPT = (
    "You are a strict grading judge for an AI assistant's answers. You are given an ANSWER and a "
    "numbered list of CHECKS, each a plain-language requirement the answer must meet. For every "
    "check, in order, decide whether the answer satisfies it: `passed` true or false, and a one "
    "sentence `reason` either way. Grade only what each check actually asks for — never invent extra "
    "requirements — and when a check's answer is unclear or missing, fail it rather than guess."
)


class Verdict(pydantic.BaseModel):
    """One check's grade: the check text itself (so a report can show it without re-reading the
    golden file), whether the answer satisfies it, and why."""

    check: str
    passed: bool
    reason: str


class Verdicts(pydantic.BaseModel):
    """The judge's whole reply: one `Verdict` per check, in the same order they were asked — the
    structured-output schema `model.py`'s fake branch and the real Haiku judge both fill in."""

    items: list[Verdict]


def _prompt(answer: str, checks: list[str]) -> str:
    """Build the judge's human message: the answer under test, then its checks, numbered from 1.

    model.py's fake `Verdicts` branch parses this exact shape ("Answer:\\n...\\n\\nChecks:\\n1. ...") to
    play the judge for free — changing this format also means updating that branch.
    """
    numbered = "\n".join(f"{i}. {check}" for i, check in enumerate(checks, start=1))
    return f"Answer:\n{answer}\n\nChecks:\n{numbered}"


async def grade(model: BaseChatModel, answer: str, checks: list[str]) -> tuple[Verdicts, int, int]:
    """Ask the judge model to grade `answer` against `checks`. Returns (verdicts, input_tokens,
    output_tokens) — the caller turns the tokens into a dollar cost with `model.cost_usd`, the same way
    every other model call in this app reports its cost.

    Example: grade(model, "Here are 3 ideas: ...", ["Provides exactly 3 distinct video ideas"])
             -> (Verdicts(items=[Verdict(check="Provides exactly 3 distinct video ideas", passed=True,
                 reason="...")]), 210, 18)

    If the model's structured output doesn't parse (garbled real output — the fake always parses), every
    check is failed rather than the run crashing: an ungraded check is honestly a failure, never a pass.
    """
    judge = model.with_structured_output(Verdicts, include_raw=True)
    messages = [SystemMessage(JUDGE_PROMPT), HumanMessage(_prompt(answer, checks))]
    out = await judge.ainvoke(messages)
    tokens_in, tokens_out = tokens_used(out["raw"])
    parsed = out["parsed"]
    if parsed is None:
        parsed = Verdicts(items=[Verdict(check=c, passed=False, reason="judge output did not parse") for c in checks])
    return parsed, tokens_in, tokens_out
