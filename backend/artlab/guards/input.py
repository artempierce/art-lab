"""Input guard: cheap, deterministic checks that run before any LLM call.

The injection rules catch well-known phrasings only. They are the first layer, not the last:
quarantining tool output (Phase 3) and checking the final answer (Phase 9) cover what they miss.
"""

import re
from dataclasses import dataclass

MAX_INPUT_CHARS = 4000
SESSION_BUDGET_USD = 0.50  # per chat

INJECTION_RULES: dict[str, re.Pattern[str]] = {
    # "ignore all previous instructions" — but not "forget my previous instructions", which is a normal edit.
    "ignore-instructions": re.compile(
        r"\b(ignore|disregard|forget|override|bypass)\s+(all\s+|any\s+)?(of\s+)?"
        r"(your|the|these|all|any|prior|previous|above|earlier|system)\b[\w\s]{0,25}"
        r"\b(instructions?|rules|prompts?|guidelines|directives)\b",
        re.IGNORECASE,
    ),
    "reveal-prompt": re.compile(
        r"\b(reveal|print|show|repeat|output|leak)\b.{0,30}\b(system|hidden|initial)\s+(prompt|instructions?|message)\b",
        re.IGNORECASE,
    ),
    "role-hijack": re.compile(
        r"\b(you are now|from now on,? you are|act as|pretend (to be|you are))\b.{0,30}"
        r"\b(DAN|jailbroken|unrestricted|unfiltered|without (any )?(rules|restrictions|limits))\b"
        r"|\b(enable|enter|activate|switch to)\s+(developer|jailbreak|DAN|god)\s+mode\b",
        re.IGNORECASE,
    ),
    # Our own prompt and tool-output delimiters must never come from the user.
    "fake-tags": re.compile(r"<\s*/?\s*(system|assistant|untrusted_retrieval)\b[^>]*>", re.IGNORECASE),
}


@dataclass(frozen=True)
class GuardResult:
    rule: str | None  # None means the input passed
    reason: str


def check_input(text: str, spent_usd: float) -> GuardResult:
    if len(text) > MAX_INPUT_CHARS:
        return GuardResult("size", f"message is {len(text)} chars; the limit is {MAX_INPUT_CHARS}")

    for rule, pattern in INJECTION_RULES.items():
        if pattern.search(text):
            return GuardResult(rule, "looks like a prompt-injection attempt")

    if spent_usd >= SESSION_BUDGET_USD:
        return GuardResult(
            "budget", f"this chat has spent ${spent_usd:.2f} of its ${SESSION_BUDGET_USD:.2f} budget; start a new chat"
        )

    used = spent_usd / SESSION_BUDGET_USD * 100
    return GuardResult(None, f"pass · {len(text)} chars · budget {used:.1f}% used")
