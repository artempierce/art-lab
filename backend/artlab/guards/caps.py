"""
guards/caps.py — turn caps: a turn stops at MAX_TURN_SECONDS or MAX_TURN_USD, whichever it hits
first, keeping whatever answer has already been produced (docs/contracts.md § 14, ticket G2, design
book FR-5: "a run stops at max steps, max tokens or max seconds, with the best result so far").

The step-limit circuit breaker (agents/supervisor.py's MAX_STEPS) already stops a runaway handoff
loop. Turn caps are a second, independent ceiling on the same turn: even a *single* dispatch can run
long (a worker's tool loop can make several model calls) or cost a lot (a big prompt, or a nested
`ask_<callee>` call), so this file caps wall-clock time and dollars directly, instead of counting
dispatches.

Where the numbers come from:
  MAX_TURN_SECONDS = 90   a turn that's still running after a minute and a half feels broken to
                           whoever is waiting on it, whatever it's doing.
  MAX_TURN_USD = 0.05     the "max tokens" budget from FR-5, expressed as a dollar figure because
                           every node already reports cost, not tokens (model.py's cost_usd). At
                           model.py's Haiku price table (PRICE_PER_MTOK), $0.05 is roughly 25,000
                           output tokens — generous for one turn, even a multi-step one.

Who reads these: `check()` is the supervisor's yes/no test, run before every dispatch (a route, a
plan step, a handoff). `limits_from()` turns the same state into a `Limits` a worker's tool loop
(agents/tool_loop.py) can check *inside* its own while loop, before each model call — a worker like
content_ideator can make more than one model call in a single dispatch, so the supervisor's own
per-dispatch check isn't enough on its own.
"""

import time
from dataclasses import dataclass

from artlab.agents.state import ChatState

# See the module docstring for why these values.
MAX_TURN_SECONDS = 90
MAX_TURN_USD = 0.05

# What the chat sees when a cap trips, appended after whatever answer already exists — the same
# "stopped, but keep what we have" wording as supervisor.py's STEP_LIMIT_NOTICE, formatted with
# which cap fired ("time" or "cost").
STOP_NOTICE = "Stopped at the turn's {kind} limit. The answer above is the best result so far."


@dataclass(frozen=True)
class Limits:
    """One turn's remaining budget, as of the moment this was built.

    deadline     a `time.time()` reading: the turn must stop at or before this instant.
    spend_left   dollars left in this turn's cost budget right now. A worker's tool loop spends this
                 down as it runs (agents/tool_loop.py tracks its own running total and compares
                 against this snapshot — it never mutates `Limits` itself, since a dataclass this
                 simple is easiest to reason about as a fixed snapshot, not a shared counter).
    """

    deadline: float
    spend_left: float


def limits_from(state: ChatState) -> Limits:
    """Build this turn's `Limits` from the state the guard set at the turn's start (`turn_started_at`,
    `turn_spent_start`) plus the chat's running total (`spent_usd`).

    `state.get(..., default)` guards every field: a test or an old checkpoint that never went through
    `agents/guard.py` (which is what actually sets these fields each turn) still gets a sensible
    Limits — a deadline effectively "now plus the full budget" and the full spend budget untouched —
    rather than a crash.
    """
    started = state.get("turn_started_at", time.time())
    spent_start = state.get("turn_spent_start", 0.0)
    spent_this_turn = state.get("spent_usd", 0.0) - spent_start
    return Limits(deadline=started + MAX_TURN_SECONDS, spend_left=MAX_TURN_USD - spent_this_turn)


def check(state: ChatState) -> str | None:
    """"time", "cost", or None: whether this turn is over one of the caps right now.

    Checked in this order (time first) only because that's the order the module docstring lists
    them in — a turn that's simultaneously over both caps is still just "stopped", and the caller
    (agents/supervisor.py) only needs one word to build its trace line and stop notice with.

    `now` is read once, up front — not as `state.get(..., time.time())`'s default, which Python
    would evaluate on every call whether or not it's actually used (the guard always sets
    `turn_started_at`, so that default almost never is).
    """
    now = time.time()
    started = state.get("turn_started_at", now)
    if now - started >= MAX_TURN_SECONDS:
        return "time"
    spent_start = state.get("turn_spent_start", 0.0)
    if state.get("spent_usd", 0.0) - spent_start >= MAX_TURN_USD:
        return "cost"
    return None
