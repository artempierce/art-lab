# Art Lab — Requirements (Phase 2 draft, awaiting sign-off)

> **Superseded 2026-09-23** by the PRD in `docs/design.html`. Kept for history only.

## Job (one sentence)

A CLI-based multi-agent system where a supervisor routes personal requests to one of three
specialist agents (YouTube/content strategy, English learning, book/philosophy discussion) —
built to learn LangGraph orchestration, LangSmith tracing, and LLM eval-harness patterns
firsthand, not to be a production product.

## Why this shape

Learning goal, not a product goal. Scope was deliberately cut from an earlier 5-persona,
web-app idea down to 3 specialists + supervisor, CLI only, because additional personas repeat
the same orchestration pattern without teaching anything new — see the "prove the pattern on
one, then generalize" split rule already used in `personal-knowledge-assistant`. Music
production and gym-coach personas are parked as backlog, not v1.

## Agents

### Supervisor
- **Trigger:** every user message in the CLI session
- **Inputs:** the user message, in-process conversation history (not persisted across process
  restarts in v1)
- **Actions:** classify intent → route to exactly one specialist, or answer directly if none
  match. No specialist-to-specialist handoff, no shared cross-specialist state, in v1.
- **Success criteria:** routes an unambiguous message to the correct specialist in a hand-written
  eval set (start at 10-15 cases, one clearly-worded message per case, one expected route each)
- **Out of scope (v1):** multi-hop chains, ambiguity-clarification turns (pick best guess on
  unclear input), persistence across restarts

### YouTube / content strategist
- **Trigger:** routed by supervisor
- **Inputs:** user message + a stubbed "trends" tool returning fixed fake data
- **Actions:** read-only conversation; suggests content ideas/workflows, citing the stub tool's
  output. No write actions.
- **Success criteria:** given a niche/topic prompt, produces a coherent suggestion that
  references at least one tool call's result
- **Out of scope (v1):** real YouTube/trends API, saved history of past suggestions

### English teacher
- **Trigger:** routed by supervisor
- **Inputs:** user message; one fixed, hard-coded placement quiz (not adaptive)
- **Actions:** administers the quiz → stubbed scoring function → reports an estimated
  grammar/vocab level → suggests one daily task. No write actions.
- **Success criteria:** completing the quiz flow yields a level estimate plus one task
  suggestion
- **Out of scope (v1):** adaptive difficulty, tracking level over time across sessions, grading
  free-text writing

### Book / philosophy
- **Trigger:** routed by supervisor
- **Inputs:** user message; one hard-coded fake "books read" list
- **Actions:** discusses a book from the fake list; on request, appends a highlight/quote to a
  local flat file in the repo's own data directory. This is the only write action in v1.
- **Success criteria:** can discuss one book from the list and successfully save one highlight
  to the file
- **Out of scope (v1):** real reading-tracker integration, RAG over full book text

## Risk tier

**Low.** No real external systems, no financial/health/legal actions. The one write action
(saving a highlight to a local flat file) is low-risk, reversible, and sandboxed to the repo's
own data directory.

## Human review point

You, interactively, every turn — no separate approval gate needed in v1 since there are no
consequential write/external actions yet. Revisit this when real MCP or write-capable tools are
added post-v1.

## Budget & limits

- Cheapest available model by default; never trigger a real paid API call to test without
  asking first (standing rule carried over from `ninja-agent`).
- Step limit: cap tool calls per turn (proposed: 3) to keep loops bounded and predictable.
- No spend-cap infrastructure needed yet — local, manual, single-user. Revisit if this moves
  beyond local use.

## Explicit out of scope (whole system, v1)

- Web or mobile UI (CLI only)
- Any real integration (YouTube API, reading tracker, adaptive testing engine)
- Persistent memory beyond the one flat file for book highlights
- Music-production and gym-coach personas (backlog, not v1)
- Multi-agent handoffs or shared state between specialists

## Open items

- Framework/library choices (LangGraph, LangSmith, eval library, testing approach) — Phase 3
  design, not decided here.
- Whether `judge()`/`health_scores()` from `ninja-agent` get ported into this repo's eval
  harness (they're framework-agnostic pure functions, so reuse is low-risk) — Phase 3.
