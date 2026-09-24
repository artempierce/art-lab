# Art Lab — execution plan: parallel tracks and cheaper models

**What this is:** *how* we build the remaining phases — which tasks can run at the same time, and which
model does each one — so development costs fewer tokens without lowering quality.
**What to build and in what order** stays in the design book (`docs/design.html`); this plan follows it.

Status: Waves 0–1, T3 and Phases 0–5 (Wave 2: F1, T7–T10) done; Phase 6 (approvals: T13a/T13b as P6a/P6b) done on 2026-09-24 · next: Phase 7 (memory).

---

## 1. The idea in one picture

```text
             ┌────────────── Opus (main session) ──────────────┐
             │ decide · design contracts · review · integrate  │
             └───────┬──────────────────────────────┬──────────┘
          writes tight tickets                reviews PR diffs
                     ▼                              ▲
   ┌──────────────────────────────────────────────────────────────┐
   │ Sonnet agents, in parallel, each in its own git worktree,    │
   │ each owning its own files → one branch → one PR → CI green   │
   └──────────────────────────────────────────────────────────────┘
   ┌──────────────────────────────────────────────────────────────┐
   │ Haiku agents: content, golden data, docs sync, mechanical    │
   │ edits — only work a test or a diff can verify                │
   └──────────────────────────────────────────────────────────────┘
```

The expensive model spends its tokens on the few things that need judgement (architecture, security,
integration, review). Everything that can be fully specified goes to a cheaper model, and everything that
can run at the same time does.

---

## 2. Model routing rules

| Model | Use it for | Never for | Why |
|---|---|---|---|
| **Opus** (this main session) | Architecture and contracts; LangGraph control flow (routing, interrupts, handoffs); security code (guards, tool registry, web fetch, SSE protocol); final review of cross-cutting PRs; integration and merges; talking decisions through with Sol | Boilerplate, content, doc syncing | Mistakes here are expensive and subtle; it's worth the price |
| **Sonnet** (sub-agents) | Implementing a ticket that has a written spec: files to touch, interfaces, acceptance tests. Worker agents that copy an existing pattern, tool stubs, React pages, eval runners, a first review pass on PRs | Deciding architecture; touching files it wasn't given | Strong at "build exactly this", about a third of Opus's cost |
| **Haiku** (sub-agents) | Content and data: SKILL.md drafts, golden eval questions, fake tool payloads; README / design-book status syncs; renames; lint fixes; CI tweaks | Security code; anything without a test or diff to check it | Very cheap and fast; safe when the output is checkable |

**Escalation rules**
- A Sonnet agent gets **2 rounds** to turn CI green. Still red → the ticket comes back to Opus with the failing output (not a third blind retry).
- Haiku output is always checked: by a test (e.g. every golden fact must exist in its file), by CI, or by Opus reading the diff.
- Any PR touching `guards/`, `tools/registry.py`, `rag/web.py`, the graph wiring or the SSE protocol gets an Opus review of the **diff** before merge.

---

## 3. Where the tokens go today, and the levers

| Lever | What changes | Saves |
|---|---|---|
| **Ticket = the whole prompt** | Each agent gets: goal, the exact files it owns, interfaces to use, acceptance tests, commands, "follow CLAUDE.md". No open-ended exploring | The biggest saving: agents don't read the whole repo |
| **Contracts first** (task W0.5) | Opus fixes the shared interfaces (state fields, worker signature, trace stages, tool API) *before* fanning out | Agents implement, they don't design; no conflicting designs to reconcile |
| **Split graph.py into modules** (task W0.4) | One file per node (`agents/supervisor.py`, `agents/rag_agent.py`, …); `graph.py` only wires them | Parallel agents stop editing the same file → no merge conflicts, smaller files to read |
| **Git worktrees** | Each agent works in its own copy of the repo (`isolation: "worktree"`) | Agents run at the same time without stepping on each other |
| **Free tests, CI as the checker** | Fake model + local embeddings; CI runs everything on each PR | Nobody (least of all Opus) re-reads code to "see if it works" |
| **Review by diff** | Reviewers get `git diff main...branch`, not the repo | Reviews cost a fraction of a full read |
| **Fresh session per wave** | Start each wave in a new session; `CLAUDE.md`, memory and this plan carry the context | Avoids paying for a huge, stale conversation on every turn (this session is ~80% full) |
| **Turn off the file gate for this repo** | The GateGuard hook blocks the first write of every file, so every new file is written twice. `GATEGUARD_EXEMPT_GLOBS` can exempt `art-lab/` | Roughly halves the tokens spent writing files |
| **Scripted doc edits** | Status changes to the 870-line design book are made by small scripts (by Haiku), not by reading and re-writing the page | Doc syncs become cheap |
| **Screenshots only when the UI changed** | Visual checks only on UI PRs | Screenshots are large |
| **Paid calls only at gates** | Real Claude only for the Wave 0 smoke test and the Phase 11 evals, each with Sol's OK; the eval *judge* is Haiku | Keeps API spend tiny as well |

---

## 4. The waves

Tasks in the same wave run **in parallel** unless marked otherwise. "Owns" = the only files that task may
change (plus its own tests), which is what makes parallel work safe.

### Wave 0 — foundations (short, mostly sequential, Opus + Sol)

| ID | Task | Model | Owns | Done when |
|---|---|---|---|---|
| W0.1 | Merge PR #3; optionally require CI before merging to `main` | Sol | — | `main` has Arty + CI; protection on if wanted |
| W0.2 | Decide the classifier policy: reduce privileges / block / warn | Sol | — | ✅ **Reduce privileges** (2026-09-23), design book Q16 |
| W0.3 | Real-Claude smoke test: 3 messages (< $0.01) | Opus, with Sol's OK | — | Routing, cited answer and "not found" seen on real Claude + in LangSmith |
| W0.4 | **Split `graph.py` into one module per node**, no behaviour change | Sonnet | `backend/artlab/graph.py`, new `backend/artlab/agents/*.py` | Tests pass unchanged; code identical by syntax-tree comparison (as in the docs PR) |
| W0.5 | **Contracts** for Phases 3–10: state fields (`steps`, typed `artifacts`, `pending_approval`), the worker-node signature, how a worker registers a route, trace stage names + colours, tool-registry additions (stubs, retries, timeouts), approval event shape; the **trust boundary** (§ 4a): each tool declares whether its output is untrusted (default: yes), a per-run `tainted` flag, and how taint passes through artifacts | Opus | `docs/contracts.md`, typed stubs, skipped tests | Every Wave 1–3 ticket can be written against it |
| W0.6 | Exempt `art-lab/` from the file gate | Sol | your settings | New files are written once |

### Wave 1 — independent tracks (5 in parallel)

| ID | Task | Phase | Model | Owns | Done when |
|---|---|---|---|---|---|
| T1 | **Supervisor v2**: routes come from a worker registry; step limit counted once per step; invalid output retried once, then falls back | 3 | Sonnet (Opus review) | `agents/supervisor.py`, `tests/test_supervisor.py` | Breaker test (S7) and "hi" answered directly (S11) pass |
| T2 | **Tool gateway**: stub `query_youtube_trends` + `fetch_comments` (with one poisoned comment); retry once; failures become a structured error; per-tool timeout; the gateway itself **wraps and scans every tool result** (tools stop wrapping their own) and **refuses data-changing tools while the run is tainted** (§ 4a) | 4 | Sonnet (Opus review) | `tools/*`, `tests/test_tools*.py` | Mocked 500 → one retry → partial answer (S10); wrapper can't be broken out of; a poisoned stub result taints the run, and a mutating tool is then refused |
| T3 | **Guard layer 2** (local classifier), with the W0.2 policy | 1+ | Sonnet (Opus review) | `guards/classifier.py`, the guard node, `tests/test_classifier.py` | Fake-scorer tests pass; real-model test skipped when the ~740 MB model isn't present (CI stays fast) |
| T4 | **RAG follow-up**: rag_agent may rephrase and search again (max 2 searches) | 2+ | Sonnet | `agents/rag_agent.py`, its tests | Loop stops at 2 with a scripted fake model |
| T5 | **Content**: 3 SKILL.md files; routing golden set (15 cases); golden sets for the 3 workers | 5, 8, 11 | Haiku (Opus spot-check) | `backend/skills/`, `evals/*.yaml` | A test checks every golden file's shape; Opus reads a sample |

### Wave 2 — workers and memory (after T1 + T2; up to 5 in parallel)

| ID | Task | Phase | Model | Owns | Done when |
|---|---|---|---|---|---|
| T7 | `youtube_researcher` (copies the rag_agent pattern; uses T2 stubs) | 5 | Sonnet | `agents/youtube_researcher.py` + tests | Answers from stub data (S1) ✅ |
| T8 | `content_ideator` | 5 | Sonnet | `agents/content_ideator.py` + tests | 3 ideas with hooks from a research artifact ✅ |
| T9 | `english_coach` (no tools) | 5 | Sonnet | `agents/english_coach.py` + tests | Grammar request routed and answered (S4) ✅ |
| T10 | Register T7–T9 as routes; trace colours in the UI | 5 | Sonnet, after T7–T9 | worker registry line, `TracePanel.tsx` | Supervisor routes to all four workers ✅ |
| T11 | **Long-term memory**: facts per user in a Chroma "memory" collection (reuses RAG code); extract after a run, recall before routing; summarise long chats; recalled facts are untrusted (they may come from tainted runs) and pass the same boundary | 7 | Sonnet (Opus review: user filtering) | `memory/*`, context node, tests | S8: niche recalled in a new chat |
| T12 | **Skills**: skills index in worker prompts + `load_skill` tool (uses T5 files) | 8 | Sonnet | `skills` loader, catalog entry, tests | S9: skill loaded only when needed |

### Wave 3 — cross-cutting control flow (Opus-led)

| ID | Task | Phase | Model | Owns | Done when |
|---|---|---|---|---|---|
| T13a | **Approvals protocol**: LangGraph interrupt on mutating tools, resume endpoint, `approval` SSE event, `save_ideas` tool | 6 | Opus | graph wiring, `api.py`, `tools/` | S3 passes: nothing written until Approve ✅ |
| T13b | Approve / Reject card in the UI (from T13a's event spec) | 6 | Sonnet, in parallel with T13a once the spec is written | `ChatView.tsx`, `api.ts` | Card appears, click resumes the run ✅ |
| T14 | **Handoffs + agent-calls-agent**: typed artifacts between workers; ideator calls researcher at depth 1 | 9, 9b | Opus design, Sonnet implementation | graph wiring, agents | S2: research → ideas → polish end to end |
| T15 | **Output guard + token/time caps** | 10 | Sonnet (Opus review) | `guards/output.py`, caps | Cap tests pass |

### Wave 4 — evals and dashboards (3 in parallel)

| ID | Task | Phase | Model | Owns | Done when |
|---|---|---|---|---|---|
| T16 | **Paid eval runner**: routing, faithfulness (LLM judge = Haiku), golden sets from T5; a report | 11 | Sonnet builds; run needs Sol's OK | `evals/` runner | All scenarios pass as a gate |
| T17 | **Memory page**: list / edit / delete facts (API + React) | 12 | Sonnet | memory API, `MemoryPage.tsx` | S12 passes |
| T18 | **Runs page**: persist run summaries; cost, steps, eval results | 13 | Sonnet | runs table + API, `RunsPage.tsx` | Runs listed with cost and LangSmith links |

### Continuous

| ID | Task | Model | When |
|---|---|---|---|
| D1 | Docs sync: README tables, design-book status pills and plan rows, ticket files | Haiku | After each merge (scripted edits) |
| R1 | First-pass PR review (bugs, missing tests, CLAUDE.md doc standard) | Sonnet | Every PR, before the Opus review |

---

## 4a. The trust boundary (applies to every task)

**Rule: anything that doesn't come from Sol or from Art Lab's own code and prompts is untrusted data** —
tool results, web pages, knowledge-base chunks, uploaded files, recalled memory, and any agent output
built from those. Marking it is a property of the **system**, enforced by deterministic code at every
boundary. It is never a tool an agent chooses to call: an injection's whole job is to change what the
model chooses, so a choice can't be the defence.

| Layer | What it does | Where | Status |
|---|---|---|---|
| 1. Scan on arrival | Injection rules (+ the classifier, T3) check outside text as it arrives | ingest; the tool gateway | ingest ✅ · live tool results → T2 |
| 2. Wrap at the boundary | Every tool result goes inside `<untrusted_retrieval>`, our tags escaped | **the gateway, for every tool** — not each tool | partly (each tool wraps its own) → T2 |
| 3. Tell the model | Prompts say tagged content is data, never instructions | agent prompts | rag_agent ✅ · every new worker |
| 4. **Taint → fewer privileges + approval** | Once untrusted content is in a run, data-changing or data-sending tools show Approve/Reject (not refused); the card warns about taint sources | graph state (`tainted`) + the registry + approval interrupt | Phase 6 ✅ · T13a/T13b done |
| 5. Human approval | Data-changing actions wait for Approve | Phase 6 | ✅ |

Layers 1–3 make an injection *less likely to work*; none is a guarantee. Layer 4 makes it *harmless when
it does work*: an agent that has read a poisoned page can't send or change anything without Sol. It's the
"lethal trifecta" rule — never let one run combine private data, untrusted content, and a way to send data
out, without a human in between. (The classifier's "reduce privileges" policy, Q16, is the same idea.)

**What every ticket must respect:** new tools declare `untrusted_output` (default `true`) and never wrap
their own output; new agents treat tagged content as data; uploads (RAG-7) and memory (T11) enter through
the same boundary — no special paths.

## 5. Dependencies

```text
Wave 0:  W0.1 ─ W0.2 ─ W0.3            W0.4 (split graph.py) ──┐
                                       W0.5 (contracts) ───────┤
                                                               ▼
Wave 1:  T1 supervisor v2 ──┐   T3 classifier (needs W0.2)   T4 RAG re-search   T5 content (Haiku)
         T2 tool gateway ───┤                                                        │
                            ▼                                                        │
Wave 2:  T7 researcher ─┐   T11 memory      T12 skills ◀───────────────────────────────┘
         T8 ideator ────┼──▶ T10 register routes
         T9 coach ──────┘
                            ▼
Wave 3:  T13a approvals ──▶ T13b UI card        T14 handoffs + 9b        T15 output guard + caps
                            ▼
Wave 4:  T16 paid evals          T17 Memory page          T18 Runs page
```

**Critical path:** W0.4 + W0.5 → T1 → T7–T10 → T14 → T16. Everything else fits around it.

---

## 6. How an agent is briefed (the ticket template)

Every sub-agent prompt follows this shape, so it never needs to explore:

```text
Task:        <one sentence>
Branch:      <feature/...>   (worktree; open a PR to main when CI is green)
You own:     <exact files>   — change nothing else
Read first:  CLAUDE.md, docs/contracts.md §<n>, <1–3 named files to copy the pattern from>
Interfaces:  <function signatures / state fields / events you must use or provide>
Done when:   <acceptance tests, by name>; `uv run pytest` and `npm run build` pass
Rules:       fake model only, no paid API calls; document to the CLAUDE.md standard;
             2 CI fix rounds max, then stop and report the failure
Report:      PR link + one paragraph: what changed, anything you were unsure about
```

---

## 7. Keeping parallel work from colliding

- **File ownership**: each ticket lists the files it may touch. Two tickets in the same wave never share a file; the only shared line (the worker registry) is edited by one follow-up task (T10).
- **Worktrees**: one per agent, one branch per ticket, one PR per branch.
- **Merge order within a wave**: smallest and least-shared first; later PRs rebase before merging.
- **CI is the gate** for every PR; `main` protection (W0.1) makes that automatic.

---

## 8. Cost control

- Development is free by default: fake model, local embeddings, CI on the fake model.
- Paid moments, each needing Sol's OK: the W0.3 smoke test (< $0.01) and the T16 eval runs (a few cents with a Haiku judge).
- A rough budget per wave comes from the ticket count: Opus spends on W0.5, reviews and T13a/T14; the rest is Sonnet and Haiku.

**Expected split of work:** about 20% Opus (design, security, integration, review), 60% Sonnet
(implementation), 20% Haiku (content, data, docs). Compared with doing every step in one Opus session,
that should cut development tokens by well over half — mostly from not re-reading the repo, not writing
files twice, and not carrying one giant conversation.

---

## 9. Decisions needed before Wave 0

1. Approve this plan (or change the model routing).
2. ~~W0.2: classifier policy — reduce privileges (recommended), block, or warn.~~ Decided: reduce privileges.
3. W0.1: require CI to pass before merging to `main`?
4. W0.6: exempt `art-lab/` from the GateGuard file gate?
5. W0.3: OK to spend < $0.01 on the real-Claude smoke test?
