# Art Lab

A small multi-agent **web app** built to learn how production AI agents work — guardrails, routing,
tools, knowledge bases (RAG), human approval, memory, skills and evals — where **every step runs for
real and shows itself** in a live trace panel next to the chat.

The main agent is **Arty**, an "interface friend": a little retro computer whose screen shows his face,
wearing a red beret and carrying a paintbrush. His face shows what the backend is doing: thinking,
searching the knowledge base, blocked, or happy (with a wink).

> **The idea:** agent architecture is easy to read about and hard to *see*. Diagrams show boxes; they
> don't show a guard blocking a message, a supervisor picking an agent, or a search pulling the right
> paragraph out of your documents. Art Lab is one app where each of those happens in front of you,
> one phase at a time.

**Design book** (PRD, architecture, workflows, user scenarios, build plan, decisions):
[`docs/design.html`](docs/design.html)

---

## What's in here

| Piece | What it is |
|-------|-----------|
| `docs/design.html` | The design book — the single source of truth for *what* we build and *in what order* |
| `docs/execution-plan.md` | *How* we build the remaining phases: parallel waves, and which model (Opus / Sonnet / Haiku) does each task |
| `docs/contracts.md` | The shared interfaces: state fields, node signatures, trace stage names, tool API, trust boundary |
| `docs/tickets/RAG.md` | Phase 2 tickets (knowledge base + main agent), with what was built |
| `docs/architecture.md` | The original architecture standard (manifest) this project follows, cleaned up |
| `CLAUDE.md` | Rules for AI coding sessions on this repo (documentation standard, cost rule, phase workflow) |
| `backend/artlab/api.py` | FastAPI server: streams each chat turn to the browser, lists and loads chats |
| `backend/artlab/graph.py` | Graph assembly: wires together nodes from `backend/artlab/agents/` |
| `backend/artlab/agents/` | Each node in its own file: `state.py`, `common.py`, `guard.py`, `summarize.py` (fold to 1 summary after 30+ messages), `recall.py` (load ≤12 facts), `supervisor.py`, `rag_agent.py`, `respond.py`, `tool_loop.py` (model-driven tool loop), `capabilities.py` (team registry), `workers.py` (worker registry), `remember.py` (extract ≤3 facts); Phase 5 workers: `youtube_researcher.py`, `content_ideator.py`, `english_coach.py` |
| `backend/artlab/guards/` | Input guard: layer 1 (`input.py`: size, injection rules including "disable-safety" regex block, per-chat budget; rules also scan documents) and layer 2 (`classifier.py`: ONNX prompt-injection classifier — "reduce privileges", not block). Output guard and caps: `output.py` (checks answers: leaked prompts → withheld, internal tags → stripped, content_ideator without 3 ideas → warning) and `caps.py` (90 s or $0.05 per turn). |
| `backend/artlab/model.py` | Real Claude (`claude-haiku-4-5`) or a free fake model; prices and cost formula |
| `backend/artlab/rag/` | Knowledge base: `ingest.py` (files + web pages → chunks), `web.py` (safe fetch), `embeddings.py` (local, free), `knowledge.py` (Chroma + search) |
| `backend/artlab/tools/` | Registry (who may call what), tool gateway with per-tool timeout and retry, untrusted wrapper, stub tools, catalog; `agents.py` (agent-call tools like `ask_youtube_researcher`); `save_ideas` (mutating tool, needs approval); `ideas.py` data writer; `load_skill` (reads skill text on demand) |
| `backend/artlab/graph.py` → `approval` node | Approval gate: pauses the run (`interrupt()`) on a recorded data-changing request, and after your click runs exactly the stored arguments |
| `backend/artlab/config.py` | Where things live on disk |
| `backend/artlab/memory/` | Long-term memory: `store.py` (Chroma collection "memory", upsert by key), extraction from your words only (never tool output), flagged messages skipped |
| `backend/skills/` | SKILL.md files: retention-analysis, hook-formulas, style-guide. Read at startup by `backend/artlab/skills/loader.py`; agents see only names and descriptions, and load the full text on demand with `load_skill` |
| `knowledge/` | Sample documents to search: 6 fictional studio policies + a poisoned test note |
| `evals/` | Golden question sets: routing, rag, youtube_researcher, content_ideator, english_coach; eval runners |
| `backend/tests/` | 228 tests on the fake model and local embeddings — no API calls, $0 |
| `frontend/src/` | React app: chat list, chat (with Sources), live trace panel |
| `frontend/src/components/Arty.tsx` | Arty, drawn as an SVG with five moods (idle, thinking, searching, happy, blocked) |
| `frontend/src/components/ApprovalCard.tsx` | Phase 6: Approve / Reject card for mutating tools; shows taint warning and what the tool will do |
| `frontend/src/components/TeamList.tsx` | Phase 5 sidebar section listing available workers (team agents) |
| `.env.example` | Settings template: API keys, LangSmith, fake-model switch |

---

## How one message flows through the code

What happens when you ask *"Who needs to approve a $900 equipment purchase?"*:

1. **`ChatView`** → **`App.send()`** adds your bubble and an empty reply, and opens a block in the trace panel.
2. **`api.ts → streamChat()`** POSTs `{message, thread_id}` to `/api/chat` and reads the response as a stream.
3. **`api.py → chat()`** makes a **trace ID** (this run) and uses the **thread ID** (this chat), then runs the graph.
4. The **checkpointer** loads the chat's earlier messages from SQLite, and your message is appended.
5. **`agents/guard.py`** runs **`check_input()`**: size → injection rules → budget. Blocked? A refusal, the run ends, no model is called. Passed? The local injection **classifier** (`guards/classifier.py`, if it's been downloaded) scores the message; at or above threshold, the chat is marked **tainted** — the message still gets answered, but data-changing tools are refused in this chat from then on.
5b. **`agents/summarize.py`**: if the chat has 30+ messages, fold all but the last 10 into one summary. Trimmed `messages` go to the supervisor.
5c. **`agents/recall.py`**: load up to 12 facts from long-term memory (`memory/store.py`, filtered by owner), wrapped as untrusted and read-only.
6. **`agents/supervisor.py`** — **Arty**, the *main agent* — asks the model for a **`RouteDecision`**: a plan of up to 3 steps (e.g. youtube_researcher → content_ideator → english_coach), with each step's answer passed to the next as an untrusted artifact. Arty can also call another agent nested (Phase 9b, via `ask_youtube_researcher` in `tools/agents.py`), and the cost is shared. Before dispatch, turn caps (90 s or $0.05) are checked; if a limit is hit, the run stops with the best answer so far. The route comes from the **worker registry** in `agents/workers.py`. Step counting starts (max 5 steps).
7. The chosen **worker** calls **`agents/tool_loop.py`** if it needs tools. The tool loop lets the model pick from its allowed tools (max 3 requests per turn, including refused requests), runs each through the **tool gateway** in `tools/registry.py` (which wraps results as untrusted), and finishes with a forced text answer. Agents see a **skills index** (names + descriptions) in their prompt and can load a skill with `load_skill`.
7b. **For data-changing tools** (e.g. `save_ideas`): the gateway refuses to run it directly (`ApprovalRequired`); the worker records the request as `pending_approval`, and the **`approval` node** pauses the run with LangGraph's `interrupt()`. The browser gets an `approval` event and shows a card (with a warning if the chat is tainted); your click calls `POST /api/chat/resume`, and only then does the tool run, with exactly the arguments you saw.
8. Example: **`agents/rag_agent.py`** calls the tool loop to **`search_knowledge`** (only rag_agent may). Search embeds the question locally, asks Chroma for the closest chunks, drops weak matches and flagged chunks, and returns the top 4 — each **wrapped as untrusted** by the gateway. If the first search finds nothing, rag_agent can rephrase and search again (max 2 searches).
9. The worker answers from the tool results; nothing relevant → "I couldn't find that in the knowledge base", with no model call. The sources are attached to the reply.
10. Back at the **supervisor**: someone answered and step count increments, or the agent hands back an artifact for the next worker. Handoffs and step limits are managed here. When done, the supervisor finishes.
11. **`agents/remember.py`**: after the supervisor finishes, extract ≤3 durable facts from *only your message* (not tool output or documents), upsert to long-term memory by key. Flagged messages are skipped.
12. **The `output_guard` node** (`guards/output.py`): checks the answer for leaked system-prompt sentences (withheld if found), internal tags (stripped), and content_ideator without 3 ideas (warning added). If changes are made, streams a `replace` SSE event so the chat bubble updates live with a "Redacted by the output guard" note.
13. **`api.py`** streams it all as it happens: `start` → `trace` / `token` … → `done` (tokens, cost, time, sources).
14. **`App`** grows the reply as tokens arrive, adds trace lines to **`TracePanel`**, and shows **Sources** under the answer.

Getting documents *into* the knowledge base is separate — see **Knowledge base** below.

---

## Setup

Prerequisites: [uv](https://docs.astral.sh/uv/) (Python), Node 20+ and npm.

```bash
cp .env.example .env              # starts in free fake-model mode (ARTLAB_FAKE_LLM=1)
cd backend && uv sync             # Python dependencies
cd ../frontend && npm install     # web dependencies
```

To use real Claude, put your `ANTHROPIC_API_KEY` (and optionally `LANGSMITH_API_KEY`) in `.env` and set
`ARTLAB_FAKE_LLM=0`.

The guard's layer-2 injection classifier (`backend/artlab/guards/classifier.py`) is **off until you
download it** — a one-time, free, ~740 MB fetch of its ONNX export:

```bash
cd backend && uv run python -m artlab.guards.classifier
```

Without it, the guard runs layer 1 (regex) only, exactly as before this feature existed.

## Knowledge base

```bash
cd backend && uv run python -m artlab.rag.ingest                        # knowledge/ + docs/architecture.md + docs/requirements.md
cd backend && uv run python -m artlab.rag.ingest ../my-notes/ https://example.com/guide   # your own files and pages
```

Each file or page is split into ~800-character chunks, **scanned for prompt injections** (a matching
chunk is kept but flagged, and search never returns it), embedded **locally for free** (first run
downloads a ~70 MB model), and stored in Chroma at `data/chroma/`. Re-running only re-embeds what
changed, and files deleted from disk are removed. Web pages are fetched safely: public addresses only,
2 MB and 10 s limits, text pages only. Restart the API after ingesting.

## Run

```bash
# terminal 1 — API on :8000
cd backend && uv run uvicorn artlab.api:app --reload --port 8000

# terminal 2 — web app on :5173
cd frontend && npm run dev
```

Open http://localhost:5173. Chats are stored in `data/artlab.db` — delete it to start fresh.

## Test

```bash
cd backend && uv run pytest       # 228 tests: fake model + local embeddings, no API calls, no cost
                                   # (one test class runs the real classifier and is skipped until it's downloaded)
cd backend && uv run pytest tests/test_retrieval_eval.py -s   # retrieval report: rank of the right file per golden question
cd frontend && npm run build      # type-check + production build
```

## CI

GitHub Actions ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs on **every pull request into
`main` and every push to `main`**. Three jobs run in parallel; merge when all are green:

| Job | Checks |
|-----|--------|
| **Backend tests** | `uv run pytest` — the full suite, fake model + local embeddings (model cached between runs) |
| **Frontend lint + build** | `npm run lint`, then `npm run build` (type-check + production build) |
| **Smoke test** | Starts the real API server, sends a chat message over HTTP (must stream start → guard → tokens → done), sends an injection (must be blocked before any model runs) |

The whole workflow runs with `ARTLAB_FAKE_LLM=1` and LangSmith off: no API keys, no secrets, $0.

**Docs-only PRs skip the three jobs.** A first job, *What changed*, checks the PR's files. If they're all
safe (`README.md`, `CLAUDE.md`, `.gitignore`, `docs/`), the three jobs are skipped, and GitHub counts a
skipped required check as passed. The exceptions are `docs/architecture.md` and `docs/requirements.md`:
they're ingested into the knowledge base, so they still run the tests. So do `knowledge/`, `evals/` and
`backend/skills/`, because tests read them. Pushes to `main` always run everything.

---

## Build status

Each phase ends **working, visible in the trace panel, tested, and documented**. Full plan: design book → Build plan.

| Phase | What | Status |
|-------|------|--------|
| 0 | Web MVP: streaming chat, trace panel, saved chats, LangSmith | ✅ done (real-Claude check pending) |
| 1 | Input guard: size, injection rules, per-chat budget | ✅ done |
| 2 | **Knowledge base + main agent**: ingest files and web pages (injection-scanned), local embeddings + Chroma, supervisor routes to `rag_agent`, cited answers, retrieval eval (hit@4 100%) | ✅ done (real-Claude check pending) |
| 3 | Supervisor v2: worker registry, more routes, step limit circuit breaker | ✅ done |
| 4 | Tool gateway: per-tool timeout, retries, failure handling, stub tools, taint tracking | ✅ done |
| 5 | **Worker agents phase 1**: youtube_researcher (query_youtube_trends, fetch_comments), content_ideator (3 ideas with hooks), english_coach (polish); tool loop (model-driven, max 3 per turn); capabilities registry (GET /api/agents, TeamList sidebar); trace panel keeps runs while switching chats | ✅ done |
| 6 | **Approval gate**: data-changing tools pause the graph and show Approve / Reject; tainted chats ask with a warning instead of refusing | ✅ done |
| 7 | **Long-term memory + summarizing**: automatic extraction ≤3 facts from your words only (after each turn), folding to 1 summary after 30+ messages, recall ≤12 facts after guard | ✅ done |
| 8 | **Skills**: load_skill tool, skills index in prompts, inlined patterns removed from ideator/coach | ✅ done |
| 9 | **Handoffs planned upfront**: supervisor can plan up to 3 steps, each answer passed to the next as untrusted artifact | ✅ done |
| 9b | **Agent calls agent**: content_ideator can call youtube_researcher nested (depth 1, cost shared) via ask_youtube_researcher | ✅ done |
| 10 | **Output guard + caps**: output_guard checks answers (leaked prompts withheld, internal tags stripped, content_ideator without 3 ideas warns); turn caps (90 s or $0.05) stop dispatch with best answer so far | ✅ done |
| 11–13 | Evals, dashboards, memory page | planned |

---

## Concepts used here

| Term | Meaning in this repo |
|------|---------------------|
| **LangGraph** | Library for writing an agent as a graph of steps. Here: `guard → supervisor → rag_agent → …`. |
| **Node** | One step: an async function that reads the state and returns changes. Example: `rag_agent`. |
| **State** | What the graph remembers about one chat: `messages`, `spent_usd`, `task`, `answered_by`. |
| **Reducer** | The rule for merging a node's changes into state. `messages` appends; `spent_usd` adds. |
| **Checkpointer** | Saves the state after every node. Here: SQLite, so chats survive restarts. |
| **Arty (supervisor)** | The main agent: decides who answers each message (himself or rag_agent), and finishes when one has. Shown as `arty` in the trace. |
| **Structured output** | The model fills in a fixed form (`RouteDecision`) that code validates, instead of free text. |
| **RAG** | Retrieval-augmented generation: search your documents first, then answer only from what was found. |
| **Chunk** | A ~800-character piece of a document. We search chunks, so answers get just the relevant part. |
| **Embedding** | 384 numbers capturing what a text means; similar meaning → similar numbers. Computed locally. |
| **Vector database** | Stores chunks with their embeddings and finds the closest ones to a question. Here: Chroma. |
| **Tool registry** | Every tool call is checked: does the tool exist, may this agent use it, is it read-only. |
| **Untrusted wrapper** | Retrieved text goes inside `<untrusted_retrieval>` tags, escaped, so it's read as data, not orders. |
| **Trace ID** | One run (one message). Same ID in the trace panel and in LangSmith. |
| **SSE** | Server-sent events: one long HTTP response made of small `event:` / `data:` blocks. |
| **Fake model** | A free stand-in that routes by keyword and answers predictably. Makes tests and UI work free. |

---

## Repo layout

```
art-lab/
├── .github/workflows/ci.yml      # CI: backend tests, frontend build, smoke test — on every PR and push to main
├── CLAUDE.md                     # Rules for AI coding sessions (docs standard, cost rule, phase workflow)
├── README.md                     # This file
├── .env.example                  # Settings template → copy to .env (git-ignored)
├── docs/
│   ├── design.html               # Design book: PRD, architecture, workflows, scenarios, plan, decisions
│   ├── execution-plan.md         # Parallel waves and model routing (Opus / Sonnet / Haiku)
│   ├── contracts.md              # Shared interfaces: state, signatures, traces, trust boundary
│   ├── tickets/RAG.md            # Phase 2 tickets and what was built
│   ├── architecture.md           # Original architecture standard, cleaned up
│   ├── PLAN.md                   # Pointer to the design book
│   └── requirements.md           # Superseded first draft, kept for history
├── knowledge/                    # Documents the knowledge base searches (6 sample policies)
│   └── _security/poisoned-note.md  # Test doc with a blatant and a subtle injection
├── backend/                      # Python 3.12 · uv · FastAPI · LangGraph · Chroma
│   ├── artlab/
│   │   ├── api.py                # HTTP API + SSE streaming
│   │   ├── graph.py              # Graph assembly: wires nodes from agents/ and tools/
│   │   ├── agents/               # Each node in its own file: state, guard, supervisor, rag_agent, respond, workers registry
│   │   ├── model.py              # Real Claude or fake model; prices
│   │   ├── config.py             # Paths
│   │   ├── guards/               # Input guard + injection rules; later output guard
│   │   ├── rag/                  # ingest, web fetch, embeddings, knowledge base + search
│   │   ├── memory/               # Long-term memory: Chroma collection, extraction and recall
│   │   └── tools/                # registry, gateway (timeout/retry), untrusted wrapper, stubs, catalog
│   ├── skills/                   # SKILL.md files for phases 5+: retention-analysis, hook-formulas, style-guide
│   └── tests/                    # 216 tests: api, guard, agents, tools, RAG, skills, evals, Phase 5 workers
├── evals/                        # Golden question sets and eval runners: routing, RAG, workers
├── frontend/                     # Vite · React · TypeScript · Tailwind
│   └── src/
│       ├── api.ts                # Backend client + stream reader
│       ├── App.tsx               # Page state + three-pane layout
│       ├── index.css             # Theme: colours, fonts (Geist), dark mode
│       └── components/           # Arty (the character) · Sidebar · ChatView (+ Sources) · TracePanel
└── data/                         # git-ignored: chats (artlab.db), knowledge base (chroma/), embedding model (models/)
```
