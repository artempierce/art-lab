# Art Lab

A small multi-agent **web app** built to learn how production AI agents work — guardrails, routing,
tools, knowledge bases (RAG), human approval, memory, skills and evals — where **every step runs for
real and shows itself** in a live trace panel next to the chat.

> **The idea:** agent architecture is easy to read about and hard to *see*. Diagrams show boxes; they
> don't show a guard blocking a message, a supervisor picking an agent, or a search pulling the right
> paragraph out of your documents. Art Lab is one app where each of those happens in front of you,
> one phase at a time.

**Design book** (PRD, architecture, workflows, user scenarios, build plan, decisions):
[`docs/design.html`](docs/design.html) · published at https://claude.ai/artifact/XLdjd89TAy23yNsNzcS2sj

---

## What's in here

| Piece | What it is |
|-------|-----------|
| `docs/design.html` | The design book — the single source of truth for *what* we build and *in what order* |
| `docs/tickets/RAG.md` | Phase 2 tickets (knowledge base + main agent), with what was built |
| `docs/architecture.md` | The original architecture standard (manifest) this project follows, cleaned up |
| `CLAUDE.md` | Rules for AI coding sessions on this repo (documentation standard, cost rule, phase workflow) |
| `backend/artlab/api.py` | FastAPI server: streams each chat turn to the browser, lists and loads chats |
| `backend/artlab/graph.py` | The agent graph: `guard → supervisor → respond / rag_agent → supervisor` |
| `backend/artlab/guards/input.py` | Input guard (size, injection rules, per-chat budget); the injection rules also scan documents |
| `backend/artlab/model.py` | Real Claude (`claude-haiku-4-5`) or a free fake model; prices and cost formula |
| `backend/artlab/rag/` | Knowledge base: `ingest.py` (files + web pages → chunks), `web.py` (safe fetch), `embeddings.py` (local, free), `knowledge.py` (Chroma + search) |
| `backend/artlab/tools/` | Tool registry (who may call what), untrusted-text wrapper, tool catalog |
| `backend/artlab/config.py` | Where things live on disk |
| `knowledge/` | Sample documents to search: 6 fictional studio policies + a poisoned test note |
| `evals/rag_golden.yaml` | Golden questions: 10 the knowledge base answers, 2 it can't |
| `backend/tests/` | 64 tests on the fake model and local embeddings — no API calls, $0 |
| `frontend/src/` | React app: chat list, chat (with Sources), live trace panel |
| `.env.example` | Settings template: API keys, LangSmith, fake-model switch |

---

## How one message flows through the code

What happens when you ask *"Who needs to approve a $900 equipment purchase?"*:

1. **`ChatView`** → **`App.send()`** adds your bubble and an empty reply, and opens a block in the trace panel.
2. **`api.ts → streamChat()`** POSTs `{message, thread_id}` to `/api/chat` and reads the response as a stream.
3. **`api.py → chat()`** makes a **trace ID** (this run) and uses the **thread ID** (this chat), then runs the graph.
4. The **checkpointer** loads the chat's earlier messages from SQLite, and your message is appended.
5. **`graph.py → guard`** runs **`check_input()`**: size → injection rules → budget. Blocked? A refusal, the run ends, no model is called.
6. **`graph.py → supervisor`** (the *main agent*) asks the model for a **`RouteDecision`**: `rag_agent` (our policies and docs) or `respond` (everything else), with a reason and the question rewritten to stand alone.
7. **`graph.py → rag_agent`** calls **`search_knowledge`** through the **tool registry** (only rag_agent may). Search embeds the question locally, asks Chroma for the closest chunks, drops weak matches and flagged chunks, and returns the top 4 — each **wrapped as untrusted**.
8. rag_agent asks the model to answer **only from those sources**, citing `[1] [2]`; nothing relevant → "I couldn't find that in the knowledge base", with no model call. The sources are attached to the reply.
9. Back at the **supervisor**: someone answered, so it finishes (no model call).
10. **`api.py`** streams it all as it happens: `start` → `trace` / `token` … → `done` (tokens, cost, time, sources).
11. **`App`** grows the reply as tokens arrive, adds trace lines to **`TracePanel`**, and shows **Sources** under the answer.

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
cd backend && uv run pytest       # 64 tests: fake model + local embeddings, no API calls, no cost
cd backend && uv run pytest tests/test_retrieval_eval.py -s   # retrieval report: rank of the right file per golden question
cd frontend && npm run build      # type-check + production build
```

---

## Build status

Each phase ends **working, visible in the trace panel, tested, and documented**. Full plan: design book → Build plan.

| Phase | What | Status |
|-------|------|--------|
| 0 | Web MVP: streaming chat, trace panel, saved chats, LangSmith | ✅ done (real-Claude check pending) |
| 1 | Input guard: size, injection rules, per-chat budget | ✅ done |
| 2 | **Knowledge base + main agent**: ingest files and web pages (injection-scanned), local embeddings + Chroma, supervisor routes to `rag_agent`, cited answers, retrieval eval (hit@4 100%) | ✅ done (real-Claude check pending) |
| 3 | Supervisor v2: more routes, step limit | next |
| 4–13 | Tool gateway, workers, approval, memory, skills, handoffs, caps, evals, dashboards | planned |

---

## Concepts used here

| Term | Meaning in this repo |
|------|---------------------|
| **LangGraph** | Library for writing an agent as a graph of steps. Here: `guard → supervisor → rag_agent → …`. |
| **Node** | One step: an async function that reads the state and returns changes. Example: `rag_agent`. |
| **State** | What the graph remembers about one chat: `messages`, `spent_usd`, `task`, `answered_by`. |
| **Reducer** | The rule for merging a node's changes into state. `messages` appends; `spent_usd` adds. |
| **Checkpointer** | Saves the state after every node. Here: SQLite, so chats survive restarts. |
| **Supervisor** | The main agent: decides which agent answers each message, and finishes when one has. |
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
├── CLAUDE.md                     # Rules for AI coding sessions (docs standard, cost rule, phase workflow)
├── README.md                     # This file
├── .env.example                  # Settings template → copy to .env (git-ignored)
├── docs/
│   ├── design.html               # Design book: PRD, architecture, workflows, scenarios, plan, decisions
│   ├── tickets/RAG.md            # Phase 2 tickets and what was built
│   ├── architecture.md           # Original architecture standard, cleaned up
│   ├── PLAN.md                   # Pointer to the design book
│   └── requirements.md           # Superseded first draft, kept for history
├── knowledge/                    # Documents the knowledge base searches (6 sample policies)
│   └── _security/poisoned-note.md  # Test doc with a blatant and a subtle injection
├── evals/rag_golden.yaml         # Golden questions for retrieval
├── backend/                      # Python 3.12 · uv · FastAPI · LangGraph · Chroma
│   ├── artlab/
│   │   ├── api.py                # HTTP API + SSE streaming
│   │   ├── graph.py              # guard → supervisor → respond / rag_agent
│   │   ├── model.py              # Real Claude or fake model; prices
│   │   ├── config.py             # Paths
│   │   ├── guards/input.py       # Input guard + injection rules
│   │   ├── rag/                  # ingest, web fetch, embeddings, knowledge base + search
│   │   └── tools/                # registry, untrusted wrapper, catalog
│   └── tests/                    # api, guard, ingest, web, tools, golden set, retrieval eval
├── frontend/                     # Vite · React · TypeScript · Tailwind
│   └── src/
│       ├── api.ts                # Backend client + stream reader
│       ├── App.tsx               # Page state + three-pane layout
│       ├── index.css             # Theme: colours, fonts (Geist), dark mode
│       └── components/           # Sidebar · ChatView (+ Sources) · TracePanel
└── data/                         # git-ignored: chats (artlab.db), knowledge base (chroma/), embedding model (models/)
```
