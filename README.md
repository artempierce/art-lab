# Art Lab

A small multi-agent **web app** built to learn how production AI agents work — guardrails, routing,
tools, human approval, memory, skills and evals — where **every step runs for real and shows itself**
in a live trace panel next to the chat.

> **The idea:** agent architecture is easy to read about and hard to *see*. Diagrams show boxes; they
> don't show a guard blocking a message, a supervisor picking an agent, or a tool pausing for your
> approval. Art Lab is one app where each of those happens in front of you, one phase at a time.

**Design book** (PRD, architecture, workflows, user scenarios, build plan, decisions):
[`docs/design.html`](docs/design.html) · published at https://claude.ai/artifact/XLdjd89TAy23yNsNzcS2sj

---

## What's in here

| Piece | What it is |
|-------|-----------|
| `docs/design.html` | The design book — the single source of truth for *what* we build and *in what order* |
| `docs/architecture.md` | The original architecture standard (manifest) this project follows, cleaned up |
| `CLAUDE.md` | Rules for AI coding sessions on this repo (documentation standard, cost rule, phase workflow) |
| `backend/artlab/api.py` | FastAPI server: streams each chat turn to the browser, lists and loads chats |
| `backend/artlab/graph.py` | The LangGraph agent graph — today `guard → llm`; later phases add nodes |
| `backend/artlab/guards/input.py` | Input guard: size limit, prompt-injection rules, per-chat budget |
| `backend/artlab/model.py` | Picks real Claude (`claude-haiku-4-5`) or a free fake model; prices and cost formula |
| `backend/tests/` | Tests that run the whole backend on the fake model — no API calls, $0 |
| `frontend/src/api.ts` | Browser-side client: sends messages, reads the event stream |
| `frontend/src/App.tsx` | The page: owns all state, lays out the three panes |
| `frontend/src/components/` | `Sidebar` (chats) · `ChatView` (messages + input) · `TracePanel` (live run steps) |
| `.env.example` | Settings template: API keys, LangSmith, fake-model switch |

---

## How one message flows through the code

What happens when you type *"Give me 3 video ideas"* and press Enter:

1. **`ChatView`** calls `onSend` → **`App.send()`** adds your bubble and an empty reply bubble, and opens a new block in the trace panel.
2. **`api.ts → streamChat()`** POSTs `{message, thread_id}` to `/api/chat` and starts reading the response as a stream.
3. **`api.py → chat()`** makes a **trace ID** (for this run) and uses the **thread ID** (for this chat), then runs the graph with `graph.astream(...)`.
4. The **checkpointer** loads this chat's earlier messages from SQLite, and your message is appended.
5. **`graph.py → guard`** calls **`check_input()`**: size → injection rules → budget. It writes a trace line. If blocked, it replies with a refusal and the run ends — the model is never called.
6. **`graph.py → llm`** sends the system prompt + history to Claude. Tokens stream out as Claude writes them; when done, it writes a trace line with tokens used and adds the cost to the chat's `spent_usd`.
7. **`api.py`** turns everything into server-sent events as it happens: `start` → `trace` / `token` … → `done` (tokens, cost, time).
8. **`App.onEvent`** handles each event: tokens grow the reply bubble; trace events add lines to **`TracePanel`**; `done` writes the footer and refreshes the chat list.
9. The checkpointer has saved the new state after every step, so the chat survives a reload or a server restart.

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
cd backend && uv run pytest       # 29 tests, fake model, no API calls, no cost
cd frontend && npm run build      # type-check + production build
```

---

## Build status

Each phase ends **working, visible in the trace panel, tested, and documented**. Full plan: design book → Build plan.

| Phase | What | Status |
|-------|------|--------|
| 0 | Web MVP: streaming chat, trace panel, saved chats, LangSmith | ✅ done (real-Claude check pending API keys) |
| 1 | Input guard: size, injection rules, per-chat budget | ✅ done |
| 2 | **RAG agent**: local vector DB of our files, search tool for `rag_agent` only, main agent calls it as a tool, cited answers — tickets in [`docs/tickets/RAG.md`](docs/tickets/RAG.md) | next |
| 3 | Supervisor: routing + circuit breaker | planned |
| 4–13 | Tool gateway, workers, approval, memory, skills, handoffs, caps, evals, dashboards | planned |

---

## Concepts used here

| Term | Meaning in this repo |
|------|---------------------|
| **LangGraph** | Library for writing an agent as a graph of steps. Example: `guard → llm`. |
| **Node** | One step: an async function that reads the state and returns changes. Example: `guard`. |
| **State** | What the graph remembers about one chat. Here: `messages` and `spent_usd`. |
| **Reducer** | The rule for merging a node's changes into state. `messages` appends; `spent_usd` adds. |
| **Checkpointer** | Saves the state after every node. Here: SQLite, so chats survive restarts. |
| **Thread** | One chat. Its ID is the key the checkpointer saves under. |
| **Trace ID** | One run (one message). Same ID in the trace panel and in LangSmith. |
| **Stream modes** | What LangGraph streams out: `messages` = model tokens, `custom` = our trace lines. |
| **SSE** | Server-sent events: one long HTTP response made of small `event:` / `data:` blocks. |
| **Guard** | Deterministic checks that run before the model, so bad input costs $0. |
| **Fake model** | A stand-in that replays canned text. Makes tests and UI work free. |

---

## Repo layout

```
art-lab/
├── CLAUDE.md                     # Rules for AI coding sessions (docs standard, cost rule, phase workflow)
├── README.md                     # This file
├── .env.example                  # Settings template → copy to .env (git-ignored)
├── docs/
│   ├── design.html               # Design book: PRD, architecture, workflows, scenarios, plan, decisions
│   ├── architecture.md           # Original architecture standard, cleaned up
│   ├── tickets/
│   │   └── RAG.md                # Phase 2 tickets RAG-1 … RAG-7
│   ├── PLAN.md                   # Pointer to the design book
│   └── requirements.md           # Superseded first draft, kept for history
├── backend/                      # Python 3.12 · uv · FastAPI · LangGraph
│   ├── artlab/
│   │   ├── api.py                # HTTP API + SSE streaming
│   │   ├── graph.py              # The agent graph (guard → llm)
│   │   ├── model.py              # Real Claude or fake model; prices
│   │   └── guards/
│   │       └── input.py          # Input guard: size, injection rules, budget
│   └── tests/
│       ├── test_api.py           # End-to-end API tests on the fake model
│       └── test_input_guard.py   # Guard rules: 11 attacks blocked, 7 normal messages pass
├── frontend/                     # Vite · React · TypeScript · Tailwind
│   └── src/
│       ├── api.ts                # Backend client + stream reader
│       ├── App.tsx               # Page state + three-pane layout
│       ├── index.css             # Theme: colours, fonts (Geist), dark mode
│       └── components/
│           ├── Sidebar.tsx       # Chat list
│           ├── ChatView.tsx      # Messages + input box
│           └── TracePanel.tsx    # Live run steps
└── data/                         # SQLite chat history (git-ignored, created on first run)
```
