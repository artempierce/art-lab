# Art Lab

A small multi-agent web app for learning production agent patterns: guardrails, routing, tools,
human approval, memory, skills and evals — every step visible in the app's trace panel.

Design book (PRD, architecture, workflows, scenarios, build plan): [`docs/design.html`](docs/design.html)

## Run it

```bash
cp .env.example .env          # starts in free fake-model mode; add keys and set ARTLAB_FAKE_LLM=0 for real Claude

# terminal 1 — API on :8000
cd backend && uv run uvicorn artlab.api:app --reload --port 8000

# terminal 2 — web app on :5173
cd frontend && npm install && npm run dev
```

Open http://localhost:5173.

## Test

```bash
cd backend && uv run pytest     # uses the fake model: no API calls, no cost
cd frontend && npm run build    # type-check + build
```

## Layout

| Path | What |
|---|---|
| `backend/artlab/api.py` | FastAPI: streams a chat turn as server-sent events, lists chats |
| `backend/artlab/graph.py` | The LangGraph (MVP: guard → llm); each node reports trace events |
| `backend/artlab/model.py` | Real Claude (`claude-haiku-4-5`) or the free fake model |
| `frontend/src/` | React app: chat list, chat, trace panel |
| `data/` | SQLite chat history (git-ignored) |
