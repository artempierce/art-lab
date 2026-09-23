# Art Lab — rules for AI coding sessions

Art Lab is a **learning project**: a multi-agent web app (FastAPI + LangGraph backend, React frontend)
built one phase at a time so each agent pattern is visible. The owner reads the code to learn from it.

- What to build and in what order: `docs/design.html` (the design book). Read its Build plan before starting a phase.
- How the code fits together: `README.md` → "How one message flows through the code".

## Documentation standard (required for every change)

The owner learns from the code, so it must explain itself. This overrides any general
"keep comments minimal" preference.

1. **Every file** starts with a header comment: what the file is, where it sits in the message flow,
   and the key idea a reader needs (e.g. graph.py draws the graph shape).
2. **Every function / component** has a docstring: what it does, its inputs and outputs, and *why*
   it exists or is shaped that way. Give a small example when it helps (see `model.cost_usd`).
3. **Main logic** gets step-by-step comments: number the steps in the docstring and mark them in
   the code (see `graph.py → llm`). Explain library concepts on first use (reducers, `Command`,
   stream modes, SSE…).
4. **Constants** get a comment saying what they control and why that value.
5. **Tests** have a docstring saying what behaviour they protect and why it matters.
6. Write for a learner: plain words, no jargon without a one-line explanation. Don't comment
   the obvious (`i += 1  # add one`).
7. **Keep the docs in sync** in the same change: README ("What's in here", "How one message flows",
   "Build status", "Repo layout") and the design book (phase status, code layout, risks).

## Cost rule

- Never make a real paid API call (Claude, LangSmith evals) without asking the owner first.
- Tests and UI checks use the fake model (`artlab.model.fake_model`, or `ARTLAB_FAKE_LLM=1`).

## Phase workflow

A phase is done only when all of these are true:

1. It runs, and its steps show up in the trace panel.
2. Tests pass: `cd backend && uv run pytest` and `cd frontend && npm run build`.
3. Code is documented to the standard above; README and design book are updated.
4. The design book is republished (same artifact URL) and the change is committed.

## Commands

```bash
cd backend && uv run pytest                                          # backend tests (free)
cd backend && uv run python -m artlab.rag.ingest                     # (re)build the knowledge base (free, local)
cd backend && uv run uvicorn artlab.api:app --reload --port 8000     # API
cd frontend && npm run dev                                           # web app on :5173
cd frontend && npm run build                                         # type-check + build
```
