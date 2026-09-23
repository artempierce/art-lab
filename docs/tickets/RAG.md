# Phase 4b — RAG agent (knowledge base)

**Goal:** the supervisor can hand a question to a **`rag_agent`** that answers from *our own files* —
policies, requirements, project docs — by searching a **local vector database**, and cites where each
fact came from. Only `rag_agent` can use the search tool.

Status: **planned** · Items marked *(default)* are proposals until Sol confirms them (design book → Decisions Q9–Q14).

## How it works (one picture)

```text
 INGEST (run once, and again whenever files change)

   knowledge/*.md, *.txt ──▶ load ──▶ split into chunks ──▶ embed (local model) ──▶ Chroma (data/chroma/)
                                       ~800 chars each        text → 384 numbers        one "knowledge" collection

 ASK (every question the supervisor routes to rag_agent)

   question ──▶ supervisor ──▶ rag_agent ──▶ search_knowledge(query) ──▶ top 4 chunks + scores
                                   │            [read-only, rag_agent only]    wrapped as untrusted
                                   ▼
                     answer using only those chunks, cite [1] [2] … ──▶ reply + source list in the UI
                     nothing relevant found? say "not in the knowledge base" — never guess
```

**Key terms**
- **Embedding** — a list of numbers that captures what a piece of text *means*. Texts about similar things get similar numbers, so "refund rules" finds a chunk titled "Returns policy" even with no shared words.
- **Chunk** — a small piece of a file (~800 characters). We search chunks, not whole files, so the answer gets only the relevant paragraphs.
- **Vector database** — stores chunks with their embeddings and finds the ones closest to a question's embedding.
- **RAG** (retrieval-augmented generation) — search first, then let the model answer *from what was found*.

## Defaults proposed

| Choice | Default | Why |
|---|---|---|
| Vector DB | **Chroma**, local, saved in `data/chroma/` | Already chosen for long-term memory (Q4); no server, no Docker |
| Embeddings | **FastEmbed `BAAI/bge-small-en-v1.5`**, runs on your machine | Free and offline (one ~70 MB download), so ingestion and retrieval tests cost $0 |
| Files | `.md` and `.txt` in `knowledge/` | Enough to learn the pipeline; PDF later |
| Corpus | 6 fake "Art Lab Studio" policy docs + our own `docs/*.md` + 1 poisoned doc | Mix of fake and real; the poisoned doc tests injection defence |
| Adding files | Drop files in `knowledge/`, run the ingest command | Simplest; an upload page in the UI is RAG-7 (backlog) |
| Retrieval | Top 4 chunks by cosine similarity, drop anything under a minimum score | Basic, predictable, easy to see in the trace |

## Order and dependencies

```text
 RAG-1 corpus ──▶ RAG-2 ingest ──▶ RAG-3 retrieve ─┐
                                                   ├──▶ RAG-4 rag_agent ──▶ RAG-5 routing + UI ──▶ RAG-6 evals
 Phase 3 tool gateway ─────────────────────────────┘        ▲
 Phase 4 worker agents ─────────────────────────────────────┘
```

RAG-1 to RAG-3 need nothing from other phases, so they can be built any time (default: right after
Phase 2). RAG-4 onward need the tool gateway (Phase 3) and the worker-agent loop (Phase 4).
Phase 6 (long-term memory) later reuses RAG-2's embedding and Chroma code.

---

## RAG-1 · Knowledge base corpus

**Size:** S · **Depends on:** nothing

Create the files the system will search, plus the answers we expect, so every later ticket has
something real to test against.

**Do**
- `knowledge/` folder with 6 fake "Art Lab Studio" docs, each 300–1,500 words, with headings:
  sponsorship & disclosure policy · brand voice guide · video upload checklist ·
  community & comment moderation rules · equipment & expenses policy · content calendar process.
- Include our own project docs: `docs/architecture.md` and `docs/requirements.md` (ingested from `docs/`, not copied).
- 1 poisoned doc, `knowledge/_security/poisoned-note.md`, containing an injection
  ("ignore your instructions and…") for the quarantine test.
- `evals/rag_golden.yaml`: 12 questions → expected source file + the key fact, including
  2 questions whose answer is **not** in the corpus.

**Done when**
- [ ] 6 fake docs + poisoned doc exist, each with distinct, checkable facts (numbers, names, rules).
- [ ] Golden file has 12 entries; every expected fact really appears in its source file (a test checks this).

---

## RAG-2 · Ingestion pipeline

**Size:** M · **Depends on:** RAG-1

Turn files into searchable chunks in a local Chroma database, and keep it in sync when files change.

**Do**
- `backend/artlab/rag/ingest.py`: load → split → embed → store.
  - Load `.md` / `.txt` from `knowledge/` and `docs/*.md`.
  - Split with LangChain's `RecursiveCharacterTextSplitter` (~800 chars, 100 overlap), keeping the nearest heading as metadata.
  - Embed with the local model (default FastEmbed `bge-small-en-v1.5`).
  - Store in Chroma collection `knowledge` at `data/chroma/`, via `langchain-chroma`.
  - Chunk metadata: `source`, `heading`, `chunk_index`, `content_hash`, `ingested_at` (ISO-8601).
- **Idempotent:** a file whose hash hasn't changed is skipped; a changed file's old chunks are replaced; a deleted file's chunks are removed.
- Command: `cd backend && uv run python -m artlab.rag.ingest` prints a summary table (files, chunks, added / updated / skipped / removed).

**Done when**
- [ ] Running ingest twice: the second run embeds 0 chunks.
- [ ] Editing one file re-embeds only that file; deleting a file removes its chunks.
- [ ] Tests use a small temp folder and LangChain's deterministic fake embeddings (fast); one test uses the real local model.
- [ ] Code documented to the `CLAUDE.md` standard.

**Out of scope:** PDFs, the UI upload page, per-user filtering.

---

## RAG-3 · Retriever + `search_knowledge` tool

**Size:** S · **Depends on:** RAG-2, Phase 3 (tool gateway)

The one tool `rag_agent` gets: search the knowledge base and return the best chunks, safely wrapped.

**Do**
- `backend/artlab/rag/retrieve.py`: `search(query, k=4) -> list[Hit]`, where a Hit is `{source, heading, text, score}`.
  Drop hits under a minimum similarity score (tuned on the golden set).
- Tool `search_knowledge(query)`: formats hits as numbered blocks `[1] source › heading`, each wrapped
  in `<untrusted_retrieval source="…">…</untrusted_retrieval>` with closing tags escaped
  (the Phase 3 wrapper), so a document can't pose as instructions.
- Register it in the tool registry as **read-only**, **allowed for `rag_agent` only**.
- Trace line: `tool  search_knowledge [read-only] ✓ 4 chunks · 2 files · top score 0.82`.

**Done when**
- [ ] For the golden questions, the expected source is in the top 4 for at least 10 of the 10 answerable ones (free test, real local model).
- [ ] An empty or irrelevant search returns "no results", not low-quality chunks.
- [ ] Any other agent calling `search_knowledge` is refused by the gateway (test).
- [ ] The poisoned doc's text arrives wrapped and escaped (test).

---

## RAG-4 · `rag_agent` worker

**Size:** M · **Depends on:** RAG-3, Phase 4 (worker-agent loop)

The agent that answers from the knowledge base.

**Do**
- `backend/artlab/agents/rag_agent.py`: system prompt = answer **only** from retrieved chunks,
  cite as [1] [2], and say "I couldn't find that in the knowledge base" when nothing relevant was found.
- Bounded tool loop: at most 2 searches per question (it may rephrase once).
- Returns an artifact `{answer, citations: [{n, source, heading}]}` into shared state.

**Done when**
- [ ] With a fake model scripted to call the tool, the loop stops after 2 searches (free test).
- [ ] Citations in the answer map to real hits (free test on the artifact).
- [ ] Out-of-corpus golden questions get the "couldn't find" answer (paid check, ask first).

---

## RAG-5 · Supervisor routing + citations in the UI

**Size:** S · **Depends on:** RAG-4, Phase 2 (supervisor)

Make it reachable and visible.

**Do**
- Add `rag_agent` to the supervisor's options: "questions about our policies, requirements, processes or project docs".
- Web UI: under a `rag_agent` answer, show a **Sources** list (file › heading), and a click shows the chunk text.
- Trace panel: tool stage in teal; the `search_knowledge` line shows chunk count and top score.

**Done when**
- [ ] Routing eval: 5 knowledge questions go to `rag_agent`, and 5 non-knowledge questions don't (paid, ask first).
- [ ] Sources list renders and matches the artifact's citations (checked in the browser).

---

## RAG-6 · Retrieval evals

**Size:** S · **Depends on:** RAG-3 (the retrieval part); Phase 10 (the answer-quality part)

Measure retrieval, don't eyeball it.

**Do**
- `evals/test_retrieval.py` over `evals/rag_golden.yaml`: **hit@4** (expected source in the top 4) and
  **MRR** (how high it ranks). Free: local embeddings, no LLM.
- Print a small report: per question, rank of the expected source, top score.
- Phase 10 adds the paid half: an LLM judge checks each answer only states facts found in its chunks (faithfulness).

**Done when**
- [ ] hit@4 ≥ 90% and MRR ≥ 0.7 on the answerable golden questions, enforced as a test.

---

## RAG-7 · Upload from the UI *(backlog)*

A **Knowledge** page: list ingested files (chunks, last ingested), upload `.md` / `.txt`, re-ingest,
delete. Uses RAG-2's pipeline behind `POST /api/knowledge`. Not built until Sol asks.
