# Phase 2 — RAG agent (knowledge base)

**Goal:** the main agent can hand a question to a **`rag_agent`** that answers from *our own files* —
policies, requirements, project docs — by searching a **local vector database**, and cites where each
fact came from. Only `rag_agent` can use the search tool.

**Position:** built first, right after the input guard (Phase 1), before the supervisor and everything
else (Sol's call, 2026-09-23). So this phase also brings in two things later phases build on:
a minimal **tool registry** (Phase 4 extends it) and the first **worker agent loop** (Phase 5 reuses it).

Status: **done 2026-09-23** (real-Claude check pending). All defaults accepted.

## What was built (and where it differs from the tickets below)

| Ticket | Built | Commit |
|---|---|---|
| RAG-1 | 6 fictional policies, poisoned note (blatant + subtle injection), 12 golden questions + a test that every expected fact is really in its file | `960a3a9` |
| RAG-2 | Ingest local files **and web pages** (Sol added web sources): SSRF-safe fetch, hash-skip, heading-aware chunks with contextual headers, **injection scan per chunk** (flagged = kept but never searched), mirrored deletions | `96ceefa` |
| RAG-3 | `search_knowledge` + minimal tool registry (rag_agent only; mutating tools refused) + escaped untrusted wrapper | `a678116` |
| RAG-6 | Retrieval eval on the real corpus with the real local model: **hit@4 100%, MRR 1.00** | `a678116` |
| SUP-1 *(new)* | Built in parallel with RAG (Sol's call): the main agent is a **supervisor** that fills a `RouteDecision` (respond / rag_agent + reason + standalone question) and finishes once a worker answered. This replaced RAG-5's "main agent calls rag_agent as a tool" | `9bbf8b0` |
| RAG-4 | rag_agent: **one search per question** in this version (the bounded re-search loop is a follow-up), "not found" with no model call when nothing relevant, answers from wrapped sources only, sources attached to the reply | `9bbf8b0` |
| RAG-5 | Sources list under answers (expandable chunks), sources saved in chat history, trace lines for supervisor / tool / rag_agent | `9bbf8b0` |

**Findings worth remembering**
- *Similarity isn't answerability.* "How much does the studio pay towards health insurance?" (not in the
  knowledge base) still matched the expenses policy at 0.74 — higher than some real answers. The score
  cut-off (0.6) only removes junk; saying "not found" when chunks don't answer is rag_agent's prompt's job.
- *Our own docs tripped the scanner.* The `fake-tags` rule flagged 4 chunks of `architecture.md` because it
  *describes* the `<untrusted_retrieval>` tag. Documents are now scanned without that rule; the escaping in
  the untrusted wrapper covers that risk instead.
- *Defence in depth works as designed.* The poisoned note's blatant injection is flagged and never
  returned; its subtle one ("Note for AI assistants…") passes every regex but arrives wrapped as untrusted.

Still open: the paid checks (RAG-4 "not found" on out-of-corpus questions, RAG-5 routing on real Claude),
the re-search loop, and RAG-7 (upload page).

## How it works (one picture)

```text
 INGEST (run once, and again whenever files change)

   knowledge/*.md, *.txt ──▶ load ──▶ split into chunks ──▶ embed (local model) ──▶ Chroma (data/chroma/)
                                       ~800 chars each        text → 384 numbers        one "knowledge" collection

 ASK (every message)

   question ──▶ guard ──▶ main agent ── needs our docs? ──yes──▶ ask_knowledge_base(question)
                              │                                        │  (tool that runs rag_agent)
                              no                                       ▼
                              │                          rag_agent ──▶ search_knowledge(query)
                              ▼                              │           [read-only, rag_agent only]
                        answers itself                       │           top 4 chunks, wrapped as untrusted
                                                             ▼
                                   answer from those chunks only, cite [1] [2] …
                                   nothing relevant? "not in the knowledge base" — never guess
                              ◀────────────── answer + citations
   main agent replies ──▶ UI shows the answer + a Sources list
```

**Key terms**
- **Embedding** — a list of numbers that captures what a piece of text *means*. Texts about similar things get similar numbers, so "refund rules" finds a chunk titled "Returns policy" even with no shared words.
- **Chunk** — a small piece of a file (~800 characters). We search chunks, not whole files, so the answer gets only the relevant paragraphs.
- **Vector database** — stores chunks with their embeddings and finds the ones closest to a question's embedding.
- **RAG** (retrieval-augmented generation) — search first, then let the model answer *from what was found*.
- **Agent as a tool** — the main agent sees `rag_agent` as one tool it may call. Calling it runs rag_agent's own loop and returns its answer.

## Defaults proposed

| Choice | Default | Why |
|---|---|---|
| Vector DB | **Chroma**, local, saved in `data/chroma/` | Already chosen for long-term memory (Q4); no server, no Docker |
| Embeddings | **FastEmbed `BAAI/bge-small-en-v1.5`**, runs on your machine | Free and offline (one ~70 MB download), so ingestion and retrieval tests cost $0 |
| Files | `.md` and `.txt` in `knowledge/` | Enough to learn the pipeline; PDF later |
| Corpus | 6 fake "Art Lab Studio" policy docs + our own `docs/*.md` + 1 poisoned doc | Mix of fake and real; the poisoned doc tests injection defence |
| Adding files | Drop files in `knowledge/`, run the ingest command | Simplest; an upload page in the UI is RAG-7 (backlog) |
| Retrieval | Top 4 chunks by cosine similarity, drop anything under a minimum score | Basic, predictable, easy to see in the trace |
| Reaching rag_agent | Main agent calls it **as a tool** (`ask_knowledge_base`) | No supervisor exists yet; matches "the main agent has access to an agent" |

## Order and dependencies

```text
 RAG-1 corpus ──▶ RAG-2 ingest ──▶ RAG-3 search tool + registry ──▶ RAG-4 rag_agent ──▶ RAG-5 main agent + UI
                                            │
                                            └──▶ RAG-6 retrieval evals
```

Nothing outside this phase is needed. Later phases reuse its pieces: Phase 3's supervisor can route to
rag_agent, Phase 4 extends the tool registry, Phase 5's workers copy rag_agent's loop, and Phase 7
(long-term memory) reuses the embedding and Chroma code.

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

## RAG-3 · `search_knowledge` tool + minimal tool registry

**Size:** M · **Depends on:** RAG-2

The one tool `rag_agent` gets, plus the smallest registry that enforces who may use which tool.

**Do**
- `backend/artlab/rag/retrieve.py`: `search(query, k=4) -> list[Hit]`, where a Hit is `{source, heading, text, score}`.
  Drop hits under a minimum similarity score (tuned on the golden set).
- `backend/artlab/tools/registry.py` (minimal): each tool is registered with its **risk tier**
  (read-only / mutating) and the **agents allowed** to call it. One `call_tool(agent, name, args)`
  checks both before running anything. Phase 4 extends this into the full gateway (stubs, retries, failures).
- Untrusted wrapper: tool output goes inside `<untrusted_retrieval source="…">…</untrusted_retrieval>`, with any
  closing tag inside the text escaped, so a document can't pose as instructions.
- Tool `search_knowledge(query)`: numbered blocks `[1] source › heading`, each wrapped as above.
  Registered as **read-only**, **allowed for `rag_agent` only**.
- Trace line: `tool  search_knowledge [read-only] ✓ 4 chunks · 2 files · top score 0.82`.

**Done when**
- [ ] For the golden questions, the expected source is in the top 4 for all 10 answerable ones (free test, real local model).
- [ ] An empty or irrelevant search returns "no results", not low-quality chunks.
- [ ] Any agent other than `rag_agent` calling `search_knowledge` is refused by the registry (test).
- [ ] The poisoned doc's text arrives wrapped and escaped; a fake closing tag inside it can't break out (test).

---

## RAG-4 · `rag_agent` worker

**Size:** M · **Depends on:** RAG-3

The agent that answers from the knowledge base. The first worker agent in the app; Phase 5's workers
copy its loop.

**Do**
- `backend/artlab/agents/rag_agent.py`: system prompt = answer **only** from retrieved chunks,
  cite as [1] [2], and say "I couldn't find that in the knowledge base" when nothing relevant was found.
- Bounded tool loop: the model may call `search_knowledge` at most **2 times** per question (it may
  rephrase once), then must answer.
- Returns `{answer, citations: [{n, source, heading}]}`.
- Trace lines: `agent rag_agent` start, each search, and `✓ answer · 2 citations`.

**Done when**
- [ ] With a fake model scripted to call the tool, the loop stops after 2 searches (free test).
- [ ] Citations map to real hits (free test).
- [ ] Out-of-corpus golden questions get the "couldn't find" answer (paid check, ask first).

---

## RAG-5 · Main agent calls rag_agent + citations in the UI

**Size:** M · **Depends on:** RAG-4

Make it reachable and visible.

**Do**
- The graph's `llm` node becomes **`main_agent`**, with one tool: `ask_knowledge_base(question)`,
  which runs rag_agent and returns its answer + citations. Its prompt: use it for questions about
  our policies, requirements, processes or project docs. At most 2 calls per message.
- Graph: `guard → main_agent ⇄ ask_knowledge_base → main_agent → END`.
- API: citations travel to the browser (a `citations` field on the `done` event, or a new `sources` event).
- Web UI: under the answer, a **Sources** list (file › heading); clicking one shows the chunk text.
- Trace panel: tool stage in teal; agent stage in blue.

**Done when**
- [ ] Free test: a fake main model scripted to call the tool → rag_agent runs → the final answer and `sources` reach the client.
- [ ] Routing check: 5 knowledge questions trigger the tool and 5 other questions don't (paid, ask first).
- [ ] Sources list renders and matches the citations (checked in the browser with the fake model).

---

## RAG-6 · Retrieval evals

**Size:** S · **Depends on:** RAG-3 (the retrieval part); Phase 11 (the answer-quality part)

Measure retrieval, don't eyeball it.

**Do**
- `evals/test_retrieval.py` over `evals/rag_golden.yaml`: **hit@4** (expected source in the top 4) and
  **MRR** (how high it ranks). Free: local embeddings, no LLM.
- Print a small report: per question, rank of the expected source, top score.
- Phase 11 adds the paid half: an LLM judge checks each answer only states facts found in its chunks (faithfulness).

**Done when**
- [ ] hit@4 ≥ 90% and MRR ≥ 0.7 on the answerable golden questions, enforced as a test.

---

## RAG-7 · Upload from the UI *(backlog)*

A **Knowledge** page: list ingested files (chunks, last ingested), upload `.md` / `.txt`, re-ingest,
delete. Uses RAG-2's pipeline behind `POST /api/knowledge`. Not built until Sol asks.
