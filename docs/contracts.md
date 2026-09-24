# Art Lab — contracts for Wave 1 (Phases 3–4)

**What this is:** the shared interfaces that parallel tickets are written against (task W0.5 in
`docs/execution-plan.md`). An agent implementing a ticket *uses* these; it doesn't redesign them. If a
contract turns out wrong, the ticket stops and reports back. It doesn't work around it.

Scope: what Wave 1 needs (T1 supervisor v2, T2 tool gateway, T4 RAG re-search, T5 content). Later waves
extend this file (approval events, typed artifacts, memory) before their tickets start.

Status: 2026-09-23 · Wave 0 foundations. The parts marked **(in code)** already exist on
`feat/wave0-foundations`, and everything else is for the named ticket to build.

---

## 1. Chat state (`agents/state.py`)

| Field | Type · reducer | Who writes it | Meaning |
|---|---|---|---|
| `messages` | list · `add_messages` | every node | the conversation (unchanged) |
| `spent_usd` | float · `operator.add` | every node that calls a model | chat's total cost (unchanged) |
| `task` | str · overwrite | supervisor | this turn's standalone question (unchanged) |
| `answered_by` | str · overwrite | workers; guard resets to `""` | who answered this turn (unchanged) |
| `steps` **(in code)** | int · overwrite | supervisor; guard resets to `0` | how many times the supervisor has dispatched a worker **this turn** |
| `handoff` **(in code)** | str · overwrite | workers; guard and supervisor reset to `""` | a worker's request that another worker continue (name, or `""`) |
| `tainted` **(in code)** | bool · `operator.or_` | workers, from `ToolResult.untrusted` | untrusted content has entered this chat |

**`tainted` is sticky for the whole chat**, not per run. Once untrusted text is in `messages` (or in an
answer built from it), every later turn sends it back to the model, so the chat stays tainted. Its
reducer is logical OR: `False` never un-taints. (The execution plan said "per run". This is the
stricter reading, and it only matters once data-changing tools exist in Phase 6.)

## 2. Worker nodes (`agents/<name>.py`)

A **worker** is a node the supervisor can route to (`respond`, `rag_agent`, and the Phase 5 workers).

```python
def make_node(model: BaseChatModel, tools: ToolRegistry) -> Callable[[ChatState], Awaitable[dict]]:
    async def <name>(state: ChatState) -> dict: ...
    return <name>
```

A worker returns a state update with:

- `messages: [AIMessage]`: its answer (sources go in `additional_kwargs["sources"]`)
- `answered_by: "<name>"`: always, so the supervisor knows the step finished
- `spent_usd`: the model cost, if it called a model
- `tainted: True`: if any `ToolResult` it used had `untrusted=True` (omit it otherwise)
- `handoff: "<other worker>"`: only when it wants another worker to continue (no worker does yet; Phase 9 uses it)

It works on `state["task"]` (the standalone question), writes one trace line per action (see § 5), and calls
tools only through `await tools.call(...)` (§ 4).

## 3. Worker registry and supervisor v2 (T1)

**`agents/workers.py`** (T1 creates it) lists every worker. It's the only place a route is added:

```python
@dataclass(frozen=True)
class WorkerSpec:
    name: str          # node name = route value, e.g. "rag_agent"
    description: str   # one line the supervisor's prompt shows: when to route here
    make_node: Callable[[BaseChatModel, ToolRegistry], Node]

WORKERS: tuple[WorkerSpec, ...] = (respond, rag_agent)   # Phase 5 (T10) appends the new workers
```

- `build_graph(model, checkpointer, tools, workers=WORKERS)` adds one node per worker plus an edge back to
  the supervisor. Tests pass their own stub workers.
- The supervisor builds its `RouteDecision` from the registry: `next: Literal[<worker names>]`, and its
  prompt lists each worker's `description`. The schema **keeps the name `RouteDecision`**, because the fake
  model keys on it.
- **Invalid output:** if `parsed` is `None`, ask once more. Still invalid → fall back to `respond`, with a
  trace line that says so.
- **Finishing, counted once per step** (fixes the manifest's double count):

```text
supervisor runs
  ├─ answered_by == ""             → route (model call), steps += 1, go to the worker
  ├─ answered_by set, no handoff   → END                                   (no model call)
  ├─ handoff = a registered worker → steps += 1, go to it; answered_by, handoff reset
  │                                   (an unknown name → END with an "error" trace line)
  └─ steps == MAX_STEPS (5) and more work requested
                                   → END, append "Stopped at the step limit (5 steps). The answer
                                     above is the best result so far.", trace status "stopped"
```

  One dispatch = one step, whether it's the first route or a handoff. Trace detail shows `step n/5`.

## 4. Tool gateway (`tools/registry.py`)

**(in code)** Every tool call goes through `await tools.call(agent, name, *, tainted=False, **args) -> ToolResult`.

```python
@dataclass(frozen=True)
class ToolResult:
    tool: str
    ok: bool                 # False when the tool failed after its retry (T2)
    data: Any                # the tool's raw return value, for code (hits, citations); None if failed
    text: str                # what a model may see: each untrusted piece wrapped in <untrusted_retrieval>
    untrusted: bool          # the tool's registered untrusted_output
    error: str | None = None # e.g. "timeout after 10s", "RuntimeError: 500"; None when ok (T2)
    attempts: int = 1        # 2 when the first try failed (T2)
    flagged: bool = False    # the arrival scan found injection patterns in the text (T2)
```

Registration: `register(name, fn, tier, allowed_agents, description, *, untrusted_output=True)`.

In order, the gateway:
1. **Refuses (raises `ToolDenied`, never retried):** unknown tool, agent not on the allow-list, and
   mutating tools. When a mutating tool is refused in a tainted chat, the message says the chat is
   tainted. Phase 6 turns the untainted case into an approval request. The tainted case stays refused.
2. **Runs** the tool in a worker thread (tools are plain, blocking functions).
3. **Renders** the result as `Piece`s: if the value has `.pieces() -> list[Piece]` it uses them, a `str`
   becomes one piece with `source=<tool name>`, and anything else becomes JSON.
   `Piece(source, text, label="")`: `label` is a short line from *our* code (e.g. `[1] knowledge/x.md › Titles`),
   and it goes **outside** the wrapper so the model can cite it.
4. **Wraps** every piece with `wrap_untrusted(text, source)` when `untrusted_output` is true. **Tools never
   wrap their own output.**

**T2 adds**, inside `call`: a per-tool `timeout_s` (register keyword, default 10) using `asyncio.wait_for`,
**one retry** on any exception or timeout, and then a failed `ToolResult(ok=False, data=None,
text="[tool <name> failed]\n<the error, wrapped if the tool is untrusted>", error=..., attempts=2)` instead
of raising. (An exception message can echo outside text, so it crosses the same boundary as a result.) It also adds the
**arrival scan**: `guards.input.find_injection(piece.text, skip=rag.ingest.DOCUMENT_SKIP_RULES)` on every
untrusted piece, which sets `flagged=True`. The text is still wrapped and still returned; the trace
reports it.

## 5. Trace lines

A node writes `{"stage", "status", "detail", "ms", ["input_tokens", "output_tokens"]}` with `get_stream_writer()`.

| stage | written by |
|---|---|
| `guard` | guard |
| `arty` | supervisor and `respond` (both are Arty) |
| `tool` | the worker that called the tool: `"<tool> [read-only] · …"`; on failure: status `error`, detail with attempts and error |
| `<worker name>` | each worker's own model step (`rag_agent`, later `youtube_researcher`…) |

`status` values: `ok`, `blocked` (guard), `error` (a failure the run survived), `stopped` (step cap).
The UI shows anything other than `ok` in the warning colour, so new statuses need no frontend change.
Stage colours for new workers are T10's job.

## 6. RAG re-search (T4)

`rag_agent` searches `task`. **Only if that returns no hits**, it asks the model for a rephrased query
(structured output schema `Rephrase(query: str)`) and searches once more. The cap is 2 searches, and still nothing
found → `NOT_FOUND` with no answer call. Each search writes its own `tool` trace line (`search 1/2`, `search 2/2`).
The re-search is decided by code (zero hits), so the model only writes the new wording. The fake model learns to fill
`Rephrase` (T4 owns `model.py` for this).

## 7. Who owns what in Wave 1

| Ticket | Model | Owns (only these) |
|---|---|---|
| T1 supervisor v2 | Sonnet | `agents/supervisor.py`, `agents/workers.py` (new), `graph.py`, `tests/test_supervisor.py` (new) |
| T2 tool gateway | Sonnet | `tools/*`, `tests/test_tools*.py` |
| T4 RAG re-search | Sonnet | `agents/rag_agent.py`, `model.py` (fake model only), `tests/test_rag_agent.py` (new) |
| T5 content | Haiku | `backend/skills/`, `evals/*_golden.yaml` (new ones), `tests/test_eval_sets.py` |
| docs sync | Haiku, after review | `README.md`, `docs/design.html` |

No ticket edits README or the design book. One docs-sync pass at the end avoids merge conflicts there.
