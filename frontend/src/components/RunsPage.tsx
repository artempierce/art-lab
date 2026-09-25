/**
 * RunsPage.tsx — Phase 13 (docs/contracts.md § 15): every chat request Art Lab has ever recorded,
 * with the latest eval run's summary on top.
 *
 *   ┌─────────────────────────────────────────┐
 *   │ Latest eval run (or "no eval run yet")   │
 *   ├─────────────────────────────────────────┤
 *   │ time │ prompt │ by │ status │ tok │ $ │ ms│  ← one row per recorded request, newest first
 *   ├─────────────────────────────────────────┤
 *   │ (a clicked row's trace, replayed below)  │
 *   └─────────────────────────────────────────┘
 *
 * Unlike the trace panel (session memory only, gone on reload), every row here came from the
 * backend's run log (runs/store.py): it's the permanent history the trace panel's own doc comment
 * points to. This file only reads two things — GET /api/runs(/{trace_id}) and GET /api/evals/latest
 * — and reuses TracePanel's own `RunBlock` to show a clicked row's trace, so a recorded run looks
 * exactly like it did live.
 *
 * P12 (docs/contracts.md § 15) adds the page navigation that renders this component; this file
 * doesn't touch App.tsx or Sidebar.tsx.
 */
import { useEffect, useState } from 'react'
import { type EvalSummary, type RunDetail, type RunRow, getLatestEval, getRun, getRuns } from '../api'
import type { Run } from '../App'
import { RunBlock } from './TracePanel'

// status → text colour, reusing the trace panel's own stage tokens (index.css) so the page doesn't
// need any new colours: guard amber for the input guard's own refusal, Arty's navy for a paused
// approval, the raspberry english_coach token for a breaker/cap stop, green for a clean finish, red
// for an error, and the skill teal for "redacted" (the output guard cleaned something up after the
// fact — not a refusal, so it gets its own colour rather than red or amber).
const STATUS_COLOR: Record<RunRow['status'], string> = {
  ok: 'text-t-tool',
  error: 'text-t-human',
  blocked: 'text-t-guard',
  approval: 'text-t-agent',
  stopped: 'text-t-coach',
  redacted: 'text-t-skill',
}

/** Short token count: 812 → "812", 1234 → "1.2k" (TracePanel.tsx's own formatTokens). */
function formatTokens(n: number): string {
  return n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n)
}

/** `started_at` is seconds since the epoch (Python's `time.time()`, api.py); show it the way a
 * human reads a timestamp, in their own local time. */
function formatTime(startedAt: number): string {
  return new Date(startedAt * 1000).toLocaleString()
}

/** A small uppercase pill naming a run's status, coloured by STATUS_COLOR. */
function StatusBadge({ status }: { status: RunRow['status'] }) {
  return (
    <span className={`inline-block rounded border-2 border-current px-1.5 py-0.5 font-mono text-[11px] uppercase ${STATUS_COLOR[status]}`}>
      {status}
    </span>
  )
}

/** A small error card, the same shape Sidebar.tsx uses for its own "couldn't load" state. */
function ErrorCard({ message }: { message: string }) {
  return <p className="card bg-[#fde2dc] p-3 text-sm font-medium text-danger">{message}</p>
}

/**
 * Turn one fetched RunDetail into the `Run` shape TracePanel's RunBlock already knows how to draw
 * (App.tsx builds the same shape live, from SSE events) — so a recorded run replays in exactly the
 * same card a live one streamed into.
 */
function toRun(detail: RunDetail): Run {
  return {
    traceId: detail.trace_id,
    prompt: detail.prompt,
    lines: detail.trace,
    summary: { input_tokens: detail.input_tokens, output_tokens: detail.output_tokens, cost_usd: detail.cost_usd, ms: detail.ms },
    error: detail.status === 'error' ? 'This request ended in an error — see the trace above.' : undefined,
  }
}

/** The eval card on top: loading, the "never run yet" fallback, a fetch error, or the summary. */
function EvalSummaryCard({ summary, error }: { summary: EvalSummary | null | undefined; error: boolean }) {
  if (error) return <ErrorCard message="Could not load the latest eval run." />
  if (summary === undefined) return <p className="card p-4 text-sm text-muted">Loading the latest eval run…</p>
  if (summary === null) {
    return (
      <p className="card p-4 text-sm">
        No eval run yet — see evals/README or <code className="font-mono">python -m artlab.evals.run</code>.
      </p>
    )
  }

  const suites = Object.entries(summary.suites)
  return (
    <div className="card p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <p className="display text-2xl uppercase">Latest eval run</p>
        <p className="text-xs text-muted">
          {summary.model} · {new Date(summary.started_at).toLocaleString()}
          {summary.dry_run && ' · dry run ($0)'}
        </p>
      </div>
      <ul className="mt-2 flex flex-wrap gap-4 text-sm">
        {suites.map(([name, suite]) => (
          <li key={name}>
            <span className="font-semibold">{name}</span>: {suite.passed}/{suite.total}
          </li>
        ))}
      </ul>
      <p className="mt-2 text-xs text-muted tabular-nums">${summary.cost_usd.toFixed(4)}</p>
    </div>
  )
}

/** The table: one row per recorded request, newest first. Loading/empty/error states of its own. */
function RunsTable({
  rows,
  error,
  selectedId,
  onSelect,
}: {
  rows: RunRow[] | undefined
  error: boolean
  selectedId: string | null
  onSelect: (traceId: string) => void
}) {
  if (error) return <ErrorCard message="Could not load the run log." />
  if (rows === undefined) return <p className="card p-4 text-sm text-muted">Loading runs…</p>
  if (rows.length === 0) {
    return <p className="card p-4 text-sm">No requests recorded yet — send a message in the chat to see it here.</p>
  }

  return (
    <div className="card overflow-x-auto p-0">
      <table className="w-full text-left text-sm">
        <thead className="border-b-2 border-ink font-mono text-xs uppercase">
          <tr>
            <th className="px-3 py-2">Time</th>
            <th className="px-3 py-2">Prompt</th>
            <th className="px-3 py-2">Answered by</th>
            <th className="px-3 py-2">Status</th>
            <th className="px-3 py-2 text-right">Tokens</th>
            <th className="px-3 py-2 text-right">Cost</th>
            <th className="px-3 py-2 text-right">ms</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr
              key={row.trace_id}
              onClick={() => onSelect(row.trace_id)}
              aria-selected={row.trace_id === selectedId}
              className={`cursor-pointer border-b border-shade last:border-0 hover:bg-sage ${
                row.trace_id === selectedId ? 'bg-sage' : ''
              }`}
            >
              <td className="px-3 py-2 whitespace-nowrap text-muted">{formatTime(row.started_at)}</td>
              <td className="max-w-xs truncate px-3 py-2">{row.prompt}</td>
              <td className="px-3 py-2 text-muted">{row.answered_by ?? '—'}</td>
              <td className="px-3 py-2">
                <StatusBadge status={row.status} />
              </td>
              <td className="px-3 py-2 text-right tabular-nums">{formatTokens(row.input_tokens + row.output_tokens)}</td>
              <td className="px-3 py-2 text-right tabular-nums">${row.cost_usd.toFixed(4)}</td>
              <td className="px-3 py-2 text-right tabular-nums">{row.ms}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/**
 * The Runs page: fetches the run log and the latest eval summary once on mount, and a clicked
 * row's own trace on demand. All three loads track their own loading/error state independently, so
 * one failing (say, no eval report yet) never blocks the other two from showing.
 */
export function RunsPage() {
  const [evalSummary, setEvalSummary] = useState<EvalSummary | null | undefined>(undefined) // undefined = still loading
  const [evalError, setEvalError] = useState(false)

  const [rows, setRows] = useState<RunRow[] | undefined>(undefined) // undefined = still loading
  const [rowsError, setRowsError] = useState(false)

  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [selected, setSelected] = useState<Run | undefined>(undefined) // undefined = still loading
  const [selectedError, setSelectedError] = useState(false)

  // Load the eval summary and the run log once, when the page opens.
  useEffect(() => {
    getLatestEval().then(setEvalSummary).catch(() => setEvalError(true))
    getRuns()
      .then(setRows)
      .catch(() => setRowsError(true))
  }, [])

  /** A row was clicked: fetch its full detail (trace lines included) and show it below the table. */
  function openRow(traceId: string) {
    setSelectedId(traceId)
    setSelected(undefined)
    setSelectedError(false)
    getRun(traceId)
      .then((detail) => setSelected(toRun(detail)))
      .catch(() => setSelectedError(true))
  }

  return (
    <div className="h-full min-h-0 overflow-y-auto p-6">
      <div className="mx-auto flex max-w-4xl flex-col gap-5">
        <header>
          <h1 className="display text-5xl uppercase">Runs</h1>
          <p className="mt-1 text-sm text-muted">Every chat request, recorded — click a row to see its trace.</p>
        </header>

        <EvalSummaryCard summary={evalSummary} error={evalError} />
        <RunsTable rows={rows} error={rowsError} selectedId={selectedId} onSelect={openRow} />

        {selectedId && (
          <section>
            {selectedError && <ErrorCard message="Could not load this run's trace." />}
            {!selectedError && selected === undefined && <p className="card p-4 text-sm text-muted">Loading trace…</p>}
            {!selectedError && selected && <RunBlock run={selected} running={false} />}
          </section>
        )}
      </div>
    </div>
  )
}
