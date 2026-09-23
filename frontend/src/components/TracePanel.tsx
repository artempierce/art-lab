/**
 * TracePanel.tsx — the right pane: a log of what the backend did for each message.
 * This is where the app's "visible" goal lives.
 *
 * One white card per message you sent (a Run), each with:
 *   › your message (shortened)                         trace ID (click to copy, find it in LangSmith)
 *   ✓ guard        pass · 44 chars · budget 0.4% used                                             1ms
 *   ✓ arty         → rag_agent · question about studio policy                                   620ms
 *   ✓ tool         search_knowledge [read-only] · 4 chunks · 2 files · top 0.81                  40ms
 *   ✓ rag_agent    claude-haiku-4-5 · 812 in / 240 out · cites [1] [2]                         2310ms
 *   ✓ arty         done · answered by rag_agent                                                   0ms
 *   1.4k tok · $0.0024 · 3.0s                                              ← footer, when the run ends
 *
 * The lines come straight from the backend's `trace` events (each graph node writes one). New stages
 * show up here automatically as later phases add them; they only need a colour in STAGE_COLOR.
 */
import { useEffect, useRef, useState } from 'react'
import type { Run } from '../App'

// Stage name → text colour: guard amber, Arty navy, rag_agent red, tools green (dark enough for white cards).
const STAGE_COLOR: Record<string, string> = {
  guard: 'text-t-guard',
  arty: 'text-t-agent',
  rag_agent: 'text-t-rag',
  tool: 'text-t-tool',
}

// Status → icon at the start of the line. Unknown statuses get a plain dot.
const STATUS_ICON: Record<string, string> = { ok: '✓', blocked: '⛔', error: '✕' }

/** Short token count: 812 → "812", 1234 → "1.2k". */
function formatTokens(n: number): string {
  return n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n)
}

export function TracePanel({ runs, busy }: { runs: Run[]; busy: boolean }) {
  const endRef = useRef<HTMLDivElement>(null)

  // Keep the newest line in view as runs and lines are added.
  useEffect(() => {
    endRef.current?.scrollIntoView({ block: 'end' })
  }, [runs])

  return (
    // Hidden below 1024px wide (lg).
    <aside aria-label="Trace" className="hidden min-h-0 flex-col border-l-2 border-ink bg-sage font-mono text-[12.5px] lg:flex">
      <header className="border-b-2 border-ink px-5 py-3">
        <h2 className="display text-4xl uppercase">Trace</h2>
        <p className="mt-1 font-sans text-xs">Every step Arty takes, live</p>
      </header>
      <div className="min-h-0 flex-1 space-y-5 overflow-y-auto px-5 py-5">
        {runs.length === 0 && (
          <p className="font-sans text-sm leading-relaxed">
            Send a message and each step shows up here: the guard, Arty's decision, the knowledge-base search, the
            answer — with tokens, cost and time.
          </p>
        )}
        {runs.map((run, i) => (
          // Only the last run can still be running.
          <RunBlock key={i} run={run} running={busy && i === runs.length - 1} />
        ))}
        <div ref={endRef} />
      </div>
    </aside>
  )
}

/** One run's card: header (prompt + trace ID), one line per stage, then a running/error/footer line. */
function RunBlock({ run, running }: { run: Run; running: boolean }) {
  return (
    <section className="card p-3">
      <div className="mb-2 flex items-baseline justify-between gap-3 text-muted">
        <span className="truncate">› {run.prompt}</span>
        {run.traceId && <CopyId id={run.traceId} />}
      </div>
      <ul className="space-y-1">
        {run.lines.map((line, i) => {
          // Anything other than "ok" (blocked, error) is shown in the warning colour.
          const failed = line.status !== 'ok'
          const color = failed ? 'text-t-human' : (STAGE_COLOR[line.stage] ?? 'text-ink')
          return (
            // Four columns: icon | stage | detail (wraps if long) | time
            <li key={i} className="grid grid-cols-[1.4em_6.5em_minmax(0,1fr)_auto] gap-x-2">
              <span className={color}>{STATUS_ICON[line.status] ?? '•'}</span>
              <span className={`font-medium ${color}`}>{line.stage}</span>
              <span className="break-words">{line.detail}</span>
              <span className="text-muted tabular-nums">{line.ms}ms</span>
            </li>
          )
        })}
        {running && !run.summary && !run.error && <li className="animate-pulse text-muted">… running</li>}
        {run.error && <li className="break-words text-t-human">✕ {run.error}</li>}
      </ul>
      {run.summary && (
        <div className="mt-2 border-t-2 border-dashed border-shade pt-2 text-muted tabular-nums">
          {formatTokens(run.summary.input_tokens + run.summary.output_tokens)} tok · ${run.summary.cost_usd.toFixed(4)} ·{' '}
          {(run.summary.ms / 1000).toFixed(1)}s
        </div>
      )}
    </section>
  )
}

/**
 * The first 8 characters of the trace ID, as a button. Clicking copies the full ID so you can
 * paste it into LangSmith's search; the label says "copied" for 1.5 seconds.
 */
function CopyId({ id }: { id: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <button
      type="button"
      title={`Trace ID ${id} — click to copy, then search it in LangSmith`}
      onClick={() =>
        navigator.clipboard.writeText(id).then(() => {
          setCopied(true)
          setTimeout(() => setCopied(false), 1500)
        })
      }
      className="shrink-0 px-1 text-muted hover:text-ink focus-visible:outline-1 focus-visible:outline-ink"
    >
      {copied ? 'copied' : id.slice(0, 8)}
    </button>
  )
}
