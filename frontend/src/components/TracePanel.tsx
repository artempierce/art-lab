/**
 * TracePanel.tsx — the right pane: a terminal-style log of what the backend did for each
 * message. This is where the app's "visible" goal lives.
 *
 * One block per message you sent (a Run), each with:
 *   › your message (shortened)                         trace ID (click to copy, find it in LangSmith)
 *   ✓ guard        pass · 44 chars · budget 0.4% used                                             1ms
 *   ✓ supervisor   → rag_agent · question about studio policy                                   620ms
 *   ✓ tool         search_knowledge [read-only] · 4 chunks · 2 files · top 0.81                  40ms
 *   ✓ rag_agent    claude-haiku-4-5 · 812 in / 240 out · cites [1] [2]                         2310ms
 *   ✓ supervisor   finish · answered by rag_agent                                                 0ms
 *   1.4k tok · $0.0024 · 3.0s                                              ← footer, when the run ends
 *
 * The lines come straight from the backend's `trace` events (each graph node writes one).
 * New stages (supervisor, tools, memory…) show up here automatically as later phases add them;
 * they only need a colour in STAGE_COLOR.
 */
import { useEffect, useRef, useState } from 'react'
import type { Run } from '../App'

// Stage name → text colour, matching the design book: guard amber, agents blue, tools teal.
// (memory violet and human rose are defined in index.css for later phases.)
const STAGE_COLOR: Record<string, string> = {
  guard: 'text-t-guard',
  supervisor: 'text-t-agent',
  respond: 'text-t-agent',
  rag_agent: 'text-t-agent',
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
    // Hidden below 1024px wide (lg). Always dark, like a terminal, in both light and dark themes.
    <aside aria-label="Trace" className="hidden min-h-0 flex-col bg-term font-mono text-[12.5px] text-term-ink lg:flex">
      <header className="border-b border-term-rule px-4 py-3.5 text-xs tracking-widest text-term-dim uppercase">Trace</header>
      <div className="min-h-0 flex-1 space-y-6 overflow-y-auto px-4 py-4">
        {runs.length === 0 && (
          <p className="leading-relaxed text-term-dim">
            Send a message and each step of the run shows up here: the guard, the model call, tokens, cost and time.
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

/** One run's block: header (prompt + trace ID), one line per stage, then a running/error/footer line. */
function RunBlock({ run, running }: { run: Run; running: boolean }) {
  return (
    <section>
      <div className="mb-2 flex items-baseline justify-between gap-3 text-term-dim">
        <span className="truncate">› {run.prompt}</span>
        {run.traceId && <CopyId id={run.traceId} />}
      </div>
      <ul className="space-y-1">
        {run.lines.map((line, i) => {
          // Anything other than "ok" (blocked, error) is shown in the warning colour.
          const failed = line.status !== 'ok'
          const color = failed ? 'text-t-human' : (STAGE_COLOR[line.stage] ?? 'text-term-ink')
          return (
            // Four columns: icon | stage | detail (wraps if long) | time
            <li key={i} className="grid grid-cols-[1.4em_6.5em_minmax(0,1fr)_auto] gap-x-2">
              <span className={color}>{STATUS_ICON[line.status] ?? '•'}</span>
              <span className={color}>{line.stage}</span>
              <span className="break-words">{line.detail}</span>
              <span className="text-term-dim tabular-nums">{line.ms}ms</span>
            </li>
          )
        })}
        {running && !run.summary && !run.error && <li className="animate-pulse text-term-dim">… running</li>}
        {run.error && <li className="break-words text-t-human">✕ {run.error}</li>}
      </ul>
      {run.summary && (
        <div className="mt-2 border-t border-term-rule pt-2 text-term-dim tabular-nums">
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
      className="shrink-0 rounded px-1 text-term-dim hover:text-term-ink focus-visible:outline-1 focus-visible:outline-term-ink"
    >
      {copied ? 'copied' : id.slice(0, 8)}
    </button>
  )
}
