/**
 * TracePanel.tsx — the right pane: a log of what the backend did for each message.
 * This is where the app's "visible" goal lives.
 *
 * A terminal-dark column with one box per message you sent (a Run), each with:
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

// Stage name → text colour (light pastels from index.css, readable on the terminal background): guard
// amber, Arty blue, rag_agent peach, tools teal, youtube_researcher sky, content_ideator lilac,
// english_coach pink.
const STAGE_COLOR: Record<string, string> = {
  guard: 'text-t-guard',
  arty: 'text-t-agent',
  rag_agent: 'text-t-rag',
  tool: 'text-t-tool',
  youtube_researcher: 'text-t-research',
  content_ideator: 'text-t-ideas',
  english_coach: 'text-t-coach',
  memory: 'text-t-memory', // Phase 7: recall, remember and summarize lines
  skill: 'text-t-skill', // Phase 8: a load_skill call, kept apart from the generic "tool" colour
}

// Status → icon at the start of the line. Unknown statuses get a plain dot. `flagged` is the input
// guard's classifier tainting a message without blocking it; `stopped` is a turn cut off at its
// time/cost cap (Phase 10, contracts.md § 14); `approval` is the tool loop pausing for your
// Approve/Reject click (Phase 6).
const STATUS_ICON: Record<string, string> = { ok: '✓', blocked: '⛔', error: '✕', flagged: '⚑', stopped: '■', approval: '?' }

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
    <aside aria-label="Trace" className="hidden min-h-0 flex-col border-l border-rule bg-term font-mono text-[12.5px] text-term-ink lg:flex">
      <header className="border-b border-term-rule px-4 py-3.5 text-xs tracking-widest text-term-dim uppercase">
        Trace
      </header>
      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-4 py-4">
        {runs.length === 0 && (
          <p className="leading-relaxed text-term-dim">
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

/**
 * One run's card: header (prompt + trace ID), one line per stage, then a running/error/footer line.
 * Exported so the Runs page (Phase 13, contracts.md § 15) can reuse this exact card to replay a
 * recorded request's trace, instead of a second copy of this layout.
 */
export function RunBlock({ run, running }: { run: Run; running: boolean }) {
  return (
    // Its own terminal-dark box (not a `card`), so a replayed run on the Runs page looks exactly like
    // it did live in the trace panel.
    // `@container` makes this box a *container* for container queries: the `@lg:` classes below
    // react to the box's own width (≥ 32rem), not the window's — so the same card lays itself out
    // differently in the narrow trace panel and on the wide Runs page.
    <section className="@container rounded-lg border border-term-rule bg-term p-3 font-mono text-[12.5px] text-term-ink">
      <div className="mb-2 flex items-baseline justify-between gap-3 text-term-dim">
        <span className="truncate">› {run.prompt}</span>
        {run.traceId && <CopyId id={run.traceId} />}
      </div>
      <ul className="space-y-1.5 @lg:space-y-1">
        {run.lines.map((line, i) => {
          // Anything other than "ok" (blocked, error) is shown in the warning colour.
          const failed = line.status !== 'ok'
          const color = failed ? 'text-t-human' : (STAGE_COLOR[line.stage] ?? 'text-term-ink')
          return (
            // Narrow box (the trace panel): two rows, so the detail gets the full width.
            //   icon | stage          | time
            //        | detail (spans stage + time columns)
            // Wide box (the Runs page, @lg): one row of four columns.
            //   icon | stage (11em fits "youtube_researcher") | detail | time
            <li key={i} className="grid grid-cols-[1.4em_minmax(0,1fr)_auto] gap-x-2 @lg:grid-cols-[1.4em_11em_minmax(0,1fr)_auto]">
              <span className={color}>{STATUS_ICON[line.status] ?? '•'}</span>
              <span className={`font-medium ${color}`}>{line.stage}</span>
              <span className="col-span-2 col-start-2 row-start-2 break-words @lg:col-span-1 @lg:col-start-3 @lg:row-start-1">
                {line.detail}
              </span>
              <span className="col-start-3 row-start-1 text-right text-term-dim tabular-nums @lg:col-start-4">{line.ms}ms</span>
            </li>
          )
        })}
        {running && !run.summary && !run.error && <li className="animate-pulse text-term-dim">… running</li>}
        {run.error && <li className="break-words text-t-human">✕ {run.error}</li>}
      </ul>
      {run.summary && (
        <div className="mt-2 border-t border-dashed border-term-rule pt-2 text-term-dim tabular-nums">
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
      className="shrink-0 px-1 text-term-dim hover:text-term-ink focus-visible:outline-1 focus-visible:outline-accent"
    >
      {copied ? 'copied' : id.slice(0, 8)}
    </button>
  )
}
