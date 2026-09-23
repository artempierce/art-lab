import { useEffect, useRef, useState } from 'react'
import type { Run } from '../App'

// Stage colours match the design book: guard amber, agent blue (tools, memory and human join in later phases).
const STAGE_COLOR: Record<string, string> = { guard: 'text-t-guard', llm: 'text-t-agent' }
const STATUS_ICON: Record<string, string> = { ok: '✓', blocked: '⛔', error: '✕' }

function formatTokens(n: number): string {
  return n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n)
}

export function TracePanel({ runs, busy }: { runs: Run[]; busy: boolean }) {
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: 'end' })
  }, [runs])

  return (
    <aside aria-label="Trace" className="hidden min-h-0 flex-col bg-term font-mono text-[12.5px] text-term-ink lg:flex">
      <header className="border-b border-term-rule px-4 py-3.5 text-xs tracking-widest text-term-dim uppercase">Trace</header>
      <div className="min-h-0 flex-1 space-y-6 overflow-y-auto px-4 py-4">
        {runs.length === 0 && (
          <p className="leading-relaxed text-term-dim">
            Send a message and each step of the run shows up here: the guard, the model call, tokens, cost and time.
          </p>
        )}
        {runs.map((run, i) => (
          <RunBlock key={i} run={run} running={busy && i === runs.length - 1} />
        ))}
        <div ref={endRef} />
      </div>
    </aside>
  )
}

function RunBlock({ run, running }: { run: Run; running: boolean }) {
  return (
    <section>
      <div className="mb-2 flex items-baseline justify-between gap-3 text-term-dim">
        <span className="truncate">› {run.prompt}</span>
        {run.traceId && <CopyId id={run.traceId} />}
      </div>
      <ul className="space-y-1">
        {run.lines.map((line, i) => {
          const failed = line.status !== 'ok'
          const color = failed ? 'text-t-human' : (STAGE_COLOR[line.stage] ?? 'text-term-ink')
          return (
            <li key={i} className="grid grid-cols-[1.4em_4.5em_minmax(0,1fr)_auto] gap-x-2">
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
