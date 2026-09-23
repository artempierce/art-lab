import type { Thread } from '../api'

type Props = {
  threads: Thread[]
  error: string | null
  activeId: string | null
  onOpen: (id: string) => void
  onNew: () => void
}

function timeAgo(iso: string): string {
  const minutes = Math.round((Date.now() - new Date(iso).getTime()) / 60_000)
  if (minutes < 1) return 'just now'
  if (minutes < 60) return `${minutes}m ago`
  if (minutes < 60 * 24) return `${Math.round(minutes / 60)}h ago`
  return `${Math.round(minutes / 60 / 24)}d ago`
}

export function Sidebar({ threads, error, activeId, onOpen, onNew }: Props) {
  return (
    <aside className="hidden min-h-0 flex-col gap-4 border-r border-rule p-4 lg:flex">
      <div className="font-display text-xl font-bold tracking-tight">Art Lab</div>

      <button
        type="button"
        onClick={onNew}
        className="rounded-lg border border-rule bg-surface px-3 py-2 text-left text-sm font-medium hover:border-accent focus-visible:outline-2 focus-visible:outline-accent"
      >
        + New chat
      </button>

      <nav aria-label="Chats" className="-mx-1 flex min-h-0 flex-1 flex-col gap-0.5 overflow-y-auto px-1">
        {error && <p className="text-sm text-danger">{error}</p>}
        {!error && threads.length === 0 && <p className="px-3 text-sm text-muted">No chats yet.</p>}
        {threads.map((t) => {
          const active = t.thread_id === activeId
          return (
            <button
              key={t.thread_id}
              type="button"
              onClick={() => onOpen(t.thread_id)}
              aria-current={active ? 'page' : undefined}
              className={`rounded-lg px-3 py-2 text-left focus-visible:outline-2 focus-visible:outline-accent ${
                active ? 'bg-accent/12 text-ink' : 'text-muted hover:bg-surface hover:text-ink'
              }`}
            >
              <span className="block truncate text-sm">{t.title}</span>
              <span className="block text-xs text-muted">{timeAgo(t.updated_at)}</span>
            </button>
          )
        })}
      </nav>

      <div className="border-t border-rule pt-3">
        <p className="mb-1 px-3 font-mono text-[11px] tracking-wider text-muted uppercase">Coming later</p>
        <p className="px-3 py-1 text-sm text-muted">Memory</p>
        <p className="px-3 py-1 text-sm text-muted">Runs</p>
      </div>
    </aside>
  )
}
