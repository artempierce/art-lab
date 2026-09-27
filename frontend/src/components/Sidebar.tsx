/**
 * Sidebar.tsx — the left pane: the app name, "New chat", the list of saved chats, a collapsible Team
 * section (X3), and the Memory / Runs page-navigation buttons (Phase 12, P12).
 *
 * It only displays what App gives it and reports clicks back through onOpen / onNew / onNavigate /
 * onDelete (each chat row has an × that deletes the chat and all its server-side data).
 * Team is the one exception: it fetches its own data (see TeamList.tsx) since nothing else on the
 * page needs it. The open chat (or page) is tinted with the accent colour.
 */
import type { Thread } from '../api'
import type { View } from '../App'
import { TeamList } from './TeamList'

type Props = {
  threads: Thread[] // chats to list, newest first
  error: string | null // shown instead of the list when loading failed
  activeId: string | null // the open chat, highlighted (only meaningful while view === 'chat')
  view: View // which page is open right now; highlights the matching nav button
  onOpen: (id: string) => void // a chat was clicked
  onNew: () => void // "New chat" was clicked
  onNavigate: (view: View) => void // "Memory" or "Runs" was clicked
  onDelete: (id: string, title: string) => void // a chat's × was clicked (App asks to confirm)
  busy: boolean // an answer is streaming; delete buttons are disabled meanwhile
}

/** Turn an ISO timestamp into a short relative label: "just now", "5m ago", "3h ago", "2d ago". */
function timeAgo(iso: string): string {
  const minutes = Math.round((Date.now() - new Date(iso).getTime()) / 60_000)
  if (minutes < 1) return 'just now'
  if (minutes < 60) return `${minutes}m ago`
  if (minutes < 60 * 24) return `${Math.round(minutes / 60)}h ago`
  return `${Math.round(minutes / 60 / 24)}d ago`
}

/** Classes for a Memory/Runs nav button: accent-tinted while its page is open. */
function navClass(active: boolean): string {
  return `rounded-lg px-3 py-1.5 text-sm focus-visible:outline-2 focus-visible:outline-accent ${
    active ? 'bg-accent/15 text-ink' : 'text-muted hover:bg-surface hover:text-ink'
  }`
}

export function Sidebar({ threads, error, activeId, view, onOpen, onNew, onNavigate, onDelete, busy }: Props) {
  return (
    // Hidden below 1024px wide (lg); App shows only the chat there.
    <aside className="hidden min-h-0 flex-col gap-4 border-r border-rule bg-bg p-4 lg:flex">
      <div className="px-1 text-xl font-bold tracking-tight">Art Lab</div>

      <button
        type="button"
        onClick={onNew}
        className="rounded-lg border border-rule bg-surface px-3 py-2 text-left text-sm font-medium hover:border-accent focus-visible:outline-2 focus-visible:outline-accent"
      >
        + New chat
      </button>

      {/* The chat list scrolls on its own; the header and footer stay put. */}
      <nav aria-label="Chats" className="-mx-1 flex min-h-0 flex-1 flex-col gap-0.5 overflow-y-auto px-1">
        {error && <p className="px-1 text-sm text-danger">{error}</p>}
        {!error && threads.length === 0 && <p className="px-3 text-sm text-muted">No chats yet.</p>}
        {threads.map((t) => {
          const active = t.thread_id === activeId && view === 'chat'
          // One row, two buttons side by side: open (most of the row) and delete (the ×).
          // They're siblings, not nested — a <button> inside a <button> is invalid HTML. `group`
          // lets the × show only while the row is hovered or focused (`group-hover:`), so the list
          // stays calm until you reach for it.
          return (
            <div
              key={t.thread_id}
              className={`group flex items-stretch rounded-lg ${active ? 'bg-accent/15' : 'hover:bg-surface'}`}
            >
              <button
                type="button"
                onClick={() => onOpen(t.thread_id)}
                aria-current={active ? 'page' : undefined} // tells screen readers which chat is open
                className={`min-w-0 flex-1 rounded-lg px-3 py-2 text-left focus-visible:outline-2 focus-visible:outline-accent ${
                  active ? 'text-ink' : 'text-muted group-hover:text-ink'
                }`}
              >
                <span className="block truncate text-sm">{t.title}</span>
                <span className="block text-xs text-muted">{timeAgo(t.updated_at)}</span>
              </button>
              <button
                type="button"
                onClick={() => onDelete(t.thread_id, t.title)}
                disabled={busy} // can't delete while an answer streams (it might be this chat's)
                aria-label={`Delete chat: ${t.title}`}
                title="Delete chat"
                className="rounded-lg px-2.5 text-muted opacity-0 group-focus-within:opacity-100 group-hover:opacity-100 hover:text-danger focus-visible:opacity-100 focus-visible:outline-2 focus-visible:outline-accent disabled:opacity-0"
              >
                ×
              </button>
            </div>
          )
        })}
      </nav>

      <TeamList />

      {/* Page navigation: the Memory page (Phase 12) and the Runs page (Phase 13). */}
      <nav aria-label="Pages" className="flex gap-1 border-t border-rule pt-3">
        <button
          type="button"
          onClick={() => onNavigate('memory')}
          aria-current={view === 'memory' ? 'page' : undefined}
          className={navClass(view === 'memory')}
        >
          Memory
        </button>
        <button
          type="button"
          onClick={() => onNavigate('runs')}
          aria-current={view === 'runs' ? 'page' : undefined}
          className={navClass(view === 'runs')}
        >
          Runs
        </button>
      </nav>
    </aside>
  )
}
