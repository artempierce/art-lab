/**
 * Sidebar.tsx — the left pane: the Art Lab logo with Arty, "New chat", the list of saved chats, a
 * collapsible Team section (X3), and the Memory / Runs page-navigation buttons (Phase 12, P12).
 *
 * It only displays what App gives it and reports clicks back through onOpen / onNew / onNavigate.
 * Team is the one exception: it fetches its own data (see TeamList.tsx) since nothing else on the
 * page needs it. Style: the sage page with white cards (see .card, .btn-black in index.css); the
 * open chat (or page) is orange.
 */
import type { Thread } from '../api'
import type { View } from '../App'
import { Arty } from './Arty'
import { TeamList } from './TeamList'

type Props = {
  threads: Thread[] // chats to list, newest first
  error: string | null // shown instead of the list when loading failed
  activeId: string | null // the open chat, highlighted (only meaningful while view === 'chat')
  view: View // which page is open right now; highlights the matching nav button
  onOpen: (id: string) => void // a chat was clicked
  onNew: () => void // "New chat" was clicked
  onNavigate: (view: View) => void // "Memory" or "Runs" was clicked
}

/** Turn an ISO timestamp into a short relative label: "just now", "5m ago", "3h ago", "2d ago". */
function timeAgo(iso: string): string {
  const minutes = Math.round((Date.now() - new Date(iso).getTime()) / 60_000)
  if (minutes < 1) return 'just now'
  if (minutes < 60) return `${minutes}m ago`
  if (minutes < 60 * 24) return `${Math.round(minutes / 60)}h ago`
  return `${Math.round(minutes / 60 / 24)}d ago`
}

export function Sidebar({ threads, error, activeId, view, onOpen, onNew, onNavigate }: Props) {
  return (
    // Hidden below 1024px wide (lg); App shows only the chat there.
    <aside className="hidden min-h-0 flex-col gap-5 border-r-2 border-ink bg-sage p-5 lg:flex">
      {/* Logo: Arty next to the app name in heavy display type. */}
      <div className="flex items-center gap-3">
        <Arty size={64} />
        <div>
          <div className="display text-4xl uppercase">Art Lab</div>
          <div className="mt-1 text-xs font-medium">with Arty, your interface friend</div>
        </div>
      </div>

      <button type="button" onClick={onNew} className="btn-black px-3 py-2.5 text-lg">
        + New chat
      </button>

      {/* The chat list scrolls on its own; the logo and footer stay put. */}
      <nav aria-label="Chats" className="-mx-1 flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto px-1 pt-1 pb-3">
        {error && <p className="card bg-[#fde2dc] p-3 text-sm font-medium text-danger">{error}</p>}
        {!error && threads.length === 0 && <p className="text-sm font-medium">No chats yet — say hi to Arty!</p>}
        {threads.map((t) => {
          const active = t.thread_id === activeId
          return (
            <button
              key={t.thread_id}
              type="button"
              onClick={() => onOpen(t.thread_id)}
              aria-current={active ? 'page' : undefined} // tells screen readers which chat is open
              className={`card px-3 py-2 text-left transition-transform hover:-translate-y-0.5 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ink ${
                active ? 'bg-orange' : ''
              }`}
            >
              <span className="block truncate text-sm font-semibold">{t.title}</span>
              <span className="block text-xs text-muted">{timeAgo(t.updated_at)}</span>
            </button>
          )
        })}
      </nav>

      <TeamList />

      {/* Page navigation: Memory (Phase 12, this ticket) and Runs (Phase 13, a placeholder here). */}
      <nav aria-label="Pages" className="border-t-2 border-ink pt-4">
        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => onNavigate('memory')}
            aria-current={view === 'memory' ? 'page' : undefined}
            className={`btn-outline px-3 py-1 text-sm ${view === 'memory' ? 'bg-orange' : ''}`}
          >
            Memory
          </button>
          <button
            type="button"
            onClick={() => onNavigate('runs')}
            aria-current={view === 'runs' ? 'page' : undefined}
            className={`btn-outline px-3 py-1 text-sm ${view === 'runs' ? 'bg-orange' : ''}`}
          >
            Runs
          </button>
        </div>
      </nav>
    </aside>
  )
}
