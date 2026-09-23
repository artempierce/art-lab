/**
 * Sidebar.tsx — the left pane: the Art Lab logo with Arty, "New chat", the list of saved chats, and
 * placeholders for the Memory and Runs pages that come in later phases.
 *
 * It only displays what App gives it and reports clicks back through onOpen / onNew.
 * Style: a periwinkle block; chats are small "sticker" cards (see .sticker in index.css).
 */
import type { Thread } from '../api'
import { Arty } from './Arty'

type Props = {
  threads: Thread[] // chats to list, newest first
  error: string | null // shown instead of the list when loading failed
  activeId: string | null // the open chat, highlighted
  onOpen: (id: string) => void // a chat was clicked
  onNew: () => void // "New chat" was clicked
}

/** Turn an ISO timestamp into a short relative label: "just now", "5m ago", "3h ago", "2d ago". */
function timeAgo(iso: string): string {
  const minutes = Math.round((Date.now() - new Date(iso).getTime()) / 60_000)
  if (minutes < 1) return 'just now'
  if (minutes < 60) return `${minutes}m ago`
  if (minutes < 60 * 24) return `${Math.round(minutes / 60)}h ago`
  return `${Math.round(minutes / 60 / 24)}d ago`
}

export function Sidebar({ threads, error, activeId, onOpen, onNew }: Props) {
  return (
    // Hidden below 1024px wide (lg); App shows only the chat there.
    <aside className="hidden min-h-0 flex-col gap-4 border-r-3 border-ink bg-peri p-4 lg:flex">
      {/* Logo: Arty next to the app name in outlined display type. */}
      <div className="flex items-center gap-2">
        <Arty size={64} />
        <div>
          <div className="outlined text-4xl leading-none">Art Lab</div>
          <div className="mt-1 text-xs font-black tracking-wide uppercase">with Arty</div>
        </div>
      </div>

      <button type="button" onClick={onNew} className="btn-pop bg-salmon px-3 py-2 text-lg text-white">
        + New chat
      </button>

      {/* The chat list scrolls on its own; the logo and footer stay put. */}
      <nav aria-label="Chats" className="-mx-1 flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto px-1 pt-1 pb-2">
        {error && <p className="sticker bg-blush p-3 text-sm font-bold text-danger">{error}</p>}
        {!error && threads.length === 0 && <p className="px-1 text-sm font-bold">No chats yet — say hi to Arty!</p>}
        {threads.map((t) => {
          const active = t.thread_id === activeId
          return (
            <button
              key={t.thread_id}
              type="button"
              onClick={() => onOpen(t.thread_id)}
              aria-current={active ? 'page' : undefined} // tells screen readers which chat is open
              className={`rounded-xl border-3 border-ink px-3 py-2 text-left transition-transform hover:-translate-y-0.5 focus-visible:outline-3 focus-visible:outline-offset-2 focus-visible:outline-ink ${
                active ? 'bg-peach shadow-[3px_3px_0_var(--color-ink)]' : 'bg-cream'
              }`}
            >
              <span className="block truncate text-sm font-extrabold">{t.title}</span>
              <span className="block text-xs font-semibold text-muted">{timeAgo(t.updated_at)}</span>
            </button>
          )
        })}
      </nav>

      {/* Roadmap placeholders: Memory page (Phase 12) and Runs page (Phase 13). */}
      <div className="border-t-3 border-ink pt-3">
        <p className="mb-2 font-display text-sm tracking-wider uppercase">Coming later</p>
        <div className="flex gap-2">
          <span className="rounded-full border-2 border-ink bg-mint px-3 py-0.5 text-xs font-extrabold">Memory</span>
          <span className="rounded-full border-2 border-ink bg-blush px-3 py-0.5 text-xs font-extrabold">Runs</span>
        </div>
      </div>
    </aside>
  )
}
