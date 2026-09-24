/**
 * MemoryPage.tsx — the Memory page (Phase 12, P12): every fact Arty has learned about you from your
 * own chat messages (agents/remember.py, docs/contracts.md § 11), as a list of cards you can edit or
 * delete. There's no "add a fact" here — facts only ever come in through your own words in a chat;
 * this page only lets you fix a wrong one or make Arty forget it.
 *
 * Fetches its own data on mount (the same pattern TeamList.tsx uses): nothing else on the page needs
 * the fact list. Style: the sage page with white cards, black/outline buttons (see index.css);
 * the delete confirm step follows ApprovalCard.tsx's "ask before a one-way action" shape.
 */
import { useCallback, useEffect, useState } from 'react'
import { type Fact, deleteMemory, getMemory, updateMemory } from '../api'

const BACKEND_DOWN = "Can't reach the backend. Start it with: cd backend && uv run uvicorn artlab.api:app --port 8000"

/** A short, readable date for a fact's `created_at` (a Unix timestamp in seconds). */
function formatDate(createdAt: number): string {
  return new Date(createdAt * 1000).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
}

export function MemoryPage() {
  const [facts, setFacts] = useState<Fact[] | null>(null) // null while the first load is in flight
  const [loadError, setLoadError] = useState<string | null>(null)

  /** (Re)load the fact list from the backend. Passed down to each card so edit/delete can refresh
   * the whole list from the source of truth instead of guessing what changed locally. */
  const load = useCallback(() => {
    getMemory()
      .then((f) => {
        setFacts(f)
        setLoadError(null)
      })
      .catch(() => setLoadError(BACKEND_DOWN))
  }, [])

  useEffect(load, [load])

  return (
    <main className="min-h-0 overflow-y-auto bg-sage p-6">
      <h1 className="display text-3xl uppercase">Memory</h1>
      <p className="mt-1 max-w-2xl text-sm text-muted">
        Facts Arty has picked up from your own messages — never from a tool result, a document, or anything else it
        read (that's the safety rule behind long-term memory). Fix one if it's wrong, or delete it to make Arty
        forget.
      </p>

      <div className="mt-6 flex max-w-2xl flex-col gap-3">
        {loadError && (
          <p role="alert" className="card bg-[#fde2dc] p-3 text-sm font-medium text-danger">
            {loadError}
          </p>
        )}
        {!loadError && facts === null && <p className="text-sm font-medium">Loading…</p>}
        {!loadError && facts?.length === 0 && (
          <p className="text-sm font-medium">
            No facts yet. Tell Arty something durable about yourself or your channel in a chat — your niche, your
            audience, your posting schedule — and it'll show up here.
          </p>
        )}
        {facts?.map((fact) => (
          <FactCard key={fact.key} fact={fact} onChanged={load} />
        ))}
      </div>
    </main>
  )
}

/** The three things a card can be doing: just showing the fact, editing its value, or asking you to
 * confirm a delete. Only one of edit/confirm-delete is ever open at once. */
type Mode = 'view' | 'edit' | 'confirm-delete'

/**
 * One fact: its key, value, when it was learned, and the Edit/Delete controls. `onChanged` is
 * called after a successful save or delete, so the page reloads the list from the backend rather
 * than the card guessing what the new state should look like.
 */
function FactCard({ fact, onChanged }: { fact: Fact; onChanged: () => void }) {
  const [mode, setMode] = useState<Mode>('view')
  const [draft, setDraft] = useState(fact.value) // the input's text while editing
  const [busy, setBusy] = useState(false) // a save or delete request is in flight
  const [actionError, setActionError] = useState<string | null>(null)

  /** Open the inline editor, seeded with the fact's current value. */
  function startEdit() {
    setDraft(fact.value)
    setActionError(null)
    setMode('edit')
  }

  /** Send the edited value; on success, close the editor and let the parent reload the list. */
  async function save() {
    const value = draft.trim()
    if (!value) return
    setBusy(true)
    setActionError(null)
    try {
      await updateMemory(fact.key, value)
      setMode('view')
      onChanged()
    } catch {
      setActionError("Couldn't save that change. Try again.")
    } finally {
      setBusy(false)
    }
  }

  /** Delete for real, after the confirm step below asked once. */
  async function confirmDelete() {
    setBusy(true)
    setActionError(null)
    try {
      await deleteMemory(fact.key)
      onChanged() // the card itself disappears once the parent's list no longer includes this key
    } catch {
      setActionError("Couldn't delete that fact. Try again.")
      setMode('view')
      setBusy(false)
    }
  }

  return (
    <div className="card p-4">
      <div className="flex items-baseline justify-between gap-3">
        <span className="font-mono text-xs text-muted">{fact.key}</span>
        <span className="text-xs text-muted">{formatDate(fact.created_at)}</span>
      </div>

      {mode === 'edit' ? (
        <div className="mt-2">
          <label htmlFor={`fact-value-${fact.key}`} className="sr-only">
            Value for {fact.key}
          </label>
          <input
            id={`fact-value-${fact.key}`}
            type="text"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            disabled={busy}
            autoFocus
            className="w-full border-2 border-ink bg-paper px-2 py-1.5 text-sm outline-none"
          />
        </div>
      ) : (
        <p className="mt-1 text-sm font-medium">{fact.value}</p>
      )}

      {actionError && (
        <p role="alert" className="mt-2 text-xs font-medium text-danger">
          {actionError}
        </p>
      )}

      {mode === 'confirm-delete' && (
        <p role="alert" className="mt-3 border-2 border-ink bg-[#fde2dc] px-3 py-2 text-xs font-medium text-danger">
          Delete this fact? This can't be undone.
        </p>
      )}

      <div className="mt-3 flex gap-2">
        {mode === 'edit' && (
          <>
            <button type="button" onClick={save} disabled={busy || !draft.trim()} className="btn-black px-3 py-1 text-sm">
              Save
            </button>
            <button type="button" onClick={() => setMode('view')} disabled={busy} className="btn-outline px-3 py-1 text-sm">
              Cancel
            </button>
          </>
        )}
        {mode === 'confirm-delete' && (
          <>
            <button type="button" onClick={confirmDelete} disabled={busy} className="btn-black px-3 py-1 text-sm">
              Yes, delete
            </button>
            <button type="button" onClick={() => setMode('view')} disabled={busy} className="btn-outline px-3 py-1 text-sm">
              Cancel
            </button>
          </>
        )}
        {mode === 'view' && (
          <>
            <button type="button" onClick={startEdit} aria-label={`Edit ${fact.key}`} className="btn-outline px-3 py-1 text-sm">
              Edit
            </button>
            <button
              type="button"
              onClick={() => setMode('confirm-delete')}
              aria-label={`Delete ${fact.key}`}
              className="btn-outline px-3 py-1 text-sm"
            >
              Delete
            </button>
          </>
        )}
      </div>
    </div>
  )
}
