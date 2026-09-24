/**
 * ApprovalCard.tsx — the Approve / Reject card shown under an assistant reply that's waiting on you
 * (Phase 6, docs/contracts.md § 10). It appears once the message it belongs to has carried an
 * `approval` SSE event (see api.ts's `approval` ChatEvent and App.tsx, which attaches it to the
 * message as `message.approval`): a mutating tool like save_ideas never runs on its own, so the tool
 * loop stops and this card asks before it does.
 *
 * A reloaded chat's history (GET /api/threads/:id) never includes a pending approval — the backend
 * only sends the `approval` event on a live stream — so old cards never reappear after a refresh.
 */
import type { Approval } from '../api'

type Props = {
  approval: Approval // the request: which tool, whose agent, what arguments, and whether the chat is tainted
  decision?: 'approved' | 'rejected' // set once you've clicked a button; undefined while still waiting
  busy: boolean // true while any stream (including this card's own resume) is in flight
  onRespond: (approve: boolean) => void // called with true for Approve, false for Reject
}

/**
 * The card: a title naming the tool, who asked, a preview of the arguments, an optional taint
 * warning, and the two buttons. Once `decision` is set (or while `busy`) the buttons disable; an
 * `aria-live` line announces the outcome for screen readers, since nothing else on the page moves.
 */
export function ApprovalCard({ approval, decision, busy, onRespond }: Props) {
  const disabled = busy || decision !== undefined

  return (
    <div className="mt-3 border-t-2 border-ink pt-3">
      <p className="display text-lg uppercase">Arty wants to {approval.tool}</p>
      <p className="text-xs text-muted">Asked by {approval.agent}</p>

      <ArgsPreview tool={approval.tool} args={approval.args} />

      {approval.tainted && (
        <p role="alert" className="mt-2 border-2 border-ink bg-[#fde2dc] px-2 py-1.5 text-xs font-medium text-danger">
          ⚠ This chat read untrusted content ({approval.taint_sources.join(', ')}). Check the text before approving.
        </p>
      )}

      <div className="mt-3 flex gap-2">
        <button type="button" disabled={disabled} onClick={() => onRespond(true)} className="btn-black px-4 py-1.5 text-sm">
          Approve
        </button>
        <button
          type="button"
          disabled={disabled}
          onClick={() => onRespond(false)}
          className="btn-outline px-4 py-1.5 text-sm"
        >
          Reject
        </button>
      </div>

      {/* Always mounted (not only once decided) so assistive tech is already watching this region
          and announces the outcome the instant `decision` changes. */}
      <p aria-live="polite" className="mt-2 text-sm font-medium">
        {decision === 'approved' && '✓ Approved'}
        {decision === 'rejected' && '✕ Rejected'}
      </p>
    </div>
  )
}

/**
 * The tool's arguments, previewed so you can check them before approving. `save_ideas`'s only
 * argument (`ideas`) is the text you're actually approving, so it gets its own scrollable monospace
 * box; any other tool's arguments are shown as pretty-printed JSON.
 */
function ArgsPreview({ tool, args }: { tool: string; args: Record<string, unknown> }) {
  const text = tool === 'save_ideas' && typeof args.ideas === 'string' ? args.ideas : JSON.stringify(args, null, 2)
  return (
    <pre className="mt-2 max-h-48 overflow-y-auto border-2 border-ink bg-sage p-2 font-mono text-xs whitespace-pre-wrap">
      {text}
    </pre>
  )
}
