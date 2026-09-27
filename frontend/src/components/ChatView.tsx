/**
 * ChatView.tsx — the middle pane: a header with the chat title and what Arty is doing, the messages,
 * and the box you type in. Before the first message it shows a short welcome with example questions.
 *
 * Holds one piece of state of its own: `draft`, the text in the input box. Everything else
 * comes from App as props.
 */
import { useEffect, useRef, useState } from 'react'
import Markdown from 'react-markdown'
import type { Approval, Message, Source } from '../api'
import { ApprovalCard } from './ApprovalCard'

/** What Arty is doing right now, worked out in App from the latest run's trace lines. */
export type Mood = 'idle' | 'thinking' | 'searching' | 'happy' | 'blocked'

type Props = {
  title: string // shown in the header
  messages: Message[] // bubbles to show
  busy: boolean // an answer is streaming; sending is disabled
  mood: Mood // what the header's status line says
  onSend: (text: string) => void // called with the message to send
  onRespondApproval: (approval: Approval, approve: boolean) => void // Approve/Reject click on a message's card
}

// Clickable starters on the welcome screen. Clicking one sends it right away.
const EXAMPLES = [
  'What is our sponsorship disclosure rule?',
  'Who approves a $900 equipment purchase?',
  'Write a hook for a cable management video',
  'How many shorts do we post per week?',
]

// The header's status line for each mood.
const STATUS: Record<Mood, string> = {
  idle: 'Ready',
  thinking: 'Thinking…',
  searching: 'Searching the knowledge base…',
  happy: 'Done',
  blocked: 'Blocked by the guard',
}

export function ChatView({ title, messages, busy, mood, onSend, onRespondApproval }: Props) {
  const [draft, setDraft] = useState('')
  const endRef = useRef<HTMLDivElement>(null) // an empty marker after the last bubble

  // Every time the messages change (a new bubble, or a new token streaming in), scroll the
  // marker into view, so the newest text is always visible.
  useEffect(() => {
    endRef.current?.scrollIntoView({ block: 'end' })
  }, [messages])

  /** Send the draft, unless it's empty or an answer is still streaming; then clear the box. */
  function submit() {
    const text = draft.trim()
    if (!text || busy) return
    setDraft('')
    onSend(text)
  }

  return (
    <main className="flex min-h-0 min-w-0 flex-col bg-surface">
      {/* Header: chat title on the left, Arty's status on the right. */}
      <header className="flex items-center justify-between gap-4 border-b border-rule px-6 py-3">
        <h1 className="truncate text-lg font-semibold">{title}</h1>
        <span role="status" className={`shrink-0 text-xs ${busy ? 'animate-pulse text-accent' : 'text-muted'}`}>
          {STATUS[mood]}
        </span>
      </header>

      {/* The message area scrolls; the header and the input box stay fixed. */}
      <div className="min-h-0 flex-1 overflow-x-hidden overflow-y-auto">
        <div className="mx-auto flex max-w-3xl flex-col gap-5 px-6 py-8">
          {messages.length === 0 ? (
            <EmptyState onPick={onSend} disabled={busy} />
          ) : (
            // Only the last bubble can be "waiting" (the reply that's streaming in).
            messages.map((m, i) => (
              <Bubble
                key={i}
                message={m}
                waiting={busy && i === messages.length - 1}
                mood={mood}
                busy={busy}
                onRespondApproval={onRespondApproval}
              />
            ))
          )}
          <div ref={endRef} />
        </div>
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault() // stop the browser's default full-page form submit
          submit()
        }}
        className="border-t border-rule px-6 py-4"
      >
        <div className="mx-auto flex max-w-3xl items-end gap-2 rounded-xl border border-rule bg-bg p-2 focus-within:border-accent">
          <label htmlFor="composer" className="sr-only">
            Message
          </label>
          <textarea
            id="composer"
            rows={1}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              // Enter sends; Shift+Enter adds a new line. `isComposing` is true while an input
              // method (e.g. Japanese or Chinese typing) is still building a character; Enter
              // then confirms the character and must not send.
              if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault()
                submit()
              }
            }}
            placeholder="Ask Arty…"
            // field-sizing-content: the box grows with its text, up to max-h-48, then scrolls.
            className="field-sizing-content max-h-48 min-h-10 flex-1 resize-none bg-transparent px-2 py-2 outline-none placeholder:text-muted"
          />
          <button type="submit" disabled={busy || !draft.trim()} className="btn-primary px-4 py-2">
            Send
          </button>
        </div>
        <p className="mx-auto mt-2 max-w-3xl text-xs text-muted">Enter to send · Shift+Enter for a new line</p>
      </form>
    </main>
  )
}

/**
 * One chat message. Four looks:
 *   your message        → right-aligned raised bubble, text exactly as typed
 *   an error            → bordered box in the danger colour
 *   reply not started   → a pulsing status line
 *   a reply             → plain text rendered as Markdown (no bubble, so long answers read like a
 *                          page), with Sources and/or an ApprovalCard if the reply has them
 */
function Bubble({
  message,
  waiting,
  mood,
  busy,
  onRespondApproval,
}: {
  message: Message
  waiting: boolean
  mood: Mood
  busy: boolean
  onRespondApproval: (approval: Approval, approve: boolean) => void
}) {
  if (message.role === 'user') {
    return (
      <div className="ml-auto max-w-[85%] rounded-2xl rounded-br-md bg-raised px-4 py-2.5 whitespace-pre-wrap">
        {message.content}
      </div>
    )
  }
  if (message.error) {
    return (
      <div role="alert" className="rounded-lg border border-danger/40 px-4 py-3 text-sm whitespace-pre-wrap text-danger">
        {message.content}
      </div>
    )
  }
  if (!message.content && waiting) {
    return (
      <div className="animate-pulse text-muted" role="status">
        {mood === 'searching' ? 'Searching the knowledge base…' : 'Thinking…'}
      </div>
    )
  }
  // `prose` (Tailwind typography plugin) styles the HTML that Markdown produces; `prose-invert`
  // is its light-text-on-dark variant.
  return (
    <div className="min-w-0">
      <div className="prose max-w-none prose-invert prose-p:my-2 prose-pre:bg-bg">
        <Markdown>{message.content}</Markdown>
      </div>
      {/* Set by App's `replace` handler (Phase 10, contracts.md § 14): the output guard swapped
          this reply's text after it had already streamed in. */}
      {message.redacted && <p className="mt-2 text-xs text-muted">Redacted by the output guard.</p>}
      {message.sources && message.sources.length > 0 && <Sources sources={message.sources} />}
      {message.approval && (
        <ApprovalCard
          approval={message.approval}
          decision={message.approvalDecision}
          busy={busy}
          onRespond={(approve) => onRespondApproval(message.approval!, approve)}
        />
      )}
    </div>
  )
}

/**
 * The passages a knowledge-base answer was built from, numbered like the [1] [2] citations in it.
 * Each is a native <details> element: click the line to unfold the chunk's text (no extra state needed).
 */
function Sources({ sources }: { sources: Source[] }) {
  return (
    <div className="mt-3 border-t border-rule pt-3">
      <p className="mb-2 text-xs font-medium tracking-wider text-muted uppercase">Sources</p>
      <ol className="space-y-1.5">
        {sources.map((s) => (
          <li key={s.n}>
            <details className="text-sm">
              <summary className="cursor-pointer hover:text-accent">
                <span className="mr-1.5 inline-block min-w-6 rounded bg-raised text-center font-mono text-xs">{s.n}</span>
                {s.source}
                {s.heading && ` › ${s.heading}`} <span className="font-mono text-xs text-muted">· {s.score.toFixed(2)}</span>
              </summary>
              <p className="mt-2 ml-8 rounded-lg border border-rule bg-bg p-3 text-xs whitespace-pre-wrap text-muted">
                {s.text}
              </p>
            </details>
          </li>
        ))}
      </ol>
    </div>
  )
}

/** The welcome screen: a headline, one line of intro, and example questions as pill buttons. */
function EmptyState({ onPick, disabled }: { onPick: (text: string) => void; disabled: boolean }) {
  return (
    <div className="flex flex-col gap-4 pt-[12vh]">
      <h2 className="text-3xl font-bold tracking-tight text-balance">What are we working on today?</h2>
      <p className="text-muted">
        Ask Arty anything. For studio policies he checks the knowledge base and shows his sources; the panel on the
        right shows every step he takes.
      </p>
      <div className="flex flex-wrap gap-2">
        {EXAMPLES.map((text) => (
          <button
            key={text}
            type="button"
            disabled={disabled}
            onClick={() => onPick(text)}
            className="rounded-full border border-rule px-3 py-1.5 text-left text-sm hover:border-accent hover:text-accent focus-visible:outline-2 focus-visible:outline-accent disabled:opacity-40"
          >
            {text}
          </button>
        ))}
      </div>
    </div>
  )
}
