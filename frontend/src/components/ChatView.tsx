/**
 * ChatView.tsx — the middle pane: a header with Arty's status, the message cards, and the box you type
 * in. Before the first message it shows a welcome screen: a huge headline crossed by a tilted ticker
 * ribbon, Arty, and example questions as big clickable lines.
 *
 * Holds one piece of state of its own: `draft`, the text in the input box. Everything else
 * comes from App as props.
 */
import { useEffect, useRef, useState } from 'react'
import Markdown from 'react-markdown'
import type { Approval, Message, Source } from '../api'
import { ApprovalCard } from './ApprovalCard'
import { Arty, type ArtyMood } from './Arty'

type Props = {
  title: string // shown in the header
  messages: Message[] // bubbles to show
  busy: boolean // an answer is streaming; sending is disabled
  mood: ArtyMood // how Arty looks right now (worked out in App)
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

// What the header says next to Arty for each mood.
const STATUS: Record<ArtyMood, string> = {
  idle: 'Ready when you are',
  thinking: 'Arty is thinking…',
  searching: 'Searching the knowledge base…',
  happy: 'Done! Ask another?',
  blocked: 'Blocked by the guard',
}

// The ribbon's ticker text. It's rendered twice in a row so the CSS animation can loop seamlessly.
const TICKER = 'Ask me anything · Policies & processes · Hooks & titles · Every step shows in the trace · '

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
    <main className="flex min-h-0 min-w-0 flex-col bg-sage">
      {/* Header: chat title in heavy display type, Arty's current status on the right. */}
      <header className="flex items-center justify-between gap-4 border-b-2 border-ink px-6 py-3">
        <h1 className="display truncate text-3xl uppercase">{title}</h1>
        <div className="card flex shrink-0 items-center gap-2 py-0.5 pr-4 pl-1" role="status">
          <Arty mood={mood} size={34} shadow={false} />
          <span className="text-sm font-semibold">{STATUS[mood]}</span>
        </div>
      </header>

      {/* The message area scrolls; the header and the input box stay fixed. */}
      <div className="min-h-0 flex-1 overflow-x-hidden overflow-y-auto">
        {messages.length === 0 ? (
          <EmptyState onPick={onSend} disabled={busy} />
        ) : (
          <div className="mx-auto flex max-w-3xl flex-col gap-6 px-6 py-8">
            {/* Only the last bubble can be "waiting" (the reply that's streaming in). */}
            {messages.map((m, i) => (
              <Bubble
                key={i}
                message={m}
                waiting={busy && i === messages.length - 1}
                mood={mood}
                busy={busy}
                onRespondApproval={onRespondApproval}
              />
            ))}
            <div ref={endRef} />
          </div>
        )}
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault() // stop the browser's default full-page form submit
          submit()
        }}
        className="border-t-2 border-ink px-6 py-4"
      >
        <div className="card mx-auto flex max-w-3xl items-end gap-2 p-2">
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
          <button type="submit" disabled={busy || !draft.trim()} className="btn-black px-6 py-2 text-lg">
            Send
          </button>
        </div>
        <p className="mx-auto mt-2 max-w-3xl text-xs">Enter to send · Shift+Enter for a new line</p>
      </form>
    </main>
  )
}

/**
 * One chat message. Four looks:
 *   your message        → right-aligned black card, text exactly as typed
 *   an error            → card with red text
 *   reply not started   → Arty (thinking or searching) with a status line
 *   a reply             → Arty + a white card, rendered as Markdown, with Sources and/or an
 *                          ApprovalCard if the reply has them
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
  mood: ArtyMood
  busy: boolean
  onRespondApproval: (approval: Approval, approve: boolean) => void
}) {
  if (message.role === 'user') {
    return (
      <div className="card ml-auto max-w-[85%] bg-ink px-4 py-2.5 whitespace-pre-wrap text-white">{message.content}</div>
    )
  }
  if (message.error) {
    return (
      <div role="alert" className="card bg-[#fde2dc] px-4 py-3 text-sm font-medium whitespace-pre-wrap text-danger">
        {message.content}
      </div>
    )
  }
  if (!message.content && waiting) {
    return (
      <div className="flex items-center gap-3" role="status">
        <Arty mood={mood === 'searching' ? 'searching' : 'thinking'} size={64} />
        <span className="display text-2xl">{mood === 'searching' ? 'Searching the knowledge base…' : 'Thinking…'}</span>
      </div>
    )
  }
  // `prose` (Tailwind typography plugin) styles the HTML that Markdown produces.
  return (
    <div className="flex items-start gap-3">
      <Arty mood="idle" size={52} className="shrink-0" />
      <div className="card min-w-0 flex-1 px-4 py-3">
        <div className="prose max-w-none prose-p:my-2">
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
    </div>
  )
}

/**
 * The passages a knowledge-base answer was built from, numbered like the [1] [2] citations in it.
 * Each is a native <details> element: click the line to unfold the chunk's text (no extra state needed).
 */
function Sources({ sources }: { sources: Source[] }) {
  return (
    <div className="mt-3 border-t-2 border-ink pt-3">
      <p className="display mb-2 text-lg uppercase">Sources</p>
      <ol className="space-y-1.5">
        {sources.map((s) => (
          <li key={s.n}>
            <details className="text-sm">
              <summary className="cursor-pointer hover:underline">
                <span className="mr-1.5 inline-block min-w-6 border-2 border-ink bg-orange text-center text-xs font-bold">{s.n}</span>
                {s.source}
                {s.heading && ` › ${s.heading}`} <span className="font-mono text-xs text-muted">· {s.score.toFixed(2)}</span>
              </summary>
              <p className="mt-2 ml-8 border-2 border-ink bg-sage p-3 text-xs whitespace-pre-wrap">{s.text}</p>
            </details>
          </li>
        ))}
      </ol>
    </div>
  )
}

/** The welcome screen: headline crossed by the ribbon, Arty, a short intro, and example questions. */
function EmptyState({ onPick, disabled }: { onPick: (text: string) => void; disabled: boolean }) {
  return (
    <div className="flex flex-col items-center px-6 pt-8 pb-12 text-center">
      {/* Headline with the tilted ticker ribbon laid across it (decorative, hidden from screen readers). */}
      <div className="relative w-full">
        {/* Extra line spacing leaves a gap between the two lines for the ribbon to cross. */}
        <h2 className="display text-[clamp(3rem,6.5vw,5.75rem)] leading-[1.25] uppercase">
          Hey, I'm
          <br />
          Arty!
        </h2>
        <div className="ribbon absolute top-1/2 -right-10 -left-10 -translate-y-1/2 overflow-hidden py-1" aria-hidden="true">
          <div className="marquee-track display text-lg whitespace-nowrap uppercase">
            <span>{TICKER.repeat(4)}</span>
            <span>{TICKER.repeat(4)}</span>
          </div>
        </div>
      </div>

      <Arty size={150} className="mt-4" />

      <p className="display mt-4 text-3xl">Come ask me anything.</p>
      <p className="mt-2 max-w-md text-sm">
        For questions about the studio's policies I check the knowledge base and show my sources. The panel on the
        right shows every step I take.
      </p>

      <hr className="my-6 w-full max-w-md border-t-2 border-ink" />

      <p className="text-xs">Try asking:</p>
      <ul className="mt-2 flex flex-col gap-1">
        {EXAMPLES.map((text) => (
          <li key={text}>
            <button
              type="button"
              disabled={disabled}
              onClick={() => onPick(text)}
              className="display text-2xl decoration-2 underline-offset-4 hover:underline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ink disabled:opacity-40"
            >
              {text}
            </button>
          </li>
        ))}
      </ul>
    </div>
  )
}
