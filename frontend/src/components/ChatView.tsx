/**
 * ChatView.tsx — the middle pane: a header with Arty's status, a scrolling ticker, the message bubbles,
 * and the box you type in. Before the first message it shows a welcome screen starring Arty.
 *
 * Holds one piece of state of its own: `draft`, the text in the input box. Everything else
 * comes from App as props.
 */
import { useEffect, useRef, useState } from 'react'
import Markdown from 'react-markdown'
import type { Message, Source } from '../api'
import { Arty, type ArtyMood } from './Arty'

type Props = {
  title: string // shown in the header
  messages: Message[] // bubbles to show
  busy: boolean // an answer is streaming; sending is disabled
  mood: ArtyMood // how Arty looks right now (worked out in App)
  onSend: (text: string) => void // called with the message to send
}

// Clickable starters on the welcome screen. Clicking one sends it right away. Each gets its own colour.
const EXAMPLES = [
  { text: 'What is our sponsorship disclosure rule?', color: 'bg-peach' },
  { text: 'Write a hook for a video about cable management', color: 'bg-blush' },
  { text: 'Who approves a $900 equipment purchase?', color: 'bg-mint' },
]

// What the header says next to Arty for each mood.
const STATUS: Record<ArtyMood, string> = {
  idle: 'Ready when you are',
  thinking: 'Arty is thinking…',
  searching: 'Searching the knowledge base…',
  happy: 'Done! Ask another?',
  blocked: 'Blocked by the guard',
}

// The ticker text. It's rendered twice in a row so the CSS animation can loop seamlessly.
const TICKER = 'Ask Arty anything ★ Policies & processes ★ Hooks & titles ★ Every step shows in the trace ★ '

export function ChatView({ title, messages, busy, mood, onSend }: Props) {
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
    <main className="flex min-h-0 min-w-0 flex-col bg-cream">
      {/* Header: chat title in outlined type, Arty's current status on the right. */}
      <header className="flex items-center justify-between gap-4 border-b-3 border-ink bg-salmon px-6 py-3">
        <h1 className="outlined truncate text-2xl">{title}</h1>
        <div className="flex shrink-0 items-center gap-2 rounded-full border-3 border-ink bg-cream py-0.5 pr-4 pl-1" role="status">
          <Arty mood={mood} size={34} />
          <span className="text-sm font-extrabold">{STATUS[mood]}</span>
        </div>
      </header>

      {/* Ticker (decorative, so hidden from screen readers; stops moving with reduced motion). */}
      <div className="overflow-hidden border-b-3 border-ink bg-ink py-1.5 text-cream" aria-hidden="true">
        <div className="marquee-track font-display text-sm tracking-wider whitespace-nowrap uppercase">
          <span>{TICKER.repeat(4)}</span>
          <span>{TICKER.repeat(4)}</span>
        </div>
      </div>

      {/* The message area scrolls; the header and the input box stay fixed. */}
      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto flex max-w-3xl flex-col gap-5 px-6 py-8">
          {messages.length === 0 ? (
            <EmptyState onPick={onSend} disabled={busy} />
          ) : (
            // Only the last bubble can be "waiting" (the reply that's streaming in).
            messages.map((m, i) => (
              <Bubble key={i} message={m} waiting={busy && i === messages.length - 1} mood={mood} />
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
        className="border-t-3 border-ink bg-peri px-6 py-4"
      >
        <div className="sticker mx-auto flex max-w-3xl items-end gap-2 bg-white p-2">
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
            className="field-sizing-content max-h-48 min-h-10 flex-1 resize-none bg-transparent px-2 py-2 font-semibold outline-none placeholder:text-muted"
          />
          <button type="submit" disabled={busy || !draft.trim()} className="btn-pop bg-salmon px-5 py-2 text-lg text-white">
            Send
          </button>
        </div>
        <p className="mx-auto mt-2 max-w-3xl text-xs font-bold">Enter to send · Shift+Enter for a new line</p>
      </form>
    </main>
  )
}

/**
 * One chat bubble. Four looks:
 *   your message        → right-aligned periwinkle sticker, text exactly as typed
 *   an error            → pink sticker with red text
 *   reply not started   → Arty (thinking or searching) with a status line
 *   a reply             → Arty's avatar + a white sticker, rendered as Markdown, with Sources if any
 */
function Bubble({ message, waiting, mood }: { message: Message; waiting: boolean; mood: ArtyMood }) {
  if (message.role === 'user') {
    return (
      <div className="sticker ml-auto max-w-[85%] rounded-br-md bg-peri px-4 py-2.5 font-bold whitespace-pre-wrap">
        {message.content}
      </div>
    )
  }
  if (message.error) {
    return (
      <div role="alert" className="sticker bg-blush px-4 py-3 text-sm font-bold whitespace-pre-wrap text-danger">
        {message.content}
      </div>
    )
  }
  if (!message.content && waiting) {
    return (
      <div className="flex items-center gap-3" role="status">
        <Arty mood={mood === 'searching' ? 'searching' : 'thinking'} size={56} />
        <span className="font-extrabold">{mood === 'searching' ? 'Searching the knowledge base…' : 'Thinking…'}</span>
      </div>
    )
  }
  // `prose` (Tailwind typography plugin) styles the HTML that Markdown produces.
  return (
    <div className="flex items-start gap-3">
      <Arty mood="idle" size={44} className="mt-1 shrink-0" />
      <div className="sticker min-w-0 flex-1 rounded-tl-md bg-white px-4 py-3">
        <div className="prose max-w-none prose-p:my-2 prose-strong:text-ink">
          <Markdown>{message.content}</Markdown>
        </div>
        {message.sources && message.sources.length > 0 && <Sources sources={message.sources} />}
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
    <div className="mt-3 border-t-2 border-dashed border-ink pt-3">
      <p className="mb-2 font-display text-sm tracking-wider uppercase">Sources</p>
      <ol className="space-y-1.5">
        {sources.map((s) => (
          <li key={s.n}>
            <details className="text-sm">
              <summary className="cursor-pointer font-semibold hover:underline">
                <span className="mr-1 rounded-md border-2 border-ink bg-peach px-1.5 font-black">{s.n}</span>
                {s.source}
                {s.heading && ` › ${s.heading}`} <span className="font-mono text-xs text-muted">· {s.score.toFixed(2)}</span>
              </summary>
              <p className="mt-2 ml-7 rounded-xl border-2 border-ink bg-mint p-3 text-xs whitespace-pre-wrap">{s.text}</p>
            </details>
          </li>
        ))}
      </ol>
    </div>
  )
}

/** The welcome screen shown before the first message: Arty, a big outlined hello, and example prompts. */
function EmptyState({ onPick, disabled }: { onPick: (text: string) => void; disabled: boolean }) {
  return (
    <div className="flex flex-col items-center gap-4 pt-[6vh] text-center">
      <Arty size={170} />
      <h2 className="outlined text-6xl leading-none">Hey, I'm Arty!</h2>
      <p className="max-w-md font-bold">
        Ask me anything. For questions about the studio's policies I'll check the knowledge base and show my
        sources — and the panel on the right shows every step I take.
      </p>
      <div className="mt-2 flex flex-wrap justify-center gap-3">
        {EXAMPLES.map(({ text, color }) => (
          <button
            key={text}
            type="button"
            disabled={disabled}
            onClick={() => onPick(text)}
            className={`sticker ${color} px-4 py-2 text-left text-sm font-extrabold transition-transform hover:-translate-y-0.5 focus-visible:outline-3 focus-visible:outline-offset-2 focus-visible:outline-ink`}
          >
            {text}
          </button>
        ))}
      </div>
    </div>
  )
}
