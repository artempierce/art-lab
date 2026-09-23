import { useEffect, useRef, useState } from 'react'
import Markdown from 'react-markdown'
import type { Message } from '../api'

type Props = { title: string; messages: Message[]; busy: boolean; onSend: (text: string) => void }

const EXAMPLES = [
  'Give me 5 video ideas about budget desk setups',
  'Write a hook for a video about cable management',
  'Fix my grammar: me and him goes to shoot video tomorrow',
]

export function ChatView({ title, messages, busy, onSend }: Props) {
  const [draft, setDraft] = useState('')
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: 'end' })
  }, [messages])

  function submit() {
    const text = draft.trim()
    if (!text || busy) return
    setDraft('')
    onSend(text)
  }

  return (
    <main className="flex min-h-0 min-w-0 flex-col bg-surface">
      <header className="border-b border-rule px-6 py-3">
        <h1 className="truncate font-display text-lg font-semibold">{title}</h1>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto flex max-w-3xl flex-col gap-5 px-6 py-8">
          {messages.length === 0 ? (
            <EmptyState onPick={onSend} disabled={busy} />
          ) : (
            messages.map((m, i) => <Bubble key={i} message={m} waiting={busy && i === messages.length - 1} />)
          )}
          <div ref={endRef} />
        </div>
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault()
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
              if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault()
                submit()
              }
            }}
            placeholder="Message Art Lab…"
            className="field-sizing-content max-h-48 min-h-10 flex-1 resize-none bg-transparent px-2 py-2 outline-none placeholder:text-muted"
          />
          <button
            type="submit"
            disabled={busy || !draft.trim()}
            className="rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-accent-ink focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:opacity-40"
          >
            Send
          </button>
        </div>
        <p className="mx-auto mt-2 max-w-3xl text-xs text-muted">Enter to send · Shift+Enter for a new line</p>
      </form>
    </main>
  )
}

function Bubble({ message, waiting }: { message: Message; waiting: boolean }) {
  if (message.role === 'user') {
    return (
      <div className="ml-auto max-w-[85%] rounded-2xl rounded-br-md bg-accent px-4 py-2.5 whitespace-pre-wrap text-accent-ink">
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
        Thinking…
      </div>
    )
  }
  return (
    <div className="prose max-w-none prose-neutral dark:prose-invert prose-p:my-2 prose-pre:bg-bg prose-pre:text-ink">
      <Markdown>{message.content}</Markdown>
    </div>
  )
}

function EmptyState({ onPick, disabled }: { onPick: (text: string) => void; disabled: boolean }) {
  return (
    <div className="flex flex-col gap-4 pt-[12vh]">
      <h2 className="font-display text-3xl font-bold tracking-tight text-balance">What are we making today?</h2>
      <p className="text-muted">Ask anything. The panel on the right shows each step the backend takes to answer.</p>
      <div className="flex flex-wrap gap-2">
        {EXAMPLES.map((text) => (
          <button
            key={text}
            type="button"
            disabled={disabled}
            onClick={() => onPick(text)}
            className="rounded-full border border-rule px-3 py-1.5 text-left text-sm hover:border-accent hover:text-accent focus-visible:outline-2 focus-visible:outline-accent"
          >
            {text}
          </button>
        ))}
      </div>
    </div>
  )
}
