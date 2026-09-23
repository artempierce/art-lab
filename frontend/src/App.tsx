/**
 * App.tsx — the whole page: three panes side by side, and the state they share.
 *
 *   ┌──────────┬──────────────────────┬──────────────┐
 *   │ Sidebar  │ ChatView             │ TracePanel   │
 *   │ (chats)  │ (messages + input)   │ (run steps)  │
 *   └──────────┴──────────────────────┴──────────────┘
 *
 * All state lives here, in App, and flows down to the panes as props. The panes don't talk to
 * each other or to the backend directly; they call back into App (onSend, onOpen, onNew).
 * One owner means one place to look when something shows the wrong thing.
 */
import { useCallback, useEffect, useState } from 'react'
import {
  type ChatEvent,
  type Message,
  type RunSummary,
  type Thread,
  type TraceLine,
  getThread,
  listThreads,
  streamChat,
} from './api'
import type { ArtyMood } from './components/Arty'
import { ChatView } from './components/ChatView'
import { Sidebar } from './components/Sidebar'
import { TracePanel } from './components/TracePanel'

/**
 * One message you sent, and everything the backend reported while answering it: the trace ID
 * (to find it in LangSmith), the trace lines, the totals, and any error. The trace panel shows one
 * block per Run.
 */
export type Run = { traceId: string; prompt: string; lines: TraceLine[]; summary?: RunSummary; error?: string }

/**
 * How Arty should look right now, read from the latest run's trace lines:
 *   no runs yet                         → idle
 *   streaming, just routed to rag_agent → searching (the knowledge-base search is running)
 *   streaming, anything else            → thinking
 *   finished, the guard blocked it      → blocked
 *   finished with an answer             → happy
 */
function artyMood(runs: Run[], busy: boolean): ArtyMood {
  const run = runs[runs.length - 1]
  if (!run) return 'idle'
  if (busy) {
    const last = run.lines[run.lines.length - 1]
    return last?.stage === 'arty' && last.detail.startsWith('→ rag_agent') ? 'searching' : 'thinking'
  }
  if (run.lines.some((line) => line.status === 'blocked')) return 'blocked'
  return run.summary ? 'happy' : 'idle'
}

const BACKEND_DOWN = "Can't reach the backend. Start it with: cd backend && uv run uvicorn artlab.api:app --port 8000"

export default function App() {
  const [threads, setThreads] = useState<Thread[]>([]) // sidebar list
  const [threadsError, setThreadsError] = useState<string | null>(null) // shown in the sidebar if the list failed to load
  const [threadId, setThreadId] = useState<string | null>(null) // the open chat; null = a new, unsaved chat
  const [messages, setMessages] = useState<Message[]>([]) // bubbles in the open chat
  const [runs, setRuns] = useState<Run[]>([]) // trace panel blocks, for messages sent since the chat was opened
  const [busy, setBusy] = useState(false) // true while an answer is streaming; disables sending

  /** Reload the sidebar's chat list from the backend. */
  const refreshThreads = useCallback(() => {
    listThreads()
      .then((t) => {
        setThreads(t)
        setThreadsError(null)
      })
      .catch(() => setThreadsError(BACKEND_DOWN))
  }, [])

  // Load the chat list once, when the page opens.
  useEffect(() => {
    refreshThreads()
  }, [refreshThreads])

  // The two helpers below change the *last* run / the *last* message (the one currently
  // streaming). They pass a function to setRuns/setMessages instead of a new value, so React
  // hands them the latest state. That matters because events arrive faster than React
  // re-renders; using a stale copy would drop tokens.

  /** Update the run that's in progress (always the last one). */
  const updateLastRun = (change: (run: Run) => Run) =>
    setRuns((rs) => [...rs.slice(0, -1), change(rs[rs.length - 1])])

  /** Update the assistant reply that's streaming in (always the last message). */
  const updateReply = (change: (reply: Message) => Message) =>
    setMessages((ms) => [...ms.slice(0, -1), change(ms[ms.length - 1])])

  /** Show a failure in both places: the reply bubble (in red) and the trace panel. */
  const showError = (message: string) => {
    updateReply((m) => ({ ...m, content: m.content ? `${m.content}\n\n${message}` : message, error: true }))
    updateLastRun((r) => ({ ...r, error: message }))
  }

  /**
   * Send a message and stream the answer in.
   *
   * 1. Add your bubble plus an empty assistant bubble (it fills up as tokens arrive).
   * 2. Start a new trace-panel block for this run.
   * 3. Stream the response; each event updates the bubble or the trace block (see onEvent).
   * 4. When the stream ends, re-enable sending and refresh the sidebar (a new chat now has a title).
   */
  async function send(text: string) {
    setBusy(true)
    setMessages((ms) => [...ms, { role: 'user', content: text }, { role: 'assistant', content: '' }])
    setRuns((rs) => [...rs, { traceId: '', prompt: text, lines: [] }])

    // What each server event does to the page.
    const onEvent = (e: ChatEvent) => {
      switch (e.type) {
        case 'start':
          // For a new chat this is the moment it gets its ID; later messages then go to the same chat.
          setThreadId(e.thread_id)
          updateLastRun((r) => ({ ...r, traceId: e.trace_id }))
          break
        case 'trace':
          updateLastRun((r) => ({ ...r, lines: [...r.lines, e] }))
          break
        case 'token':
          updateReply((m) => ({ ...m, content: m.content + e.text }))
          break
        case 'done':
          // The totals go to the trace panel; the sources (if rag_agent answered) go under the reply.
          updateLastRun((r) => ({ ...r, summary: e }))
          updateReply((m) => ({ ...m, sources: e.sources }))
          break
        case 'error':
          showError(e.message)
          break
      }
    }

    try {
      await streamChat(text, threadId, onEvent)
    } catch {
      // streamChat only throws when the backend can't be reached at all.
      showError(BACKEND_DOWN)
    } finally {
      setBusy(false)
      refreshThreads()
    }
  }

  /** Open a chat from the sidebar: load its history and clear the trace panel. */
  function openThread(id: string) {
    if (busy || id === threadId) return
    setThreadId(id)
    setRuns([])
    getThread(id)
      .then((t) => setMessages(t.messages))
      .catch(() => setMessages([{ role: 'assistant', content: 'Could not load this chat.', error: true }]))
  }

  /** Start a fresh chat. It's only saved (and appears in the sidebar) once you send a message. */
  function newChat() {
    if (busy) return
    setThreadId(null)
    setMessages([])
    setRuns([])
  }

  const title = threads.find((t) => t.thread_id === threadId)?.title ?? 'New chat'
  const mood = artyMood(runs, busy)

  // Three columns on wide screens (lg = 1024px and up). Narrower, only the chat shows.
  return (
    <div className="grid h-full grid-cols-1 lg:grid-cols-[270px_minmax(0,1fr)_380px]">
      <Sidebar threads={threads} error={threadsError} activeId={threadId} onOpen={openThread} onNew={newChat} />
      <ChatView title={title} messages={messages} busy={busy} mood={mood} onSend={send} />
      <TracePanel runs={runs} busy={busy} />
    </div>
  )
}
