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
import { ChatView } from './components/ChatView'
import { Sidebar } from './components/Sidebar'
import { TracePanel } from './components/TracePanel'

// One message you sent, and everything the backend reported while answering it.
export type Run = { traceId: string; prompt: string; lines: TraceLine[]; summary?: RunSummary; error?: string }

const BACKEND_DOWN = "Can't reach the backend. Start it with: cd backend && uv run uvicorn artlab.api:app --port 8000"

export default function App() {
  const [threads, setThreads] = useState<Thread[]>([])
  const [threadsError, setThreadsError] = useState<string | null>(null)
  const [threadId, setThreadId] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [runs, setRuns] = useState<Run[]>([])
  const [busy, setBusy] = useState(false)

  const refreshThreads = useCallback(() => {
    listThreads()
      .then((t) => {
        setThreads(t)
        setThreadsError(null)
      })
      .catch(() => setThreadsError(BACKEND_DOWN))
  }, [])

  useEffect(() => {
    refreshThreads()
  }, [refreshThreads])

  const updateLastRun = (change: (run: Run) => Run) =>
    setRuns((rs) => [...rs.slice(0, -1), change(rs[rs.length - 1])])

  const updateReply = (change: (reply: Message) => Message) =>
    setMessages((ms) => [...ms.slice(0, -1), change(ms[ms.length - 1])])

  const showError = (message: string) => {
    updateReply((m) => ({ ...m, content: m.content ? `${m.content}\n\n${message}` : message, error: true }))
    updateLastRun((r) => ({ ...r, error: message }))
  }

  async function send(text: string) {
    setBusy(true)
    setMessages((ms) => [...ms, { role: 'user', content: text }, { role: 'assistant', content: '' }])
    setRuns((rs) => [...rs, { traceId: '', prompt: text, lines: [] }])

    const onEvent = (e: ChatEvent) => {
      switch (e.type) {
        case 'start':
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
          updateLastRun((r) => ({ ...r, summary: e }))
          break
        case 'error':
          showError(e.message)
          break
      }
    }

    try {
      await streamChat(text, threadId, onEvent)
    } catch {
      showError(BACKEND_DOWN)
    } finally {
      setBusy(false)
      refreshThreads()
    }
  }

  function openThread(id: string) {
    if (busy || id === threadId) return
    setThreadId(id)
    setRuns([])
    getThread(id)
      .then((t) => setMessages(t.messages))
      .catch(() => setMessages([{ role: 'assistant', content: 'Could not load this chat.', error: true }]))
  }

  function newChat() {
    if (busy) return
    setThreadId(null)
    setMessages([])
    setRuns([])
  }

  const title = threads.find((t) => t.thread_id === threadId)?.title ?? 'New chat'

  return (
    <div className="grid h-full grid-cols-1 lg:grid-cols-[250px_minmax(0,1fr)_380px]">
      <Sidebar threads={threads} error={threadsError} activeId={threadId} onOpen={openThread} onNew={newChat} />
      <ChatView title={title} messages={messages} busy={busy} onSend={send} />
      <TracePanel runs={runs} busy={busy} />
    </div>
  )
}
