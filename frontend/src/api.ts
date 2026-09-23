// Client for the FastAPI backend. Vite proxies /api to it (see vite.config.ts).

export type Thread = { thread_id: string; title: string; updated_at: string }
export type Message = { role: 'user' | 'assistant'; content: string; error?: boolean }

export type TraceLine = { stage: string; status: string; detail: string; ms: number }
export type RunSummary = { input_tokens: number; output_tokens: number; cost_usd: number; ms: number }

// One event per server-sent event from POST /api/chat.
export type ChatEvent =
  | { type: 'start'; trace_id: string; thread_id: string }
  | ({ type: 'trace' } & TraceLine)
  | { type: 'token'; text: string }
  | ({ type: 'done' } & RunSummary)
  | { type: 'error'; message: string }

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`${url} returned ${res.status}`)
  return res.json()
}

export const listThreads = () => getJson<Thread[]>('/api/threads')

export const getThread = (threadId: string) =>
  getJson<{ thread_id: string; messages: Message[] }>(`/api/threads/${threadId}`)

export async function streamChat(
  message: string,
  threadId: string | null,
  onEvent: (event: ChatEvent) => void,
): Promise<void> {
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, thread_id: threadId }),
  })
  if (!res.ok || !res.body) throw new Error(`The server returned ${res.status}`)

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader()
  let buffer = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += value
    // Events are separated by a blank line: "event: name\ndata: {json}\n\n"
    let end: number
    while ((end = buffer.indexOf('\n\n')) >= 0) {
      const block = buffer.slice(0, end)
      buffer = buffer.slice(end + 2)
      const name = block.match(/^event: (.*)$/m)?.[1]
      const data = block.match(/^data: (.*)$/m)?.[1]
      if (name && data) onEvent({ type: name, ...JSON.parse(data) } as ChatEvent)
    }
  }
}
