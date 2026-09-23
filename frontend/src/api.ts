/**
 * api.ts — everything the web page needs to talk to the FastAPI backend.
 *
 * The page never calls the backend's address directly: it calls `/api/...` on its own origin,
 * and the Vite dev server forwards (proxies) those requests to http://127.0.0.1:8000
 * (see vite.config.ts). That avoids browser cross-origin (CORS) problems during development.
 *
 * The types below mirror what backend/artlab/api.py sends. If you change one side, change both.
 */

/** One chat in the sidebar (GET /api/threads). `updated_at` is an ISO-8601 timestamp. */
export type Thread = { thread_id: string; title: string; updated_at: string }

/** One chat bubble. `error` marks a bubble that shows a failure instead of a real answer. */
export type Message = { role: 'user' | 'assistant'; content: string; error?: boolean }

/** One line in the trace panel, written by a graph node (e.g. stage "guard", status "ok"). */
export type TraceLine = { stage: string; status: string; detail: string; ms: number }

/** The totals for one run, shown as the trace panel's footer line. */
export type RunSummary = { input_tokens: number; output_tokens: number; cost_usd: number; ms: number }

/**
 * One event from the POST /api/chat stream. The `type` field says which kind it is, and
 * TypeScript uses it to know which other fields exist (a "discriminated union").
 * Order in a run: start → (trace | token)* → done   — or error instead of done.
 */
export type ChatEvent =
  | { type: 'start'; trace_id: string; thread_id: string }
  | ({ type: 'trace' } & TraceLine)
  | { type: 'token'; text: string }
  | ({ type: 'done' } & RunSummary)
  | { type: 'error'; message: string }

/** GET a URL and parse its JSON body; throws if the server answers with an error status. */
async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`${url} returned ${res.status}`)
  return res.json()
}

/** All chats for the sidebar, newest first. */
export const listThreads = () => getJson<Thread[]>('/api/threads')

/** One chat's full message history, used when you click a chat in the sidebar. */
export const getThread = (threadId: string) =>
  getJson<{ thread_id: string; messages: Message[] }>(`/api/threads/${threadId}`)

/**
 * Send one message and call `onEvent` for every event the server streams back.
 * Resolves when the stream ends; throws if the backend can't be reached at all.
 *
 * Why not the browser's built-in `EventSource`? It only supports GET requests, and we need to
 * POST the message. So we read the response body as a stream and split out the events ourselves.
 */
export async function streamChat(
  message: string,
  threadId: string | null,
  onEvent: (event: ChatEvent) => void,
): Promise<void> {
  // threadId null = new chat; the server creates an ID and sends it back in the `start` event.
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, thread_id: threadId }),
  })
  if (!res.ok || !res.body) throw new Error(`The server returned ${res.status}`)

  // Turn the raw byte stream into text, then read it piece by piece as it arrives.
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader()

  // Network pieces don't line up with event boundaries: one piece may hold half an event, or
  // three events. So we collect text in `buffer` and only cut out events once they're complete.
  let buffer = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += value

    // Each event ends with a blank line: "event: name\ndata: {json}\n\n". Cut out every
    // complete event in the buffer; leave any unfinished tail for the next piece.
    let end: number
    while ((end = buffer.indexOf('\n\n')) >= 0) {
      const block = buffer.slice(0, end)
      buffer = buffer.slice(end + 2)
      const name = block.match(/^event: (.*)$/m)?.[1]
      const data = block.match(/^data: (.*)$/m)?.[1]
      // Merge the event name into the JSON so it becomes e.g. { type: 'token', text: 'Hi' }.
      if (name && data) onEvent({ type: name, ...JSON.parse(data) } as ChatEvent)
    }
  }
}
