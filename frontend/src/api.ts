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

/** One tool an agent may use (GET /api/agents). `tier` is "read-only" or "changes data". */
export type AgentTool = { name: string; tier: string; description: string }

/** One member of the team (GET /api/agents): Arty (no tools) or a worker he can route to. */
export type Agent = { name: string; description: string; tools: AgentTool[] }

/**
 * One knowledge-base passage an answer was based on. `n` is its citation number ([1], [2], …) in the
 * answer; `score` is how relevant search judged it (0–1); `text` is the chunk itself.
 */
export type Source = { n: number; source: string; heading: string; text: string; score: number }

/**
 * One fact Arty remembers about you (GET /api/memory, Phase 12 Memory page, contracts.md § 15).
 * `created_at` is a Unix timestamp (seconds); `thread_id` is the chat it was learned in.
 */
export type Fact = { key: string; value: string; created_at: number; thread_id: string }

/**
 * A mutating tool (e.g. save_ideas) waiting for your Approve/Reject click (Phase 6, contracts.md §
 * 10). `args` are the exact arguments the tool would run with; `tainted` is true when this chat has
 * read untrusted content, and `taint_sources` says where from (e.g. "fetch_comments").
 */
export type Approval = {
  id: string
  agent: string
  tool: string
  args: Record<string, unknown>
  tainted: boolean
  taint_sources: string[]
}

/**
 * One chat bubble. `error` marks a bubble that shows a failure instead of a real answer; `sources`
 * are the passages rag_agent answered from (empty for other answers). `approval` is attached by an
 * `approval` event and shows the Approve/Reject card (ApprovalCard.tsx); `approvalDecision` is set
 * once you've clicked one of its buttons, so the card can show its outcome and stay disabled.
 * `redacted` is set by a `replace` event (Phase 10, contracts.md § 14): the output guard swapped
 * this reply's text after it had already streamed to the page.
 */
export type Message = {
  role: 'user' | 'assistant'
  content: string
  error?: boolean
  sources?: Source[]
  approval?: Approval
  approvalDecision?: 'approved' | 'rejected'
  redacted?: boolean
}

/** One line in the trace panel, written by a graph node (e.g. stage "guard", status "ok"). */
export type TraceLine = { stage: string; status: string; detail: string; ms: number }

/** The totals for one run, shown as the trace panel's footer line. */
export type RunSummary = { input_tokens: number; output_tokens: number; cost_usd: number; ms: number }

/**
 * One event from the POST /api/chat (or /api/chat/resume) stream. The `type` field says which kind
 * it is, and TypeScript uses it to know which other fields exist (a "discriminated union").
 * Order in a run: start → (trace | token)* → done   — or error instead of done. A run that calls a
 * mutating tool pauses instead: start → (trace | token)* → approval → done (the reply so far is
 * "waiting for your approval"); resuming it with resumeChat below starts the same sequence again.
 * A run the output guard redacted inserts one more event before done: start → (trace | token)* →
 * replace → done (contracts.md § 14).
 */
export type ChatEvent =
  | { type: 'start'; trace_id: string; thread_id: string }
  | ({ type: 'trace' } & TraceLine)
  | { type: 'token'; text: string }
  | ({ type: 'approval' } & Approval)
  | { type: 'replace'; text: string }
  | ({ type: 'done'; sources: Source[] } & RunSummary)
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

/** The whole team and their tools, for the sidebar's Team panel (X3). */
export const getAgents = () => getJson<Agent[]>('/api/agents')

/** Every fact Arty remembers, newest first (the Memory page, Phase 12). */
export const getMemory = () => getJson<Fact[]>('/api/memory')

/** Edit a fact's value; throws (with the response's status in the message) if the key is unknown. */
export async function updateMemory(key: string, value: string): Promise<Fact> {
  const res = await fetch(`/api/memory/${encodeURIComponent(key)}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ value }),
  })
  if (!res.ok) throw new Error(`PUT /api/memory/${key} returned ${res.status}`)
  return res.json()
}

/** Delete a fact; throws (with the response's status in the message) if the key is unknown. */
export async function deleteMemory(key: string): Promise<void> {
  const res = await fetch(`/api/memory/${encodeURIComponent(key)}`, { method: 'DELETE' })
  if (!res.ok) throw new Error(`DELETE /api/memory/${key} returned ${res.status}`)
}

/**
 * Read an SSE response body and call `onEvent` for every event it contains. Shared by streamChat
 * and resumeChat below, since both talk to endpoints that stream the same event shapes — this is
 * the one place that knows the wire format, so it isn't duplicated between them.
 *
 * Why not the browser's built-in `EventSource`? It only supports GET requests, and both endpoints
 * are POSTs. So we read the response body as a stream and split out the events ourselves.
 */
async function readEventStream(res: Response, onEvent: (event: ChatEvent) => void): Promise<void> {
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

/**
 * Send one message and call `onEvent` for every event the server streams back.
 * Resolves when the stream ends; throws if the backend can't be reached at all.
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
  await readEventStream(res, onEvent)
}

/**
 * Answer a pending approval (Approve or Reject the tool it named) and call `onEvent` for the
 * continuation's events — the same shapes streamChat delivers, since the backend resumes the same
 * graph run. `id` must match the pending request's id (from its `approval` event); a stale id is
 * treated as a reject by the backend.
 */
export async function resumeChat(
  threadId: string,
  id: string,
  approve: boolean,
  onEvent: (event: ChatEvent) => void,
): Promise<void> {
  const res = await fetch('/api/chat/resume', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ thread_id: threadId, id, approve }),
  })
  await readEventStream(res, onEvent)
}
