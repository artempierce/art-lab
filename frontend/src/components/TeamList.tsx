/**
 * TeamList.tsx — the sidebar's collapsible "Team" section: every agent Arty can route to, and the
 * tools each one may use. Built from GET /api/agents (backend/artlab/agents/capabilities.py), which
 * builds the same list from the worker registry and the tool registry — so this can't drift from
 * what the code actually does, and it's the same list Arty himself answers "what can you do" from
 * (docs/contracts.md § 9).
 *
 * Fetches its own data once, on mount: nothing else on the page needs an agent list, so there's no
 * reason to thread it through App's shared state.
 */
import { useEffect, useState } from 'react'
import { type Agent, getAgents } from '../api'

// A tool's tier, coloured the same way the trace panel colours "ok" (safe) vs. a warning: read-only
// tools only look things up; "changes data" tools need your approval before they exist (Phase 6).
const TIER_COLOR: Record<string, string> = { 'read-only': 'text-t-tool', 'changes data': 'text-t-human' }

export function TeamList() {
  const [agents, setAgents] = useState<Agent[]>([])
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    getAgents()
      .then(setAgents)
      .catch(() => setFailed(true))
  }, [])

  if (failed || agents.length === 0) return null // backend not reachable yet, or still loading

  return (
    <details className="border-t-2 border-ink pt-4">
      <summary className="cursor-pointer text-xs font-medium">Team</summary>
      <ul className="mt-2 space-y-2">
        {agents.map((agent) => (
          <li key={agent.name} className="card p-2">
            <p className="text-sm font-semibold">{agent.name}</p>
            <p className="text-xs text-muted">{agent.description}</p>
            {agent.tools.length > 0 && (
              <ul className="mt-1.5 space-y-0.5">
                {agent.tools.map((tool) => (
                  <li key={tool.name} className="flex items-baseline gap-1.5 text-xs">
                    <span className="font-medium">{tool.name}</span>
                    <span className={TIER_COLOR[tool.tier] ?? 'text-muted'}>{tool.tier}</span>
                  </li>
                ))}
              </ul>
            )}
          </li>
        ))}
      </ul>
    </details>
  )
}
