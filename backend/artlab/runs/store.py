"""
runs/store.py — the run log: one row per chat request (Phase 13, docs/contracts.md § 15).

`RunStore` is plain `sqlite3`, in its own file and its own database (config.RUNS_DB), on purpose
(chosen, § 15): a run log must never be able to corrupt chat history, which lives in a separate
database of its own (config.DB_PATH, the LangGraph checkpointer). The two never share a connection
or a table.

Who uses this file: `api.py`'s `stream_run` calls `add()` once per request, at `done` or `error`
(including a resume or a pause for approval); the `GET /api/runs` and `GET /api/runs/{trace_id}`
endpoints call `list()` and `get()`. Every call goes through `asyncio.to_thread` (api.py), because
`sqlite3` itself is blocking — none of its methods are `async def`.
"""

import json
import sqlite3
import threading
from pathlib import Path

from artlab.config import RUNS_DB

# The row's columns, in the order they're stored — reused by every method below so a change here
# (adding a column, say) only has to happen in one place.
COLUMNS = (
    "trace_id", "thread_id", "started_at", "prompt", "answered_by", "status",
    "input_tokens", "output_tokens", "cost_usd", "ms", "steps", "trace_json",
)


class RunStore:
    """The run log: one row per chat request, keyed by its trace ID.

    Args:
        path: the sqlite file to open. The app uses config.RUNS_DB; tests pass a temp path, so a
              test run never touches the owner's real run history.

    `trace_json` holds that request's trace lines (the same dicts `stream_run` sends as `trace`
    SSE events), as a JSON array — the only column `list()` leaves out, since the Runs page's table
    only needs it once a row is actually clicked (`get()`).
    """

    def __init__(self, path: Path = RUNS_DB):
        path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: asyncio.to_thread can run each call on a different worker thread.
        # The lock below is what actually keeps access safe, the same one connection every time.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    trace_id TEXT PRIMARY KEY,
                    thread_id TEXT,
                    started_at REAL,
                    prompt TEXT,
                    answered_by TEXT,
                    status TEXT,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    cost_usd REAL,
                    ms INTEGER,
                    steps INTEGER,
                    trace_json TEXT
                )
                """
            )
            self._conn.commit()

    def add(self, run: dict) -> None:
        """Save one row, keyed by `run["trace_id"]`. `run` must have every column in COLUMNS
        (`trace_json` a JSON string, already `json.dumps`-ed by the caller). Called at most once per
        request, so a plain INSERT is enough — but OR REPLACE keeps this safe to call twice with the
        same trace_id (e.g. a retried recording) instead of raising.
        """
        placeholders = ", ".join("?" for _ in COLUMNS)
        with self._lock:
            self._conn.execute(
                f"INSERT OR REPLACE INTO runs ({', '.join(COLUMNS)}) VALUES ({placeholders})",
                [run[c] for c in COLUMNS],
            )
            self._conn.commit()

    def list(self, limit: int = 50) -> list[dict]:
        """The `limit` newest requests, newest first, without `trace_json` — what the Runs page's
        table shows. `limit` keeps this cheap even once the log holds thousands of rows.
        """
        cols = [c for c in COLUMNS if c != "trace_json"]
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {', '.join(cols)} FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(zip(cols, row)) for row in rows]

    def get(self, trace_id: str) -> dict | None:
        """One request's full row, `trace_json` parsed back into a `trace` list of trace-line dicts
        — or None if no run was ever recorded under this trace ID.
        """
        with self._lock:
            row = self._conn.execute(
                f"SELECT {', '.join(COLUMNS)} FROM runs WHERE trace_id = ?", (trace_id,)
            ).fetchone()
        if row is None:
            return None
        data = dict(zip(COLUMNS, row))
        data["trace"] = json.loads(data.pop("trace_json") or "[]")
        return data
