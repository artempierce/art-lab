"""
conftest.py — pytest's shared setup for every test in this folder (pytest loads it automatically).

One job only: work around a crash that happens *after* the tests, on macOS.

What goes wrong: when Python shuts down after a test run, native libraries we load (onnxruntime for the
local classifier and embeddings, Chroma) tear down their background threads in an order macOS's C++
runtime sometimes rejects: "libc++abi: … recursive_mutex lock failed". The process then aborts with exit
code 134 even though every test passed, so roughly one local run in four *looks* failed.

The fix: once pytest has printed its summary, end the process straight away with pytest's own exit code
(`os._exit`), skipping the native teardown that crashes. Only on macOS: CI runs Linux, which doesn't have
the problem, so CI keeps a completely normal shutdown.
"""

import os
import sys

import pytest

# pytest's exit status for this run (0 = all passed), captured when the session finishes.
_exit_status: int | None = None


@pytest.fixture(autouse=True)
def isolated_default_stores(tmp_path, monkeypatch):
    """Every test gets a private run log and memory store unless it passes its own.

    `create_app(runs=None, memory=None)` means "open the real ones in data/" (the app's own defaults).
    A test that forgets to pass them would otherwise write rows into your real run log (data/runs.db)
    and facts into your real long-term memory. This swaps api.py's defaults for temp-folder versions,
    so forgetting can't leak test data into the real app.
    """
    import artlab.api as api
    from artlab.memory.store import MemoryStore
    from artlab.runs.store import RunStore

    monkeypatch.setattr(api, "RunStore", lambda *a, **kw: RunStore(*(a or (tmp_path / "runs.db",)), **kw))
    monkeypatch.setattr(api, "MemoryStore", lambda *a, **kw: MemoryStore(*(a or (tmp_path / "chroma-memory",)), **kw))


def pytest_sessionfinish(session, exitstatus):
    """Remember pytest's own verdict for the run, so the early exit below reports exactly that."""
    global _exit_status
    _exit_status = int(exitstatus)


def pytest_unconfigure(config):
    """Runs last, after the summary is printed. On macOS, exit now with pytest's status (see the header)."""
    if sys.platform == "darwin" and _exit_status is not None:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(_exit_status)
