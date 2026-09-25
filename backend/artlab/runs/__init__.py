"""
runs/ — the run log (Phase 13, docs/contracts.md § 15): one row per chat request, for the Runs page.

Where this sits in the message flow: `api.py`'s `stream_run` writes one row here, at `done` or
`error`, every time it finishes streaming a request (a new message, a resume, or an approval pause).
`store.py` is the one file in here so far — the sqlite-backed `RunStore`.
"""
