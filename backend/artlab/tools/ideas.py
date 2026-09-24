"""
tools/ideas.py — save_ideas, content_ideator's first mutating tool (Phase 6, docs/contracts.md § 10).

Everything content_ideator did before this was read-only text generation. save_ideas is the first
tool that changes something on disk, which is exactly why the tool gateway (tools/registry.py) never
lets a model run it directly: it always raises ApprovalRequired, and it only actually runs from the
graph's `approval` node, after your Approve click.
"""

import datetime
from pathlib import Path
from typing import Callable


def make_save_ideas(dir: Path) -> Callable[[str], str]:
    """Build a `save_ideas(ideas: str) -> str` tool that writes into `dir`.

    A *factory*, not a plain function, so the folder it writes to can be swapped in: the real app
    registers one pointed at data/ideas/ (tools/catalog.py), and tests register one pointed at a temp
    folder, so running the test suite never touches the repo's own data. The tool gateway only ever
    sees the returned function — it can't tell the two apart.

    Example: make_save_ideas(Path("data/ideas"))("Idea one...") writes to
             "data/ideas/2026-09-24.md" and returns "Saved to data/ideas/2026-09-24.md".
    """

    def save_ideas(ideas: str) -> str:
        """Append `ideas` as a new dated section of today's ideas file (docs/contracts.md § 10),
        creating the folder and file the first time. Markdown in `ideas` is written as-is: it's the
        model's own writing from this same chat, not outside text, so there's nothing to escape —
        and the tool is registered with untrusted_output=False, so the gateway doesn't wrap the
        confirmation text this returns either.
        """
        dir.mkdir(parents=True, exist_ok=True)
        date = datetime.date.today().isoformat()
        with (dir / f"{date}.md").open("a", encoding="utf-8") as f:
            f.write(f"## {date}\n\n{ideas}\n\n")
        return f"Saved to data/ideas/{date}.md"

    return save_ideas
