"""
skills/loader.py — reads every `backend/skills/<name>/SKILL.md` into a `Skill`, once at startup
(Phase 8, docs/contracts.md § 12). This is what makes "an agent sees only a name and a one-line
description until it asks" possible: `load_skills` never keeps anything from the model except that
short index (`skills_index`) — the full markdown body only leaves this module through the `load_skill`
tool (tools/skills.py), and only for the one skill a model actually names.

Front matter is parsed the same way tests/test_eval_sets.py already checks it: split on lines, find
the closing "---", then `yaml.safe_load` everything in between.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

from artlab.config import SKILLS_DIR


@dataclass(frozen=True)
class Skill:
    """One loaded skill.

    name         the folder name — also what `load_skill(name)` and the skills index refer to it by
    description  one line, shown to every agent allowed to use `load_skill` (never the full body)
    body         the markdown after the front matter; only ever reaches a model via `load_skill`
    """

    name: str
    description: str
    body: str


def load_skills(dir: Path = SKILLS_DIR) -> dict[str, Skill]:
    """Read every `<dir>/<folder>/SKILL.md` into a `Skill`, keyed by name.

    Steps, for each subfolder of `dir` (skipped, with a note printed instead of a crash, if any step
    fails — one bad skill folder shouldn't take the rest of the app down):
      1. Read SKILL.md; skip a folder that doesn't have one.
      2. Split off the YAML front matter between the opening and closing "---" lines; skip a file with
         no front matter, or none properly closed.
      3. Parse the front matter and check its own `name` field matches the folder it's in — a mismatch
         usually means a copy-pasted or renamed folder nobody updated the front matter for, and loading
         it under the wrong key would make `load_skill` findable by a name that isn't the folder's.

    Example: `load_skills()` on this repo's `backend/skills/` returns a 3-entry dict, keyed
    "hook-formulas", "retention-analysis", "style-guide".
    """
    skills: dict[str, Skill] = {}
    for folder in sorted(p for p in dir.iterdir() if p.is_dir()):
        path = folder / "SKILL.md"
        if not path.exists():
            continue

        text = path.read_text()
        if not text.startswith("---"):
            print(f"skills/loader: {path} has no YAML front matter, skipping")
            continue
        lines = text.split("\n")
        closing_idx = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
        if closing_idx is None:
            print(f"skills/loader: {path} front matter not closed with ---, skipping")
            continue

        front_matter = yaml.safe_load("\n".join(lines[1:closing_idx])) or {}
        name = front_matter.get("name")
        if name != folder.name:
            print(f"skills/loader: {path} front matter name {name!r} != folder {folder.name!r}, skipping")
            continue

        body = "\n".join(lines[closing_idx + 1:]).strip()
        skills[name] = Skill(name=name, description=front_matter.get("description", ""), body=body)
    return skills


def skills_index(skills: dict[str, Skill]) -> str:
    """The block `agents/tool_loop.py` appends to a system prompt for an agent allowed to use
    `load_skill` (docs/contracts.md § 12): every skill's name and one-line description, sorted by name
    so the text is the same run to run — never a skill's full body.

    Example, with one skill loaded:

        Skills you can load with load_skill (load one only when the task needs it):
        - hook-formulas: Six proven formulas to hook viewers in the first 5 seconds; each with pattern, example, and rules
    """
    lines = "\n".join(f"- {s.name}: {s.description}" for s in sorted(skills.values(), key=lambda s: s.name))
    return f"Skills you can load with load_skill (load one only when the task needs it):\n{lines}"
