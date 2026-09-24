"""
tools/skills.py — load_skill, the tool that pulls one whole skill's text into an agent's own turn
(Phase 8, docs/contracts.md § 12). skills/loader.py reads every skill once at startup; this file only
ever hands one back, and only when a model asks for it by name — that's the difference between an
agent and a skill (design book: "just know-how → skill").

A factory, not a plain function, the same reason tools/ideas.py's make_save_ideas is one: the tool
gateway (tools/registry.py) only ever sees the returned function, so it can't tell "load_skill built
from the repo's real skills" apart from "load_skill built from a test's smaller dict".
"""

from typing import Callable

from artlab.skills.loader import Skill


def make_load_skill(skills: dict[str, Skill]) -> Callable[[str], str]:
    """Build a `load_skill(name: str) -> str` tool returning `skills[name]`'s body, or a message
    naming every skill that IS loaded when `name` doesn't match one.

    Example: with hook-formulas, retention-analysis and style-guide loaded,
             load_skill("hook-formulas") returns that file's markdown body;
             load_skill("nope") returns "No skill named 'nope'. Available: hook-formulas,
             retention-analysis, style-guide."
    """

    def load_skill(name: str) -> str:
        skill = skills.get(name)
        if skill is None:
            available = ", ".join(sorted(skills))
            return f"No skill named '{name}'. Available: {available}."
        return skill.body

    return load_skill
