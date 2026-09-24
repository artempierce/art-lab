"""
skills/ — skills loaded on demand (Phase 8, docs/contracts.md § 12): the "know-how, not code" folder.

Where this sits in the message flow: `loader.py` reads every `backend/skills/<name>/SKILL.md` once at
startup into a `Skill(name, description, body)`. An agent's system prompt only ever gets the names and
descriptions (`skills_index`, used by `agents/tool_loop.py`); the full body reaches a model only through
the `load_skill` tool (`tools/skills.py`), and only when the model actually asks for it by name.
"""
