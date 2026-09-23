"""
Checks the golden question set itself (evals/rag_golden.yaml), before anything uses it.

A golden set is only useful if it's right: if an "expected fact" isn't actually in its source file,
the retrieval eval would be measuring against a wrong answer. These tests keep the set honest as the
knowledge files change.
"""

import yaml

from artlab.config import GOLDEN_PATH, REPO_ROOT


def load_golden() -> list[dict]:
    """The golden entries: [{question, source, fact}]; `source` is None for out-of-corpus questions."""
    return yaml.safe_load(GOLDEN_PATH.read_text())


def test_every_expected_fact_is_in_its_source_file():
    """Each answerable question's fact appears word for word (ignoring case) in the file it points to."""
    for entry in load_golden():
        if entry["source"] is None:
            continue
        text = (REPO_ROOT / entry["source"]).read_text()
        assert entry["fact"].lower() in text.lower(), f"{entry['fact']!r} not found in {entry['source']}"


def test_golden_set_has_answerable_and_unanswerable_questions():
    """10 questions the knowledge base can answer, and 2 it can't (to test "not found")."""
    entries = load_golden()
    assert sum(e["source"] is not None for e in entries) == 10
    assert sum(e["source"] is None for e in entries) == 2
