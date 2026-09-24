"""
Test the golden eval sets and skill files to ensure they maintain consistent structure and quality.

These tests protect the integrity of evaluation data by verifying:
- Routing golden set has correct shape and coverage of all routes
- Worker golden sets have well-formed inputs and check criteria
- Skill files are properly formatted with valid YAML frontmatter and substantial body content

This matters because evals must be reliable: a malformed golden set would invalidate Phase 11 paid
evaluations. A skill file with missing frontmatter would break agent loading. These tests catch
breakage early during development.
"""

import yaml
from pathlib import Path

from artlab.config import REPO_ROOT


# ===== Routing Golden Set Tests =====

def test_routing_golden_has_exactly_15_entries():
    """Routing golden set must have exactly 15 entries for balanced coverage across all routes."""
    routing_path = REPO_ROOT / "evals" / "routing_golden.yaml"
    entries = yaml.safe_load(routing_path.read_text())
    assert len(entries) == 15, f"Expected 15 routing entries, got {len(entries)}"


def test_routing_golden_every_entry_has_required_fields():
    """Every routing entry must have non-empty message, route, and why fields."""
    routing_path = REPO_ROOT / "evals" / "routing_golden.yaml"
    entries = yaml.safe_load(routing_path.read_text())

    for i, entry in enumerate(entries):
        assert "message" in entry, f"Entry {i} missing 'message'"
        assert "route" in entry, f"Entry {i} missing 'route'"
        assert "why" in entry, f"Entry {i} missing 'why'"
        assert entry["message"].strip(), f"Entry {i} has empty message"
        assert entry["route"].strip(), f"Entry {i} has empty route"
        assert entry["why"].strip(), f"Entry {i} has empty why"


def test_routing_golden_uses_only_allowed_routes():
    """Every route must be one of the five allowed routes."""
    allowed_routes = {"respond", "rag_agent", "youtube_researcher", "content_ideator", "english_coach"}
    routing_path = REPO_ROOT / "evals" / "routing_golden.yaml"
    entries = yaml.safe_load(routing_path.read_text())

    for i, entry in enumerate(entries):
        route = entry["route"].strip()
        assert route in allowed_routes, f"Entry {i} has unknown route '{route}'"


def test_routing_golden_each_route_appears_exactly_3_times():
    """Each of the 5 routes must appear in exactly 3 entries (5 routes × 3 = 15 entries)."""
    routing_path = REPO_ROOT / "evals" / "routing_golden.yaml"
    entries = yaml.safe_load(routing_path.read_text())

    route_counts = {}
    for entry in entries:
        route = entry["route"].strip()
        route_counts[route] = route_counts.get(route, 0) + 1

    expected_routes = {"respond", "rag_agent", "youtube_researcher", "content_ideator", "english_coach"}
    for route in expected_routes:
        assert route_counts.get(route) == 3, f"Route '{route}' appears {route_counts.get(route, 0)} times, expected 3"


# ===== Worker Golden Set Tests =====

def test_youtube_researcher_golden_has_at_least_5_entries():
    """Worker golden set must have at least 5 test cases to provide meaningful eval coverage."""
    path = REPO_ROOT / "evals" / "youtube_researcher_golden.yaml"
    entries = yaml.safe_load(path.read_text())
    assert len(entries) >= 5, f"Expected at least 5 youtube_researcher entries, got {len(entries)}"


def test_youtube_researcher_golden_every_entry_has_input_and_checks():
    """Every worker entry must have a non-empty input and at least one non-empty checks line."""
    path = REPO_ROOT / "evals" / "youtube_researcher_golden.yaml"
    entries = yaml.safe_load(path.read_text())

    for i, entry in enumerate(entries):
        assert "input" in entry, f"youtube_researcher entry {i} missing 'input'"
        assert "checks" in entry, f"youtube_researcher entry {i} missing 'checks'"
        assert entry["input"].strip(), f"youtube_researcher entry {i} has empty input"
        assert entry["checks"], f"youtube_researcher entry {i} has empty checks list"
        assert any(c.strip() for c in entry["checks"]), f"youtube_researcher entry {i} has only empty checks"


def test_content_ideator_golden_has_at_least_5_entries():
    """Worker golden set must have at least 5 test cases to provide meaningful eval coverage."""
    path = REPO_ROOT / "evals" / "content_ideator_golden.yaml"
    entries = yaml.safe_load(path.read_text())
    assert len(entries) >= 5, f"Expected at least 5 content_ideator entries, got {len(entries)}"


def test_content_ideator_golden_every_entry_has_input_and_checks():
    """Every worker entry must have a non-empty input and at least one non-empty checks line."""
    path = REPO_ROOT / "evals" / "content_ideator_golden.yaml"
    entries = yaml.safe_load(path.read_text())

    for i, entry in enumerate(entries):
        assert "input" in entry, f"content_ideator entry {i} missing 'input'"
        assert "checks" in entry, f"content_ideator entry {i} missing 'checks'"
        assert entry["input"].strip(), f"content_ideator entry {i} has empty input"
        assert entry["checks"], f"content_ideator entry {i} has empty checks list"
        assert any(c.strip() for c in entry["checks"]), f"content_ideator entry {i} has only empty checks"


def test_english_coach_golden_has_at_least_5_entries():
    """Worker golden set must have at least 5 test cases to provide meaningful eval coverage."""
    path = REPO_ROOT / "evals" / "english_coach_golden.yaml"
    entries = yaml.safe_load(path.read_text())
    assert len(entries) >= 5, f"Expected at least 5 english_coach entries, got {len(entries)}"


def test_english_coach_golden_every_entry_has_input_and_checks():
    """Every worker entry must have a non-empty input and at least one non-empty checks line."""
    path = REPO_ROOT / "evals" / "english_coach_golden.yaml"
    entries = yaml.safe_load(path.read_text())

    for i, entry in enumerate(entries):
        assert "input" in entry, f"english_coach entry {i} missing 'input'"
        assert "checks" in entry, f"english_coach entry {i} missing 'checks'"
        assert entry["input"].strip(), f"english_coach entry {i} has empty input"
        assert entry["checks"], f"english_coach entry {i} has empty checks list"
        assert any(c.strip() for c in entry["checks"]), f"english_coach entry {i} has only empty checks"


# ===== Skill File Tests =====

def _load_skill_frontmatter_and_body(path: Path) -> tuple[dict, list[str]]:
    """
    Parse a skill file into YAML frontmatter and markdown body.
    Returns (frontmatter_dict, body_lines).
    """
    text = path.read_text()

    if not text.startswith("---"):
        raise ValueError(f"{path} does not start with YAML frontmatter delimiter")

    # Find the closing --- delimiter
    lines = text.split("\n")
    closing_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            closing_idx = i
            break

    if closing_idx is None:
        raise ValueError(f"{path} frontmatter not closed with ---")

    # Extract frontmatter and body
    frontmatter_text = "\n".join(lines[1:closing_idx])
    body_lines = lines[closing_idx + 1:]

    frontmatter = yaml.safe_load(frontmatter_text)
    return frontmatter, body_lines


def test_retention_analysis_skill_file_exists():
    """Skill file must exist at the expected path."""
    path = REPO_ROOT / "backend" / "skills" / "retention-analysis" / "SKILL.md"
    assert path.exists(), f"Skill file not found: {path}"


def test_retention_analysis_skill_frontmatter_valid():
    """Skill frontmatter must be valid YAML with name and description fields."""
    path = REPO_ROOT / "backend" / "skills" / "retention-analysis" / "SKILL.md"
    frontmatter, _ = _load_skill_frontmatter_and_body(path)

    assert "name" in frontmatter, "Frontmatter missing 'name' field"
    assert "description" in frontmatter, "Frontmatter missing 'description' field"
    assert frontmatter["name"] == "retention-analysis", f"Name field should be 'retention-analysis', got '{frontmatter['name']}'"


def test_retention_analysis_skill_description_valid():
    """Description must be single line and at most 150 characters."""
    path = REPO_ROOT / "backend" / "skills" / "retention-analysis" / "SKILL.md"
    frontmatter, _ = _load_skill_frontmatter_and_body(path)

    desc = frontmatter["description"]
    assert "\n" not in desc, "Description must be a single line"
    assert len(desc) <= 150, f"Description is {len(desc)} chars, max 150"


def test_retention_analysis_skill_body_substantial():
    """Skill body must have at least 30 lines of content after frontmatter."""
    path = REPO_ROOT / "backend" / "skills" / "retention-analysis" / "SKILL.md"
    _, body_lines = _load_skill_frontmatter_and_body(path)

    # Count non-empty lines in body
    non_empty = [line for line in body_lines if line.strip()]
    assert len(non_empty) >= 30, f"Skill body has {len(non_empty)} non-empty lines, expected at least 30"


def test_hook_formulas_skill_file_exists():
    """Skill file must exist at the expected path."""
    path = REPO_ROOT / "backend" / "skills" / "hook-formulas" / "SKILL.md"
    assert path.exists(), f"Skill file not found: {path}"


def test_hook_formulas_skill_frontmatter_valid():
    """Skill frontmatter must be valid YAML with name and description fields."""
    path = REPO_ROOT / "backend" / "skills" / "hook-formulas" / "SKILL.md"
    frontmatter, _ = _load_skill_frontmatter_and_body(path)

    assert "name" in frontmatter, "Frontmatter missing 'name' field"
    assert "description" in frontmatter, "Frontmatter missing 'description' field"
    assert frontmatter["name"] == "hook-formulas", f"Name field should be 'hook-formulas', got '{frontmatter['name']}'"


def test_hook_formulas_skill_description_valid():
    """Description must be single line and at most 150 characters."""
    path = REPO_ROOT / "backend" / "skills" / "hook-formulas" / "SKILL.md"
    frontmatter, _ = _load_skill_frontmatter_and_body(path)

    desc = frontmatter["description"]
    assert "\n" not in desc, "Description must be a single line"
    assert len(desc) <= 150, f"Description is {len(desc)} chars, max 150"


def test_hook_formulas_skill_body_substantial():
    """Skill body must have at least 30 lines of content after frontmatter."""
    path = REPO_ROOT / "backend" / "skills" / "hook-formulas" / "SKILL.md"
    _, body_lines = _load_skill_frontmatter_and_body(path)

    # Count non-empty lines in body
    non_empty = [line for line in body_lines if line.strip()]
    assert len(non_empty) >= 30, f"Skill body has {len(non_empty)} non-empty lines, expected at least 30"


def test_style_guide_skill_file_exists():
    """Skill file must exist at the expected path."""
    path = REPO_ROOT / "backend" / "skills" / "style-guide" / "SKILL.md"
    assert path.exists(), f"Skill file not found: {path}"


def test_style_guide_skill_frontmatter_valid():
    """Skill frontmatter must be valid YAML with name and description fields."""
    path = REPO_ROOT / "backend" / "skills" / "style-guide" / "SKILL.md"
    frontmatter, _ = _load_skill_frontmatter_and_body(path)

    assert "name" in frontmatter, "Frontmatter missing 'name' field"
    assert "description" in frontmatter, "Frontmatter missing 'description' field"
    assert frontmatter["name"] == "style-guide", f"Name field should be 'style-guide', got '{frontmatter['name']}'"


def test_style_guide_skill_description_valid():
    """Description must be single line and at most 150 characters."""
    path = REPO_ROOT / "backend" / "skills" / "style-guide" / "SKILL.md"
    frontmatter, _ = _load_skill_frontmatter_and_body(path)

    desc = frontmatter["description"]
    assert "\n" not in desc, "Description must be a single line"
    assert len(desc) <= 150, f"Description is {len(desc)} chars, max 150"


def test_style_guide_skill_body_substantial():
    """Skill body must have at least 30 lines of content after frontmatter."""
    path = REPO_ROOT / "backend" / "skills" / "style-guide" / "SKILL.md"
    _, body_lines = _load_skill_frontmatter_and_body(path)

    # Count non-empty lines in body
    non_empty = [line for line in body_lines if line.strip()]
    assert len(non_empty) >= 30, f"Skill body has {len(non_empty)} non-empty lines, expected at least 30"
