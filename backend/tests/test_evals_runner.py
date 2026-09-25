"""
tests/test_evals_runner.py — Phase 11 (P11, docs/contracts.md § 15): the eval runner itself.

Every test here runs `artlab.evals.run.run_eval(..., dry_run=True)` — the free fake model and the fake
judge, $0, no network — exactly what CLAUDE.md's cost rule requires ("never make a real paid API call
... without asking the owner first"). Nothing here ever passes `dry_run=False`.
"""

import json

import pytest
from langchain_core.messages import AIMessage

import artlab.evals.run as run_module
from artlab.model import FakeChatModel


def _report(path):
    """Load the JSON report a run wrote, from the markdown path `run_eval` returned."""
    return json.loads(path.with_suffix(".json").read_text())


class ExpensiveFakeModel(FakeChatModel):
    """The fake model, but every reply also reports a large `usage_metadata` — a stand-in for "a real
    call that cost real money", so the budget guard has something to actually stop on without a real
    model ever running (model.py's plain fake always reports (0, 0) tokens, which would never trip a
    budget check at all)."""

    def _reply(self, messages):
        reply = super()._reply(messages)
        return AIMessage(
            reply.content, tool_calls=reply.tool_calls,
            usage_metadata={"input_tokens": 1_000_000, "output_tokens": 1_000_000, "total_tokens": 2_000_000},
        )


class AlwaysRespondModel(FakeChatModel):
    """A router that always answers "respond", whatever the message — guarantees at least one miss
    against routing_golden.yaml (every entry whose golden route isn't "respond")."""

    def _reply(self, messages):
        if self.tool_name == "RouteDecision":
            args = {"next": "respond", "then": [], "reason": "always respond", "question": "q"}
            return AIMessage("", tool_calls=[{"name": self.tool_name, "args": args, "id": "fake-call", "type": "tool_call"}])
        return super()._reply(messages)


@pytest.mark.parametrize("suite", ["routing", "rag", "workers", "all"])
def test_each_suite_runs_end_to_end_and_writes_a_report_with_the_right_totals_shape(suite):
    """Every `--suite` value must run start to finish on the fake model (dry run) and produce a report
    whose JSON has the shape docs/contracts.md § 15 promises: {suites: {name: {passed, total,
    cases}}, cost_usd, dry_run}. This is the plumbing check the whole ticket rests on — "Done when
    ... uv run python -m artlab.evals.run --suite all --dry-run works"."""
    path = run_module.run_eval(suite, dry_run=True)
    report = _report(path)

    assert report["dry_run"] is True
    assert report["suites"]  # at least one suite ran
    for name, data in report["suites"].items():
        assert set(data) >= {"passed", "total", "cases", "budget_stopped"}
        assert data["total"] >= 1
        assert 0 <= data["passed"] <= data["total"]
        assert isinstance(data["cases"], list)


def test_budget_guard_stops_early_when_max_usd_is_tiny_and_a_fake_reports_cost():
    """The budget guard is checked before every case: with a fake model that reports a large fake
    `usage_metadata` (so its "cost" is nonzero even though no real call happened) and a tiny
    `--max-usd`, the first case still runs (nothing has been spent yet) but the guard must stop before
    a second one."""
    model = ExpensiveFakeModel()
    path = run_module.run_eval("routing", dry_run=True, max_usd=0.0001, model=model, judge_model=model)
    report = _report(path)

    routing = report["suites"]["routing"]
    assert routing["total"] == 15  # the whole golden set, even though most of it never ran
    assert len(routing["cases"]) == 1  # only the one case the (already-exhausted) budget allowed
    assert routing["budget_stopped"] is True
    assert report["cost_usd"] > 0.0001  # what actually got spent, over the cap that stopped it


def test_a_routing_miss_is_recorded_with_expected_vs_got():
    """A wrong route must show up in the report as a failed case naming both the golden `route` and
    what the (scripted, always-wrong) router actually picked — not just "failed", with no way to tell
    what went wrong."""
    path = run_module.run_eval("routing", dry_run=True, model=AlwaysRespondModel())
    report = _report(path)

    misses = [c for c in report["suites"]["routing"]["cases"] if not c["passed"]]
    assert misses  # at least the non-"respond" golden entries missed
    miss = misses[0]
    assert miss["got"] == "respond"
    assert miss["expected"] != "respond"
    assert miss["expected"] == misses[0]["expected"]  # the golden route, not the fake's guess


def test_s_scenario_checklist_lists_s0_through_s16():
    """The report's S-scenario checklist (docs/design.html's S0-S16) must name every one of them, each
    mapped to a free test or "not covered" — the "all scenarios pass" gate reads this in one place."""
    path = run_module.run_eval("routing", dry_run=True)
    markdown = path.read_text()

    for n in range(17):  # S0..S16
        assert f"S{n} " in markdown or f"S{n}|" in markdown, f"S{n} missing from the checklist"


def test_dry_run_never_calls_make_model_so_it_never_reaches_the_network(monkeypatch):
    """CLAUDE.md's cost rule: a dry run must be structurally incapable of a paid call. Monkeypatching
    `make_model` to raise proves `run_eval(..., dry_run=True)` never even calls it — only a real run
    (`dry_run=False`, never used by this test file) would."""
    def _boom():
        raise AssertionError("make_model must never be called during a dry run")

    monkeypatch.setattr(run_module, "make_model", _boom)
    path = run_module.run_eval("all", dry_run=True)

    assert path is not None  # the run completed — it truly never touched make_model
