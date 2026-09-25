"""
artlab/evals/ — the Phase 11 eval runner package (docs/contracts.md § 15).

run.py     the CLI (`python -m artlab.evals.run`): routing, RAG and worker golden sets, the budget
           guard, and the report (evals/reports/*.md + *.json).
judge.py   the LLM judge (`Verdicts`) that grades an answer against golden `checks`.
"""
