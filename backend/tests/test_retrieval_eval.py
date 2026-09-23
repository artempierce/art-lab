"""
Retrieval eval (RAG-6): does search find the right file for the golden questions?

Real everything, still free: the real knowledge files, the real local embedding model, a real Chroma
database (in a temp folder, built once for this module). No LLM is involved.

Metrics, over the 10 answerable golden questions:
    hit@4  share of questions whose expected file is among the top 4 results      target ≥ 90%
    MRR    mean reciprocal rank: 1 if the right file is 1st, 1/2 if 2nd, …        target ≥ 0.7

Run with -s to see the per-question report:  uv run pytest tests/test_retrieval_eval.py -s
"""

import pytest
import yaml

from artlab.config import DEFAULT_SOURCES, GOLDEN_PATH
from artlab.rag.ingest import ingest
from artlab.rag.knowledge import KnowledgeBase


@pytest.fixture(scope="module")
def kb(tmp_path_factory):
    """The real knowledge base, built once from the default sources with the real local model."""
    knowledge = KnowledgeBase(tmp_path_factory.mktemp("chroma"))
    ingest(DEFAULT_SOURCES, knowledge)
    return knowledge


def test_search_finds_the_right_file_for_golden_questions(kb):
    """hit@4 ≥ 90% and MRR ≥ 0.7 on the answerable golden questions."""
    answerable = [e for e in yaml.safe_load(GOLDEN_PATH.read_text()) if e["source"]]
    ranks = []
    for entry in answerable:
        sources = [hit.source for hit in kb.search(entry["question"]).hits]
        rank = sources.index(entry["source"]) + 1 if entry["source"] in sources else None
        ranks.append(rank)
        print(f"  rank {rank or '-':>2}  {entry['question']}")

    hit_at_4 = sum(r is not None for r in ranks) / len(ranks)
    mrr = sum(1 / r for r in ranks if r) / len(ranks)
    print(f"  hit@4 = {hit_at_4:.0%}   MRR = {mrr:.2f}")
    assert hit_at_4 >= 0.9 and mrr >= 0.7


def test_unrelated_question_returns_nothing(kb):
    """True junk falls under the relevance cut-off, so rag_agent can answer "not found" without an LLM call."""
    assert kb.search("What's the weather like on Mars today?").hits == []


def test_out_of_corpus_questions_can_still_match_something(kb):
    """Pins a real limit (see MIN_SCORE in rag/knowledge.py): a question the knowledge base can't answer
    can still find *similar* chunks. Deciding they don't answer it is the rag_agent's prompt's job."""
    assert kb.search("How much does the studio pay towards health insurance?").hits != []


def test_poisoned_note_blatant_part_is_skipped_subtle_part_is_wrapped(kb):
    """Defence in depth on the test document knowledge/_security/poisoned-note.md:
    the blatant injection was flagged at ingest and is skipped; the subtle one no regex catches
    comes back — but wrapped as untrusted, for the agent's prompt to treat as data."""
    result = kb.search("Are sponsorships pre-approved without disclosure?")
    assert result.flagged_skipped >= 1
    assert all("Internal memo" not in hit.heading for hit in result.hits)

    supplier = kb.search("cable tray supplier sizes")
    assert supplier.hits[0].heading == "Supplier details"
    assert '<untrusted_retrieval source="knowledge/_security/poisoned-note.md">' in supplier.as_context()
