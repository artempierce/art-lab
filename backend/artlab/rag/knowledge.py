"""
rag/knowledge.py — the knowledge base: one Chroma collection of embedded chunks.

Chroma in one paragraph: a vector database. For every chunk it stores the text, the chunk's embedding,
and metadata (which file, which heading, whether it was flagged…). Given a question's embedding, it
returns the stored chunks whose embeddings are closest. Here Chroma runs inside our own Python
process and saves to data/chroma/ — no server, no Docker.

Who uses this file:
    rag/ingest.py   writes: adds, replaces and deletes a source's chunks
    rag_agent       reads: searches, through the `search_knowledge` tool

Each chunk's metadata:
    source         repo-relative file path, or the URL          "knowledge/brand-voice.md"
    title          the document's title                         "Art Lab Studio — Brand Voice Guide"
    heading        the section the chunk sits under             "Titles"
    kind           "file" or "web"
    chunk_index    position within its source                   0, 1, 2, …
    content_hash   fingerprint of the whole source's text        used to skip unchanged sources
    flagged        injection rule that matched, or ""            search skips flagged chunks
    ingested_at    when it was stored, ISO-8601 UTC              "2026-09-23T14:05:00+00:00"
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from artlab.config import CHROMA_DIR
from artlab.rag.embeddings import LocalEmbeddings
from artlab.tools.untrusted import Piece

COLLECTION = "knowledge"

# How many chunks a search returns at most. 4 chunks (~3,200 characters) is enough to answer a
# policy question and small enough to keep the answer prompt cheap.
TOP_K = 4

# Minimum relevance score (0–1) for a chunk to be returned at all. Tuned on the golden questions
# (tests/test_retrieval_eval.py): every answerable question's right chunk scored 0.69–0.82, while
# clearly unrelated questions ("weather on Mars") score ~0.4.
#
# What a threshold CAN'T do: tell whether the chunks actually *answer* the question. "How much does
# the studio pay towards health insurance?" (not in the knowledge base) still matched the expenses
# policy at 0.74 — higher than some real answers. Similar isn't the same as answering. So this cut-off
# only removes junk; deciding "these passages don't answer it" is the rag_agent's job, in its prompt.
MIN_SCORE = 0.6


@dataclass
class Hit:
    """One search result. `n` is its citation number ([1], [2], …) in the answer."""

    n: int
    source: str
    heading: str
    text: str
    score: float


@dataclass
class SearchResult:
    """What a search returns: the hits, plus how many matching chunks were skipped because they were flagged."""

    query: str
    hits: list[Hit]
    flagged_skipped: int

    def pieces(self) -> list[Piece]:
        """The hits as pieces for the tool gateway, which wraps each one as untrusted. The label is the
        citation line, so rag_agent's prompt ends up looking like:

            [1] knowledge/brand-voice.md › Titles
            <untrusted_retrieval source="knowledge/brand-voice.md">…</untrusted_retrieval>

        This class doesn't wrap anything itself: wrapping is the gateway's job, for every tool.
        """
        return [Piece(source=h.source, text=h.text, label=f"[{h.n}] {h.source}" + (f" › {h.heading}" if h.heading else "")) for h in self.hits]

    def citations(self) -> list[dict]:
        """The hits as plain dicts, for the API and the UI's Sources list."""
        return [asdict(h) for h in self.hits]


class KnowledgeBase:
    """The knowledge base's storage: a thin wrapper around one LangChain `Chroma` collection.

    Args:
        persist_dir: folder Chroma saves to. The app uses data/chroma/; tests use a temp folder.
        embeddings:  how text becomes numbers. The app uses LocalEmbeddings; tests pass a fast fake.
        min_score:   relevance cut-off for search (see MIN_SCORE). Tests with fake embeddings, whose
                     scores mean nothing, pass -1 to keep every hit.
    """

    def __init__(self, persist_dir: Path = CHROMA_DIR, embeddings: Embeddings | None = None, min_score: float = MIN_SCORE):
        self.min_score = min_score
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.store = Chroma(
            collection_name=COLLECTION,
            embedding_function=embeddings or LocalEmbeddings(),
            persist_directory=str(persist_dir),
            # "cosine" compares the *direction* of two embeddings and ignores their length:
            # the standard way to compare text embeddings. Scores come back between 0 and 1.
            collection_metadata={"hnsw:space": "cosine"},
            client_settings=chromadb.config.Settings(
                anonymized_telemetry=False, is_persistent=True, persist_directory=str(persist_dir)
            ),
        )

    def sources(self) -> dict[str, dict]:
        """Every source in the knowledge base, with what ingest needs to know about it.

        Returns {source: {"content_hash": ..., "kind": "file" | "web", "chunks": n}}.
        Built from the chunks' metadata, since Chroma stores chunks, not files.
        """
        found: dict[str, dict] = {}
        for meta in self.store.get(include=["metadatas"])["metadatas"]:
            entry = found.setdefault(meta["source"], {"content_hash": meta["content_hash"], "kind": meta["kind"], "chunks": 0})
            entry["chunks"] += 1
        return found

    def delete_source(self, source: str) -> None:
        """Remove every chunk that came from `source`."""
        ids = self.store.get(where={"source": source})["ids"]
        if ids:
            self.store.delete(ids=ids)

    def replace_source(self, source: str, chunks: list[tuple[str, str, dict]]) -> None:
        """Swap all of `source`'s chunks for new ones: [(chunk_id, text, metadata), ...].

        Delete-then-add (rather than updating in place) means a file that shrank from 5 chunks to 3
        doesn't leave 2 stale chunks behind. `add_texts` embeds each text before storing it.
        """
        self.delete_source(source)
        if chunks:
            ids, texts, metadatas = zip(*chunks)
            self.store.add_texts(texts=list(texts), metadatas=list(metadatas), ids=list(ids))

    def search(self, query: str, k: int = TOP_K) -> SearchResult:
        """Find the chunks most relevant to `query`. This is the `search_knowledge` tool.

        Steps:
          1. Embed the query and ask Chroma for the 2×k nearest chunks, each with a relevance score
             (1 = same meaning, 0 = unrelated). Asking for extra leaves room after filtering.
          2. Drop chunks below min_score: close-ish isn't good enough to answer from.
          3. Drop flagged chunks (they matched an injection rule at ingest), counting them so the
             trace can show "1 flagged chunk skipped".
          4. Keep the best k and number them 1..k for citations.
        """
        candidates = self.store.similarity_search_with_relevance_scores(query, k=k * 2)
        hits: list[Hit] = []
        flagged = 0
        for doc, score in candidates:
            if score < self.min_score:
                continue
            if doc.metadata.get("flagged"):
                flagged += 1
                continue
            if len(hits) < k:
                meta = doc.metadata
                hits.append(Hit(len(hits) + 1, meta["source"], meta.get("heading", ""), doc.page_content, round(score, 3)))
        return SearchResult(query, hits, flagged)
