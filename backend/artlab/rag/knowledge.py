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

from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from artlab.config import CHROMA_DIR
from artlab.rag.embeddings import LocalEmbeddings

COLLECTION = "knowledge"


class KnowledgeBase:
    """The knowledge base's storage: a thin wrapper around one LangChain `Chroma` collection.

    Args:
        persist_dir: folder Chroma saves to. The app uses data/chroma/; tests use a temp folder.
        embeddings:  how text becomes numbers. The app uses LocalEmbeddings; tests pass a fast fake.
    """

    def __init__(self, persist_dir: Path = CHROMA_DIR, embeddings: Embeddings | None = None):
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
