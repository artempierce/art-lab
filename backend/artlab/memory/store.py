"""
memory/store.py — long-term memory: one Chroma collection of durable facts about the owner
(Phase 7, M1). "Long-term" as opposed to `messages`: a chat's history is only ever loaded for its own
thread, but a fact saved here is loaded into state["memory"] at the start of every future chat, on any
thread (agents/recall.py).

Who uses this file:
    agents/remember.py   writes: saves at most 3 facts extracted from your own message, each turn
    agents/recall.py     reads: loads your facts into state["memory"] at the start of every turn

Storage copies rag/knowledge.py's setup (a `langchain_chroma.Chroma` collection, `LocalEmbeddings`) but
uses its own collection, "memory". By default it opens the *same folder* the knowledge base uses
(data/chroma/) — Chroma keeps collections inside one folder completely separate, the way two tables
share one database.

Docs/contracts.md § 11 is the spec this file implements.
"""

import re
import time
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from artlab.config import CHROMA_DIR
from artlab.rag.embeddings import LocalEmbeddings

COLLECTION = "memory"

# Single user today. Every read and write filters on this id (never a constructor argument), so this
# stays the one place multi-user support would change later (docs/contracts.md § 11).
OWNER_ID = "owner"

# recall() returns every fact, unsearched, up to this many. Past it, a similarity search picks the
# ones most relevant to what you just said instead of handing the model everything you've ever said.
MAX_RECALL = 12

# Keys and values are capped so one runaway "fact" can't blow out a prompt.
MAX_KEY_CHARS = 40
MAX_VALUE_CHARS = 200


def normalise_key(key: str) -> str:
    """Lowercase snake_case, capped at MAX_KEY_CHARS: how every fact's key is stored, however the
    model (real or fake) spelled it.

    Example: normalise_key("My Niche") == "my_niche"
    """
    key = re.sub(r"\s+", "_", key.strip().lower())
    key = re.sub(r"[^a-z0-9_]", "", key)
    return key[:MAX_KEY_CHARS]


class MemoryStore:
    """The owner's long-term memory: one fact per document, upserted by key.

    Args:
        persist_dir: folder Chroma saves to. The app uses data/chroma/ (the same folder the
                     knowledge base uses); tests use a temp folder.
        embeddings:  how text becomes numbers, for recall()'s similarity search once there are more
                     than MAX_RECALL facts. The app uses LocalEmbeddings; tests pass a fast fake.
    """

    def __init__(self, persist_dir: Path = CHROMA_DIR, embeddings: Embeddings | None = None):
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.store = Chroma(
            collection_name=COLLECTION,
            embedding_function=embeddings or LocalEmbeddings(),
            persist_directory=str(persist_dir),
            collection_metadata={"hnsw:space": "cosine"},
            client_settings=chromadb.config.Settings(
                anonymized_telemetry=False, is_persistent=True, persist_directory=str(persist_dir)
            ),
        )

    def save(self, key: str, value: str, thread_id: str) -> None:
        """Save one fact, upserting by key: a second "my niche is X" replaces the first, it doesn't
        add a duplicate.

        The document text is "<key>: <value>" (so a plain similarity search over the collection makes
        sense); the id is f"{OWNER_ID}:{key}" — Chroma's `add_texts` upserts by id, so saving the same
        key twice just replaces the one document, the same "delete-then-add" idea rag/knowledge.py's
        `replace_source` uses, but at the level of one fact instead of one whole file.
        """
        key = normalise_key(key)
        value = value.strip()[:MAX_VALUE_CHARS]
        self.store.add_texts(
            texts=[f"{key}: {value}"],
            metadatas=[{
                "key": key, "value": value, "owner": OWNER_ID,
                "created_at": time.time(), "thread_id": thread_id,
            }],
            ids=[f"{OWNER_ID}:{key}"],
        )

    def all(self) -> list[dict]:
        """Every fact for the owner, newest first: [{"key", "value", "owner", "created_at",
        "thread_id"}, ...]. Filtered to OWNER_ID, like every read here."""
        got = self.store.get(where={"owner": OWNER_ID}, include=["metadatas"])
        facts = list(got["metadatas"])
        facts.sort(key=lambda m: m["created_at"], reverse=True)
        return facts

    def recall(self, query: str, k: int = MAX_RECALL) -> list[str]:
        """The facts most relevant to `query`, as "<key>: <value>" strings.

        With k facts or fewer stored, there's nothing to rank — return all of them (newest first).
        Past that, a similarity search against `query` (still filtered to OWNER_ID) keeps what's
        handed to the model small and relevant instead of dumping everything you've ever said.
        """
        facts = self.all()
        if len(facts) <= k:
            chosen = facts
        else:
            hits = self.store.similarity_search(query, k=k, filter={"owner": OWNER_ID})
            chosen = [{"key": h.metadata["key"], "value": h.metadata["value"]} for h in hits]
        return [f"{f['key']}: {f['value']}" for f in chosen]

    def delete(self, key: str) -> None:
        """Remove one fact by key. Used later by the Phase 12 Memory page."""
        self.store.delete(ids=[f"{OWNER_ID}:{normalise_key(key)}"])
