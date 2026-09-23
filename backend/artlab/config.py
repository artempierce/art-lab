"""
config.py — where things live on disk. One place, so every module agrees on the same paths.

Everything under data/ is git-ignored and can be deleted to start fresh: it's rebuilt on the next run
(chats start empty; run the ingest command to rebuild the knowledge base).
"""

from pathlib import Path

# This file is backend/artlab/config.py, so the repo root is two folders up.
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
DB_PATH = DATA_DIR / "artlab.db"  # chat history (SQLite checkpointer)
CHROMA_DIR = DATA_DIR / "chroma"  # the knowledge base (Chroma vector database)
MODELS_DIR = DATA_DIR / "models"  # the downloaded local embedding model (~70 MB, fetched once)

# What `python -m artlab.rag.ingest` reads when you give it no arguments: the sample policies in
# knowledge/, plus two of our own project docs.
KNOWLEDGE_DIR = REPO_ROOT / "knowledge"
DEFAULT_SOURCES = [KNOWLEDGE_DIR, REPO_ROOT / "docs" / "architecture.md", REPO_ROOT / "docs" / "requirements.md"]

# Golden questions for the knowledge base: see evals/rag_golden.yaml.
GOLDEN_PATH = REPO_ROOT / "evals" / "rag_golden.yaml"
