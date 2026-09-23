"""
rag/embeddings.py — turns text into embeddings, on your machine, for free.

What an embedding is: a list of numbers (here 384) that captures what a piece of text *means*. Texts
with similar meaning get similar lists, so "how do refunds work?" lands close to "Returns policy:
refunds within 14 days" even though they share almost no words. Searching the knowledge base means:
embed the question, then find the stored chunks whose embeddings are closest to it.

The model: BAAI/bge-small-en-v1.5, run by the FastEmbed library on your CPU (ONNX Runtime; no PyTorch,
no API, no cost). It's downloaded once (~70 MB) into data/models/ and works offline after that.

This class implements LangChain's `Embeddings` interface: just `embed_documents` and `embed_query`.
Chroma only talks to that interface, so switching to a paid embedding API later (e.g. Voyage) means
replacing this one class and re-running ingest.
"""

from langchain_core.embeddings import Embeddings

from artlab.config import MODELS_DIR

MODEL_NAME = "BAAI/bge-small-en-v1.5"


class LocalEmbeddings(Embeddings):
    """LangChain-compatible embeddings computed locally with FastEmbed.

    The model is loaded lazily, on the first embed call, not when this object is created. That keeps
    server start-up fast and lets tests that never embed anything skip the download entirely.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._model = None  # loaded by _load() on first use

    def _load(self):
        """Load the model on first use; downloads it into data/models/ if it isn't there yet."""
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(self.model_name, cache_dir=str(MODELS_DIR))
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed chunks that are being stored. Returns one list of 384 numbers per text.

        `passage_embed` is the model's mode for documents; questions use `query_embed` (below).
        """
        return [vector.tolist() for vector in self._load().passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        """Embed a search question.

        bge models were trained with questions and passages embedded slightly differently:
        `query_embed` adds the short instruction prefix the model expects on questions, which makes
        question-to-passage matching noticeably better than embedding both the same way.
        """
        return next(iter(self._load().query_embed([text]))).tolist()
