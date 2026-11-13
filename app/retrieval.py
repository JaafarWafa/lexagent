"""Hybrid retrieval: BM25 + multilingual embeddings, fused with Reciprocal Rank Fusion.

- BM25 is strong on exact legal terms and numbers ("dépôt de garantie", "IRL", "750").
- Embeddings handle paraphrases and English questions about a French contract.
- RRF merges the two rankings without having to calibrate their scores.

Embeddings are optional: if `EMBEDDING_MODEL` is empty or fastembed cannot load,
the retriever silently falls back to BM25 only.
"""
import logging
import re
import unicodedata
from functools import lru_cache
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Plus  # BM25Plus keeps scores positive on small contracts

from app.config import settings
from app.schemas import Chunk

log = logging.getLogger(__name__)


def _tokens(text: str) -> list[str]:
    """Lowercase and strip accents, so "depot" (typed fast, or from OCR) matches "dépôt"."""
    folded = unicodedata.normalize("NFKD", text.lower())
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return re.findall(r"\w+", folded)


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return one L2-normalised vector per text, shape (n, dim)."""


class FastEmbedder:
    def __init__(self, model_name: str):
        from fastembed import TextEmbedding  # imported lazily: heavy, and optional

        self.name = model_name
        self.model = TextEmbedding(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        v = np.array(list(self.model.embed(texts)), dtype=np.float32)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


@lru_cache(maxsize=1)
def default_embedder() -> Embedder | None:
    if not settings.embedding_model:
        return None
    try:
        return FastEmbedder(settings.embedding_model)
    except Exception as e:  # missing package, no network for the first download...
        log.warning("Embeddings disabled, falling back to BM25 only: %s", e)
        return None


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda i: scores[i], reverse=True)


class Retriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None):
        self.chunks = chunks
        self.bm25 = BM25Plus([_tokens(c.text) for c in chunks]) if chunks else None
        self.embedder = embedder if chunks else None
        self.vectors = embedder.embed([c.text for c in chunks]) if self.embedder else None

    @property
    def mode(self) -> str:
        return "hybrid" if self.vectors is not None else "bm25"

    def _bm25_ranking(self, query: str) -> list[int]:
        scores = self.bm25.get_scores(_tokens(query))
        return [i for i in np.argsort(-scores, kind="stable") if scores[i] > 0]

    def _dense_ranking(self, query: str) -> list[int]:
        sims = self.vectors @ self.embedder.embed([query])[0]
        return list(np.argsort(-sims, kind="stable"))

    def search(self, query: str, k: int | None = None, keywords: str = "") -> list[Chunk]:
        """`keywords` (e.g. a French rewrite of an English question) only feeds BM25:
        the embedding model is multilingual and works best on the natural question."""
        k = k or settings.top_k
        if not self.bm25:
            return []
        bm25 = self._bm25_ranking(f"{query} {keywords}".strip())
        if self.vectors is None:
            order = bm25
        else:
            order = rrf([bm25, self._dense_ranking(query)], k=settings.rrf_k)
        return [self.chunks[i] for i in order[:k]]
