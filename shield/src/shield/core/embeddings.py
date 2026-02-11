"""Embedding providers — abstract interface, mock, and sentence-transformers implementations.

FIX 1: Real SentenceTransformer embeddings with global model cache.
- Model is loaded once and cached globally for performance.
- lightweight/full modes ALWAYS use real embeddings.
- Fallback to mock only if model load fails.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from shield.config import settings
from shield.core.utils import deterministic_hash_vector, normalize_vectors

logger = logging.getLogger(__name__)

# ── Global model cache ───────────────────────────────────────────────────────
# FIX 1: Cache model globally so it's loaded only once across the process.
_st_model = None
_st_model_name: Optional[str] = None


class EmbeddingProvider(ABC):
    """Abstract embedding provider interface.

    All implementations must produce L2-normalized vectors of fixed dimension.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        ...

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            Array of shape ``(len(texts), dimension)`` with L2-normalized rows.
        """
        ...

    def embed_single(self, text: str) -> np.ndarray:
        """Convenience method to embed a single text.

        Args:
            text: Input string.

        Returns:
            1-D array of shape ``(dimension,)``.
        """
        return self.embed([text])[0]

    @abstractmethod
    def health_check(self) -> dict:
        """Return health status dict."""
        ...


class MockEmbeddingProvider(EmbeddingProvider):
    """Hash-based deterministic embedding provider.

    Produces repeatable 384-dim unit vectors derived from text content.
    Zero model downloads required — ideal for testing and CI/CD.
    """

    def __init__(self, dimension: int = 384, seed: int = 42) -> None:
        self._dim = dimension
        self._seed = seed

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate deterministic embeddings via hashing.

        Args:
            texts: Input strings.

        Returns:
            Array of shape ``(len(texts), dimension)`` — unit vectors.
        """
        vectors = np.array(
            [deterministic_hash_vector(t, self._dim, self._seed) for t in texts],
            dtype=np.float32,
        )
        return normalize_vectors(vectors)

    def health_check(self) -> dict:
        return {"status": "mock", "model_name": "hash_embedding", "dimension": self._dim}


class SentenceTransformerEmbedding(EmbeddingProvider):
    """Real sentence-transformers embedding provider.

    FIX 1: Uses global model cache to load the model only once.
    - Model: all-MiniLM-L6-v2 (384d)
    - Embeddings are L2-normalized to unit vectors
    - Batch encoding for speed
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        self._model_name = model_name or settings.SENTENCE_TRANSFORMER_MODEL
        self._dim: Optional[int] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load sentence-transformer model into global cache (once)."""
        global _st_model, _st_model_name
        if _st_model is not None and _st_model_name == self._model_name:
            # Already loaded — reuse
            self._dim = _st_model.get_sentence_embedding_dimension()
            logger.info("Reusing cached sentence-transformer model: %s (dim=%d)", self._model_name, self._dim)
            return
        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading sentence-transformer model: %s", self._model_name)
            _st_model = SentenceTransformer(self._model_name)
            _st_model_name = self._model_name
            self._dim = _st_model.get_sentence_embedding_dimension()
            logger.info("Model loaded — dimension=%d", self._dim)
        except Exception:
            logger.exception("Failed to load sentence-transformer model: %s", self._model_name)
            raise

    @property
    def dimension(self) -> int:
        if self._dim is None:
            self._load_model()
        return self._dim  # type: ignore[return-value]

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed texts using sentence-transformers.

        FIX 1: Batch encode with normalize_embeddings=True for unit vectors.

        Args:
            texts: Input strings.

        Returns:
            Normalized embedding matrix of shape (len(texts), dimension).
        """
        vectors = _st_model.encode(  # type: ignore[union-attr]
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2-normalize to unit vectors
            batch_size=64,  # batch encode for speed
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)

    def health_check(self) -> dict:
        status = "healthy" if _st_model is not None else "unhealthy"
        return {
            "status": status,
            "model_name": self._model_name,
            "dimension": self._dim or settings.EMBEDDING_DIM,
        }


def create_embedding_provider() -> EmbeddingProvider:
    """Factory: create the appropriate EmbeddingProvider based on SHIELD_MODE.

    FIX 1: lightweight/full modes ALWAYS use real embeddings.
    Falls back to mock only if SentenceTransformer load fails.

    Returns:
        Mock provider for ``mock`` mode, SentenceTransformer for others.
    """
    if settings.SHIELD_MODE == "mock":
        logger.info("Using MockEmbeddingProvider (hash-based)")
        return MockEmbeddingProvider(dimension=settings.EMBEDDING_DIM)
    else:
        # FIX 1: Always try real embeddings for lightweight/full
        try:
            logger.info("Using SentenceTransformerEmbedding (%s)", settings.SENTENCE_TRANSFORMER_MODEL)
            return SentenceTransformerEmbedding()
        except Exception:
            logger.warning("Falling back to MockEmbeddingProvider due to model load failure")
            return MockEmbeddingProvider(dimension=settings.EMBEDDING_DIM)
