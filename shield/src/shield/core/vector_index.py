"""Vector index for nearest-neighbor search — FAISS primary, numpy fallback."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

from shield.core.utils import normalize_vectors

logger = logging.getLogger(__name__)


class VectorIndex(ABC):
    """Abstract vector index for cosine similarity search."""

    @abstractmethod
    def add(self, vectors: np.ndarray) -> None:
        """Add normalized vectors to the index.

        Args:
            vectors: Matrix of shape (N, D), must be L2-normalized.
        """
        ...

    @abstractmethod
    def search(self, query: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Find k nearest neighbors by inner product (cosine similarity).

        Args:
            query: Query vector(s) of shape (Q, D), must be normalized.
            k: Number of neighbors.

        Returns:
            Tuple of (similarities, indices) each of shape (Q, k).
        """
        ...

    @abstractmethod
    def size(self) -> int:
        """Return number of vectors in the index."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear all vectors."""
        ...

    @abstractmethod
    def health_check(self) -> dict:
        """Return health status."""
        ...


class FaissIndex(VectorIndex):
    """FAISS IndexFlatIP (inner product) for cosine similarity search.

    Vectors must be L2-normalized before adding so that inner product = cosine similarity.
    """

    def __init__(self, dimension: int) -> None:
        import faiss

        self._dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)
        logger.info("FaissIndex initialized — dimension=%d", dimension)

    def add(self, vectors: np.ndarray) -> None:
        vectors = normalize_vectors(vectors).astype(np.float32)
        self._index.add(vectors)

    def search(self, query: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        query = normalize_vectors(query.reshape(-1, self._dimension)).astype(np.float32)
        k = min(k, self._index.ntotal)
        if k == 0:
            empty = np.array([]).reshape(query.shape[0], 0)
            return empty, empty.astype(np.int64)
        similarities, indices = self._index.search(query, k)
        return similarities, indices

    def size(self) -> int:
        return self._index.ntotal

    def reset(self) -> None:
        self._index.reset()

    def health_check(self) -> dict:
        return {
            "status": "healthy",
            "backend": "faiss",
            "index_size": self._index.ntotal,
            "dimension": self._dimension,
        }


class NumpyFallbackIndex(VectorIndex):
    """Pure-numpy cosine similarity fallback when FAISS is unavailable."""

    def __init__(self, dimension: int) -> None:
        self._dimension = dimension
        self._vectors: Optional[np.ndarray] = None
        logger.warning("Using NumpyFallbackIndex — FAISS not available; performance may be degraded")

    def add(self, vectors: np.ndarray) -> None:
        vectors = normalize_vectors(vectors).astype(np.float32)
        if self._vectors is None:
            self._vectors = vectors
        else:
            self._vectors = np.vstack([self._vectors, vectors])

    def search(self, query: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        query = normalize_vectors(query.reshape(-1, self._dimension)).astype(np.float32)
        if self._vectors is None or len(self._vectors) == 0:
            empty = np.array([]).reshape(query.shape[0], 0)
            return empty, empty.astype(np.int64)
        k = min(k, len(self._vectors))
        # Inner product = cosine similarity for normalized vectors
        similarities = query @ self._vectors.T  # (Q, N)
        indices = np.argsort(-similarities, axis=1)[:, :k]
        sims = np.take_along_axis(similarities, indices, axis=1)
        return sims, indices

    def size(self) -> int:
        return 0 if self._vectors is None else len(self._vectors)

    def reset(self) -> None:
        self._vectors = None

    def health_check(self) -> dict:
        sz = self.size()
        return {
            "status": "degraded",
            "backend": "numpy_fallback",
            "index_size": sz,
            "dimension": self._dimension,
        }


def create_vector_index(dimension: int) -> VectorIndex:
    """Factory: create the best available vector index.

    Tries FAISS first, falls back to numpy.

    Args:
        dimension: Vector dimension.

    Returns:
        A VectorIndex implementation.
    """
    try:
        return FaissIndex(dimension)
    except ImportError:
        logger.warning("faiss-cpu not installed — falling back to numpy index")
        return NumpyFallbackIndex(dimension)
