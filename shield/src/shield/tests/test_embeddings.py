"""Tests for embedding providers."""

from __future__ import annotations

import numpy as np
import pytest

from shield.core.embeddings import MockEmbeddingProvider, create_embedding_provider


class TestMockEmbeddingProvider:
    """Tests for MockEmbeddingProvider."""

    def test_deterministic_output(self, mock_embedding_provider: MockEmbeddingProvider) -> None:
        """Same text must produce exactly the same vector."""
        text = "Hello, world!"
        v1 = mock_embedding_provider.embed_single(text)
        v2 = mock_embedding_provider.embed_single(text)
        np.testing.assert_array_equal(v1, v2)

    def test_different_texts_different_vectors(self, mock_embedding_provider: MockEmbeddingProvider) -> None:
        """Different texts should produce different vectors."""
        v1 = mock_embedding_provider.embed_single("Hello")
        v2 = mock_embedding_provider.embed_single("Goodbye")
        assert not np.allclose(v1, v2)

    def test_dimension(self, mock_embedding_provider: MockEmbeddingProvider) -> None:
        """Must produce vectors of configured dimension."""
        assert mock_embedding_provider.dimension == 384
        vec = mock_embedding_provider.embed_single("test")
        assert vec.shape == (384,)

    def test_normalization(self, mock_embedding_provider: MockEmbeddingProvider) -> None:
        """Mock embeddings must be unit vectors."""
        vec = mock_embedding_provider.embed_single("test normalization")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"

    def test_batch_embedding(self, mock_embedding_provider: MockEmbeddingProvider) -> None:
        """Batch embedding should return correct shape."""
        texts = [f"text {i}" for i in range(100)]
        result = mock_embedding_provider.embed(texts)
        assert result.shape == (100, 384)

    def test_batch_determinism(self, mock_embedding_provider: MockEmbeddingProvider) -> None:
        """Batch vs single embedding must produce same vectors."""
        texts = ["alpha", "beta", "gamma"]
        batch = mock_embedding_provider.embed(texts)
        for i, text in enumerate(texts):
            single = mock_embedding_provider.embed_single(text)
            np.testing.assert_array_almost_equal(batch[i], single, decimal=5)

    def test_health_check(self, mock_embedding_provider: MockEmbeddingProvider) -> None:
        """Health check should report mock status."""
        health = mock_embedding_provider.health_check()
        assert health["status"] == "mock"
        assert health["dimension"] == 384

    def test_custom_dimension(self) -> None:
        """Support custom dimensions."""
        provider = MockEmbeddingProvider(dimension=128, seed=99)
        assert provider.dimension == 128
        vec = provider.embed_single("test")
        assert vec.shape == (128,)


class TestEmbeddingFactory:
    """Tests for the create_embedding_provider factory."""

    def test_mock_mode_creates_mock(self) -> None:
        """Factory should return MockEmbeddingProvider in mock mode."""
        provider = create_embedding_provider()
        assert isinstance(provider, MockEmbeddingProvider)
