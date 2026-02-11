"""Tests for clustering logic."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from shield.core.clustering import (
    ClusterResult,
    _single_cluster_fallback,
    cluster_embeddings,
    load_clusters,
    save_clusters,
)


class TestClustering:
    """Tests for HDBSCAN clustering pipeline."""

    def test_single_cluster_fallback(self) -> None:
        """When data is too small, should create a single cluster."""
        embeddings = np.random.default_rng(42).standard_normal((3, 384)).astype(np.float32)
        result = _single_cluster_fallback(embeddings)

        assert result.num_clusters == 1
        assert result.noise_count == 0
        assert result.centroids.shape == (1, 384)
        assert len(result.labels) == 3
        assert all(l == 0 for l in result.labels)

    def test_fallback_when_corpus_too_small(self) -> None:
        """cluster_embeddings should fallback when corpus < min_cluster_size."""
        embeddings = np.random.default_rng(42).standard_normal((3, 384)).astype(np.float32)
        result = cluster_embeddings(embeddings, min_cluster_size=5)

        assert result.num_clusters == 1, "Should fallback to single cluster"

    def test_cluster_clear_groups(self) -> None:
        """With well-separated clusters, should find them."""
        rng = np.random.default_rng(42)
        dim = 10
        # Create 3 clearly separated clusters
        cluster_centers = [
            np.ones(dim) * 10,
            np.ones(dim) * -10,
            np.ones(dim) * 0,
        ]
        embeddings = []
        for center in cluster_centers:
            for _ in range(20):
                embeddings.append(center + rng.standard_normal(dim) * 0.1)
        embeddings = np.array(embeddings, dtype=np.float32)

        result = cluster_embeddings(embeddings, min_cluster_size=5, min_samples=3)

        # Should find at least 2 clusters (HDBSCAN may merge close ones)
        assert result.num_clusters >= 2, f"Expected ≥2 clusters, got {result.num_clusters}"
        assert result.centroids.shape[0] == result.num_clusters

    def test_centroid_normalization(self) -> None:
        """Centroids should be L2 normalized."""
        embeddings = np.random.default_rng(42).standard_normal((3, 10)).astype(np.float32)
        result = _single_cluster_fallback(embeddings)

        norms = np.linalg.norm(result.centroids, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_cluster_sizes_correct(self) -> None:
        """Cluster sizes should match label counts."""
        embeddings = np.random.default_rng(42).standard_normal((10, 5)).astype(np.float32)
        result = _single_cluster_fallback(embeddings)

        for label, size in result.cluster_sizes.items():
            assert size == int((result.labels == label).sum())


class TestClusterPersistence:
    """Tests for saving and loading clusters."""

    def test_save_and_load(self) -> None:
        """Save then load should produce identical results."""
        embeddings = np.random.default_rng(42).standard_normal((10, 384)).astype(np.float32)
        original = _single_cluster_fallback(embeddings)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        try:
            save_clusters(original, path)
            loaded = load_clusters(path)

            assert loaded is not None
            assert loaded.num_clusters == original.num_clusters
            assert loaded.noise_count == original.noise_count
            np.testing.assert_array_equal(loaded.labels, original.labels)
            np.testing.assert_array_almost_equal(loaded.centroids, original.centroids)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_nonexistent_returns_none(self) -> None:
        """Loading from nonexistent path should return None."""
        result = load_clusters("/tmp/nonexistent_shield_test_clusters.pkl")
        assert result is None
