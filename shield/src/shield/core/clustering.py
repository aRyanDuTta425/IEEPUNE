"""HDBSCAN clustering for jailbreak prompt grouping."""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from shield.config import settings
from shield.core.utils import normalize_vectors

logger = logging.getLogger(__name__)


@dataclass
class ClusterResult:
    """Result of clustering jailbreak embeddings."""

    labels: np.ndarray  # cluster label per prompt (-1 = noise)
    centroids: np.ndarray  # shape (K, D)
    cluster_sizes: Dict[int, int] = field(default_factory=dict)
    num_clusters: int = 0
    noise_count: int = 0


def cluster_embeddings(
    embeddings: np.ndarray,
    min_cluster_size: Optional[int] = None,
    min_samples: Optional[int] = None,
    metric: Optional[str] = None,
) -> ClusterResult:
    """Cluster embeddings using HDBSCAN.

    Args:
        embeddings: Matrix of shape (N, D).
        min_cluster_size: Override from config.
        min_samples: Override from config.
        metric: Override from config.

    Returns:
        ClusterResult with labels, centroids, and metadata.
    """
    mcs = min_cluster_size or settings.HDBSCAN_MIN_CLUSTER_SIZE
    ms = min_samples or settings.HDBSCAN_MIN_SAMPLES
    met = metric or settings.CLUSTER_METRIC

    n_samples = len(embeddings)
    if n_samples < mcs:
        logger.warning(
            "Corpus size (%d) < min_cluster_size (%d) — using single cluster fallback",
            n_samples,
            mcs,
        )
        return _single_cluster_fallback(embeddings)

    try:
        import hdbscan

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            metric=met,
            core_dist_n_jobs=1,
        )
        labels = clusterer.fit_predict(embeddings)
    except ImportError:
        logger.warning("hdbscan not installed — using single cluster fallback")
        return _single_cluster_fallback(embeddings)
    except Exception:
        logger.exception("HDBSCAN clustering failed — using single cluster fallback")
        return _single_cluster_fallback(embeddings)

    return _build_result(embeddings, labels)


def _single_cluster_fallback(embeddings: np.ndarray) -> ClusterResult:
    """Fallback: treat all embeddings as one cluster."""
    labels = np.zeros(len(embeddings), dtype=np.int32)
    centroid = np.mean(embeddings, axis=0, keepdims=True)
    centroid = normalize_vectors(centroid)
    return ClusterResult(
        labels=labels,
        centroids=centroid,
        cluster_sizes={0: len(embeddings)},
        num_clusters=1,
        noise_count=0,
    )


def _build_result(embeddings: np.ndarray, labels: np.ndarray) -> ClusterResult:
    """Compute centroids and metadata from labels."""
    unique_labels = set(labels)
    unique_labels.discard(-1)  # exclude noise

    if not unique_labels:
        logger.warning("No clusters found (all noise) — using single cluster fallback")
        return _single_cluster_fallback(embeddings)

    centroids_list: List[np.ndarray] = []
    cluster_sizes: Dict[int, int] = {}

    for label in sorted(unique_labels):
        mask = labels == label
        cluster_sizes[int(label)] = int(mask.sum())
        centroid = np.mean(embeddings[mask], axis=0)
        centroids_list.append(centroid)

    centroids = normalize_vectors(np.array(centroids_list, dtype=np.float32))
    noise_count = int((labels == -1).sum())

    logger.info(
        "Clustering complete — %d clusters, %d noise points out of %d total",
        len(unique_labels),
        noise_count,
        len(embeddings),
    )

    return ClusterResult(
        labels=labels,
        centroids=centroids,
        cluster_sizes=cluster_sizes,
        num_clusters=len(unique_labels),
        noise_count=noise_count,
    )


def save_clusters(result: ClusterResult, path: Optional[str] = None) -> None:
    """Persist cluster result to disk.

    Args:
        result: ClusterResult to save.
        path: File path override (default: from config).
    """
    filepath = Path(path or settings.CLUSTER_CACHE_PATH)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(result, f)
    logger.info("Clusters saved to %s", filepath)


def load_clusters(path: Optional[str] = None) -> Optional[ClusterResult]:
    """Load cluster result from disk.

    Args:
        path: File path override.

    Returns:
        ClusterResult or None if file not found.
    """
    filepath = Path(path or settings.CLUSTER_CACHE_PATH)
    if not filepath.exists():
        logger.info("No cluster cache found at %s", filepath)
        return None
    with open(filepath, "rb") as f:
        result = pickle.load(f)
    logger.info("Clusters loaded from %s — %d clusters", filepath, result.num_clusters)
    return result
