"""Meta-Jailbreak Detector — clusters known jailbreak prompts and scores new prompts by similarity.

FIX 2: Calibrated similarity scoring with score scaling and auto-threshold.
- Raw cosine similarity is rescaled: score = clip((cosine - 0.3) / 0.7, 0, 1)
- This widens the gap between benign (~0.1-0.3 raw) and jailbreak (~0.6-0.9 raw) prompts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from shield.config import settings
from shield.core.clustering import ClusterResult, cluster_embeddings, load_clusters, save_clusters
from shield.core.embeddings import EmbeddingProvider
from shield.core.vector_index import VectorIndex, create_vector_index

logger = logging.getLogger(__name__)

# ── FIX 2: Score calibration constants ────────────────────────────────────────
# Benign prompts typically have raw cosine < 0.3 against jailbreak centroids.
# Jailbreak prompts typically have raw cosine > 0.55.
# Rescaling: score = clip((raw - BASELINE) / (1 - BASELINE), 0, 1)
SCORE_BASELINE = 0.3
SCORE_RANGE = 1.0 - SCORE_BASELINE  # 0.7


def calibrate_score(raw_cosine: float) -> float:
    """Rescale raw cosine similarity to calibrated jailbreak score.

    FIX 2: Maps [0.3, 1.0] → [0.0, 1.0], clips below 0.3 to 0.

    Args:
        raw_cosine: Raw cosine similarity from vector index.

    Returns:
        Calibrated score in [0.0, 1.0].
    """
    scaled = (raw_cosine - SCORE_BASELINE) / SCORE_RANGE
    return float(np.clip(scaled, 0.0, 1.0))


@dataclass
class JailbreakDetectionResult:
    """Output of jailbreak detection on a single prompt."""

    jailbreak_score: float
    matched_cluster: Optional[int] = None
    matched_similarity: float = 0.0
    is_jailbreak: bool = False
    details: Dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.details is None:
            self.details = {}


class JailbreakDetector:
    """Detect jailbreak prompts by semantic similarity to known jailbreak clusters.

    Pipeline:
        1. Embed jailbreak corpus with EmbeddingProvider
        2. Cluster embeddings with HDBSCAN → compute cluster centroids
        3. Index centroids in VectorIndex (FAISS / numpy)
        4. On new prompt: embed → search nearest centroid → calibrate score

    FIX 2: Uses calibrated scoring instead of raw cosine similarity.
    """

    def __init__(self, embedding_provider: EmbeddingProvider) -> None:
        self._embedder = embedding_provider
        self._index: Optional[VectorIndex] = None
        self._cluster_result: Optional[ClusterResult] = None
        self._corpus: List[Dict[str, Any]] = []
        self._initialized = False
        self._calibrated_threshold: Optional[float] = None

    @property
    def corpus_size(self) -> int:
        return len(self._corpus)

    @property
    def num_clusters(self) -> int:
        return self._cluster_result.num_clusters if self._cluster_result else 0

    def initialize(self, corpus_path: Optional[str] = None) -> None:
        """Load jailbreak corpus and build the detection index.

        Tries to load cached clusters first; falls back to full clustering.

        Args:
            corpus_path: Path to jailbreak corpus JSON.
        """
        corpus_file = Path(corpus_path or settings.JAILBREAK_CORPUS_PATH)

        # Load corpus
        if corpus_file.exists():
            with open(corpus_file) as f:
                self._corpus = json.load(f)
            logger.info("Loaded jailbreak corpus — %d prompts from %s", len(self._corpus), corpus_file)
        else:
            logger.warning("Jailbreak corpus not found at %s — starting empty", corpus_file)
            self._corpus = []

        # Try cached clusters first
        cached = load_clusters()
        if cached is not None:
            self._cluster_result = cached
            self._build_index_from_clusters()
            self._initialized = True
            return

        # Full re-cluster
        if self._corpus:
            self.refresh_clusters()
        else:
            self._index = create_vector_index(self._embedder.dimension)

        self._initialized = True

    def refresh_clusters(
        self, min_cluster_size: Optional[int] = None, save_to_disk: bool = True
    ) -> ClusterResult:
        """Re-cluster the entire corpus from scratch.

        Args:
            min_cluster_size: Override HDBSCAN min_cluster_size.
            save_to_disk: Whether to persist cluster results.

        Returns:
            Clustering result.
        """
        texts = [p["text"] for p in self._corpus]
        if not texts:
            logger.warning("Empty corpus — cannot cluster")
            self._cluster_result = ClusterResult(
                labels=np.array([]), centroids=np.empty((0, self._embedder.dimension)), num_clusters=0
            )
            self._index = create_vector_index(self._embedder.dimension)
            return self._cluster_result

        logger.info("Embedding %d prompts for clustering…", len(texts))
        embeddings = self._embedder.embed(texts)

        self._cluster_result = cluster_embeddings(
            embeddings, min_cluster_size=min_cluster_size
        )

        if save_to_disk:
            save_clusters(self._cluster_result)

        self._build_index_from_clusters()
        return self._cluster_result

    def _build_index_from_clusters(self) -> None:
        """Build vector index from cluster centroids."""
        self._index = create_vector_index(self._embedder.dimension)
        if self._cluster_result and self._cluster_result.centroids.size > 0:
            self._index.add(self._cluster_result.centroids)
            logger.info("Indexed %d cluster centroids", self._cluster_result.num_clusters)

    def detect(self, prompt: str) -> JailbreakDetectionResult:
        """Score a single prompt for jailbreak similarity.

        FIX 2: Uses calibrated scoring — raw cosine is rescaled from
        [0.3, 1.0] → [0.0, 1.0] for better separation.

        Args:
            prompt: User prompt to evaluate.

        Returns:
            JailbreakDetectionResult with calibrated score and match details.
        """
        if self._index is None or self._index.size() == 0:
            return JailbreakDetectionResult(jailbreak_score=0.0, details={"reason": "no clusters available"})

        embedding = self._embedder.embed_single(prompt).reshape(1, -1)

        # Search top-3 nearest centroids for robustness
        k = min(3, self._index.size())
        similarities, indices = self._index.search(embedding, k=k)

        # Use max similarity across top-k matches
        raw_max = float(similarities[0, 0]) if similarities.size > 0 else 0.0
        matched_idx = int(indices[0, 0]) if indices.size > 0 else -1

        # FIX 2: Calibrate the raw cosine similarity
        jailbreak_score = calibrate_score(raw_max)
        is_jailbreak = jailbreak_score >= settings.JAILBREAK_THRESHOLD

        return JailbreakDetectionResult(
            jailbreak_score=jailbreak_score,
            matched_cluster=matched_idx if matched_idx >= 0 else None,
            matched_similarity=raw_max,
            is_jailbreak=is_jailbreak,
            details={
                "threshold": settings.JAILBREAK_THRESHOLD,
                "raw_cosine": raw_max,
                "calibrated_score": jailbreak_score,
                "corpus_size": len(self._corpus),
                "active_clusters": self.num_clusters,
            },
        )

    def auto_calibrate_threshold(self, benign_prompts: List[str]) -> float:
        """Auto-calibrate the jailbreak threshold from benign/jailbreak separation.

        FIX 2: Computes mean similarity for corpus (jailbreak) and benign prompts,
        sets threshold at midpoint.

        Args:
            benign_prompts: List of known-benign prompts.

        Returns:
            Calibrated threshold.
        """
        if not self._corpus or self._index is None or self._index.size() == 0:
            return settings.JAILBREAK_THRESHOLD

        # Mean jailbreak score (corpus items against their own centroids)
        jb_scores = []
        for p in self._corpus[:50]:  # sample for speed
            r = self.detect(p["text"])
            jb_scores.append(r.jailbreak_score)

        # Mean benign score
        benign_scores = [self.detect(p).jailbreak_score for p in benign_prompts]

        mean_jb = np.mean(jb_scores) if jb_scores else 0.5
        mean_benign = np.mean(benign_scores) if benign_scores else 0.0

        # Threshold = midpoint between benign and jailbreak means
        threshold = float((mean_jb + mean_benign) / 2)
        self._calibrated_threshold = threshold

        logger.info(
            "Auto-calibrated threshold: %.4f (jb_mean=%.4f, benign_mean=%.4f)",
            threshold, mean_jb, mean_benign,
        )
        return threshold

    def add_prompts(self, prompts: List[Dict[str, Any]], corpus_path: Optional[str] = None) -> int:
        """Add new prompts to the corpus without re-clustering.

        Args:
            prompts: List of prompt dicts with ``text``, optional ``category`` and ``severity``.
            corpus_path: Path to save updated corpus.

        Returns:
            Number of prompts added.
        """
        self._corpus.extend(prompts)

        # Persist to disk
        filepath = Path(corpus_path or settings.JAILBREAK_CORPUS_PATH)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self._corpus, f, indent=2)

        logger.info("Added %d prompts to corpus (total: %d). Re-cluster required.", len(prompts), len(self._corpus))
        return len(prompts)

    def health_check(self) -> dict:
        return {
            "status": "healthy" if self._initialized else "unhealthy",
            "corpus_size": len(self._corpus),
            "active_clusters": self.num_clusters,
            "index_backend": self._index.health_check().get("backend", "none") if self._index else "none",
            "calibrated_threshold": self._calibrated_threshold,
        }
