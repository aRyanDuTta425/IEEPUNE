"""Meta-Jailbreak Detector — clusters known jailbreak prompts and scores new prompts by similarity.

Scoring pipeline (large-dataset calibration):
  1. Top-k centroid matching: mean of top-3 similarities (robust to cluster noise)
  2. Soft normalization: score = clip((raw - μ_neg) / (μ_pos - μ_neg), 0, 1)
  3. Temperature scaling: score = sigmoid(score / T), T configurable
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from shield.config import settings
from shield.core.clustering import ClusterResult, cluster_embeddings, load_clusters, save_clusters
from shield.core.embeddings import EmbeddingProvider
from shield.core.vector_index import VectorIndex, create_vector_index

logger = logging.getLogger(__name__)

# ── Default calibration constants (fallback when uncalibrated) ──────────────
# These are overridden by calibrate_from_distributions() when data is available.
DEFAULT_MU_NEG = 0.30   # mean raw similarity for benign prompts
DEFAULT_MU_POS = 1.00   # mean raw similarity for jailbreak prompts


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ez = math.exp(x)
        return ez / (1.0 + ez)


def calibrate_score(
    raw_cosine: float,
    mu_neg: float = DEFAULT_MU_NEG,
    mu_pos: float = DEFAULT_MU_POS,
    temperature: float = 0.5,
) -> float:
    """Calibrate raw cosine similarity to a jailbreak score.

    Pipeline:
      1. Soft normalize:  s = clip((raw - μ_neg) / (μ_pos - μ_neg), 0, 1)
      2. Temperature:     s = sigmoid(s / T)

    Args:
        raw_cosine: Raw cosine similarity from vector index.
        mu_neg: Mean similarity of benign prompts.
        mu_pos: Mean similarity of jailbreak prompts.
        temperature: Temperature for sigmoid contrast (lower = sharper).

    Returns:
        Calibrated score in [0.0, 1.0].
    """
    denom = mu_pos - mu_neg
    if denom <= 0:
        denom = 0.70  # safe fallback

    # Step 1: Soft normalization
    normalized = (raw_cosine - mu_neg) / denom
    normalized = float(np.clip(normalized, 0.0, 1.0))

    # Step 2: Temperature scaling
    # Map [0,1] → centered range for sigmoid, then apply
    centered = (normalized - 0.5) / temperature
    score = _sigmoid(centered)

    return float(np.clip(score, 0.0, 1.0))


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
        4. On new prompt: embed → top-k match → soft normalize → temperature scale

    Large-dataset calibration:
        Call calibrate_from_distributions() before evaluation to compute
        μ_neg and μ_pos from actual score distributions.
    """

    def __init__(self, embedding_provider: EmbeddingProvider) -> None:
        self._embedder = embedding_provider
        self._index: Optional[VectorIndex] = None
        self._cluster_result: Optional[ClusterResult] = None
        self._corpus: List[Dict[str, Any]] = []
        self._corpus_embeddings: Optional[np.ndarray] = None
        self._initialized = False
        self._calibrated_threshold: Optional[float] = None

        # Calibration parameters (data-driven or defaults)
        self._mu_neg: float = DEFAULT_MU_NEG
        self._mu_pos: float = DEFAULT_MU_POS
        self._temperature: float = settings.JAILBREAK_TEMPERATURE

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
            if self._corpus:
                texts = [p["text"] for p in self._corpus]
                self._corpus_embeddings = self._embedder.embed(texts)
                self._build_index_from_corpus(self._corpus_embeddings)
            else:
                self._index = create_vector_index(self._embedder.dimension)
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
        """Re-cluster the entire corpus from scratch."""
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
        self._corpus_embeddings = embeddings

        self._cluster_result = cluster_embeddings(
            embeddings, min_cluster_size=min_cluster_size
        )

        if save_to_disk:
            save_clusters(self._cluster_result)

        self._build_index_from_corpus(embeddings)
        return self._cluster_result

    def _build_index_from_corpus(self, embeddings: Optional[np.ndarray] = None) -> None:
        """Build vector index from ALL corpus embeddings for higher recall."""
        self._index = create_vector_index(self._embedder.dimension)
        if embeddings is not None and embeddings.size > 0:
            self._index.add(embeddings)
            logger.info("Indexed %d corpus embeddings (clusters=%d)",
                        len(embeddings), self.num_clusters)

    def calibrate_from_distributions(
        self, benign_prompts: List[str], n_sample: int = 50
    ) -> None:
        """Calibrate scoring from actual jailbreak/benign similarity distributions.

        Computes μ_neg (mean raw similarity of benign) and μ_pos (mean raw
        similarity of jailbreaks) to replace fixed constants.

        Args:
            benign_prompts: List of known-benign prompts.
            n_sample: Max corpus items to sample for speed.
        """
        if not self._corpus or self._index is None or self._index.size() == 0:
            logger.warning("Cannot calibrate — no corpus/index")
            return

        # Sample raw similarities for jailbreak prompts
        jb_raws = []
        sample = self._corpus[:n_sample]
        for p in sample:
            embedding = self._embedder.embed_single(p["text"]).reshape(1, -1)
            k = min(3, self._index.size())
            similarities, _ = self._index.search(embedding, k=k)
            raw = float(np.mean(similarities[0, :k])) if similarities.size > 0 else 0.0
            jb_raws.append(raw)

        # Sample raw similarities for benign prompts
        benign_raws = []
        for p in benign_prompts[:n_sample]:
            embedding = self._embedder.embed_single(p).reshape(1, -1)
            k = min(3, self._index.size())
            similarities, _ = self._index.search(embedding, k=k)
            raw = float(np.mean(similarities[0, :k])) if similarities.size > 0 else 0.0
            benign_raws.append(raw)

        if jb_raws and benign_raws:
            self._mu_neg = float(np.mean(benign_raws))
            self._mu_pos = float(np.mean(jb_raws))

            # Ensure separation exists
            if self._mu_pos <= self._mu_neg:
                self._mu_pos = self._mu_neg + 0.3

            logger.info(
                "Calibrated: μ_neg=%.4f μ_pos=%.4f (separation=%.4f)",
                self._mu_neg, self._mu_pos, self._mu_pos - self._mu_neg,
            )

    def detect(self, prompt: str) -> JailbreakDetectionResult:
        """Score a single prompt for jailbreak similarity.

        Uses top-k centroid matching, soft normalization, and temperature scaling.

        Args:
            prompt: User prompt to evaluate.

        Returns:
            JailbreakDetectionResult with calibrated score and match details.
        """
        if self._index is None or self._index.size() == 0:
            return JailbreakDetectionResult(jailbreak_score=0.0, details={"reason": "no clusters available"})

        embedding = self._embedder.embed_single(prompt).reshape(1, -1)

        # Top-k centroid matching: mean of top-3 for robustness
        k = min(3, self._index.size())
        similarities, indices = self._index.search(embedding, k=k)

        # Use mean of top-k similarities (robust to cluster noise)
        raw_score = float(np.mean(similarities[0, :k])) if similarities.size > 0 else 0.0
        raw_max = float(similarities[0, 0]) if similarities.size > 0 else 0.0
        matched_idx = int(indices[0, 0]) if indices.size > 0 else -1

        # Calibrate with soft normalization + temperature
        jailbreak_score = calibrate_score(
            raw_score,
            mu_neg=self._mu_neg,
            mu_pos=self._mu_pos,
            temperature=self._temperature,
        )
        is_jailbreak = jailbreak_score >= settings.JAILBREAK_THRESHOLD

        return JailbreakDetectionResult(
            jailbreak_score=jailbreak_score,
            matched_cluster=matched_idx if matched_idx >= 0 else None,
            matched_similarity=raw_max,
            is_jailbreak=is_jailbreak,
            details={
                "threshold": settings.JAILBREAK_THRESHOLD,
                "raw_cosine": raw_max,
                "raw_mean_top3": raw_score,
                "mu_neg": self._mu_neg,
                "mu_pos": self._mu_pos,
                "temperature": self._temperature,
                "calibrated_score": jailbreak_score,
                "corpus_size": len(self._corpus),
                "active_clusters": self.num_clusters,
            },
        )

    def auto_calibrate_threshold(self, benign_prompts: List[str]) -> float:
        """Auto-calibrate the jailbreak threshold from benign/jailbreak separation.

        Args:
            benign_prompts: List of known-benign prompts.

        Returns:
            Calibrated threshold.
        """
        if not self._corpus or self._index is None or self._index.size() == 0:
            return settings.JAILBREAK_THRESHOLD

        # Mean jailbreak score (corpus items against their own centroids)
        jb_scores = []
        for p in self._corpus[:50]:
            r = self.detect(p["text"])
            jb_scores.append(r.jailbreak_score)

        # Mean benign score
        benign_scores = [self.detect(p).jailbreak_score for p in benign_prompts]

        mean_jb = np.mean(jb_scores) if jb_scores else 0.5
        mean_benign = np.mean(benign_scores) if benign_scores else 0.0

        threshold = float((mean_jb + mean_benign) / 2)
        self._calibrated_threshold = threshold

        logger.info(
            "Auto-calibrated threshold: %.4f (jb_mean=%.4f, benign_mean=%.4f)",
            threshold, mean_jb, mean_benign,
        )
        return threshold

    def add_prompts(self, prompts: List[Dict[str, Any]], corpus_path: Optional[str] = None) -> int:
        """Add new prompts to the corpus without re-clustering."""
        self._corpus.extend(prompts)

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
            "mu_neg": self._mu_neg,
            "mu_pos": self._mu_pos,
        }
