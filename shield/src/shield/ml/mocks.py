"""Mock implementations of ML models for testing and CI/CD.

FIX 5: Added SemanticTransformClassifier that uses embedding similarity
for zero-shot classification in lightweight mode.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from shield.ml.base import (
    AgeEstimate,
    BaseAgeEstimator,
    BaseFaceDetector,
    BaseTransformClassifier,
    FaceDetection,
    TransformationResult,
)

logger = logging.getLogger(__name__)


class MockFaceDetector(BaseFaceDetector):
    """Deterministic mock face detector that returns configurable results.

    Useful for testing without loading heavy CV models.
    """

    def __init__(
        self,
        default_detections: Optional[List[FaceDetection]] = None,
        detect_faces: bool = True,
    ) -> None:
        self._default_detections = default_detections
        self._detect_faces = detect_faces

    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        """Return mock face detections.

        Args:
            image: Ignored in mock mode.

        Returns:
            Configurable face detections, or a single default detection.
        """
        if not self._detect_faces:
            return []
        if self._default_detections is not None:
            return self._default_detections
        # Return one default detection
        return [
            FaceDetection(
                bbox=(100, 100, 200, 200),
                confidence=0.99,
                embedding=np.random.default_rng(42).standard_normal(512).astype(np.float32),
                is_public_figure=False,
            )
        ]

    def health_check(self) -> dict:
        return {"status": "mock", "model_name": "mock_face_detector"}


class MockAgeEstimator(BaseAgeEstimator):
    """Deterministic mock age estimator.

    Returns configurable age with is_minor derived from age < 18.
    """

    def __init__(self, default_age: int = 25) -> None:
        self._default_age = default_age

    def estimate(self, face_crop: np.ndarray) -> AgeEstimate:
        """Return mock age estimate.

        Args:
            face_crop: Ignored in mock mode.

        Returns:
            AgeEstimate with configurable default age.
        """
        return AgeEstimate(
            age=self._default_age,
            is_minor=self._default_age < 18,
            confidence=0.95,
        )

    def health_check(self) -> dict:
        return {"status": "mock", "model_name": "mock_age_estimator", "default_age": self._default_age}


class MockTransformClassifier(BaseTransformClassifier):
    """Keyword-based transformation intent classifier.

    Uses regex patterns to detect transformation types from prompt text.
    """

    # Mapping of regex patterns → transformation type
    PATTERNS: List[Tuple[str, str]] = [
        (r"\b(remove\s+(clothes|clothing|jacket|shirt|dress|pants|top))\b", "clothing_removal"),
        (r"\b(undress|strip|naked|nude)\b", "clothing_removal"),
        (r"\b(bikini|lingerie|underwear|swimsuit|revealing)\b", "sexualization"),
        (r"\b(sexy|seductive|provocative|erotic)\b", "sexualization"),
        (r"\b(swap\s+face|replace\s+face|replace\s+head|face\s+swap)\b", "face_swap"),
        (r"\b(make\s+(them|her|him)\s+younger|younger\s+version|as\s+a\s+child)\b", "age_regression"),
        (r"\b(make\s+(them|her|him)\s+older|older\s+version|age\s+progression)\b", "age_progression"),
        (r"\b(change\s+(pose|position)|different\s+pose)\b", "pose_change"),
        (r"\b(change\s+background|different\s+background|new\s+background)\b", "background_change"),
        (r"\b(change\s+face|modify\s+face|alter\s+face|facial\s+surgery)\b", "facial_modification"),
    ]

    def classify(self, prompt: str, image: Optional[np.ndarray] = None) -> List[TransformationResult]:
        """Classify transformation intent via keyword matching.

        Args:
            prompt: Text prompt to classify.
            image: Ignored in keyword mode.

        Returns:
            List of detected transformation types.
        """
        results: List[TransformationResult] = []
        lower = prompt.lower()
        seen: set = set()

        for pattern, transform_type in self.PATTERNS:
            if transform_type not in seen and re.search(pattern, lower):
                results.append(
                    TransformationResult(transform_type=transform_type, confidence=0.85)
                )
                seen.add(transform_type)

        return results

    def health_check(self) -> dict:
        return {"status": "mock", "model_name": "keyword_transform_classifier"}


class SemanticTransformClassifier(BaseTransformClassifier):
    """FIX 5: Embedding-based zero-shot transformation classifier.

    Uses precomputed label embeddings and cosine similarity to classify
    transformation intent. Provides pseudo-CLIP behavior without heavy models.

    Label groups are mapped to transformation types. If the prompt embedding's
    cosine similarity to any label in a group exceeds the threshold, the
    corresponding transformation is detected.
    """

    # Label descriptions → transformation type
    LABEL_GROUPS: Dict[str, List[str]] = {
        "clothing_removal": [
            "remove clothes from person",
            "undress this person",
            "make person naked",
            "nude photo",
            "strip clothing off",
            "remove all clothing",
        ],
        "sexualization": [
            "make this person sexy",
            "sexualize this image",
            "make person wear bikini",
            "erotic photo of person",
            "provocative revealing outfit",
            "seductive pose",
        ],
        "face_swap": [
            "swap this person's face with someone else",
            "replace face with celebrity face",
            "face swap onto another body",
            "put someone else's face on this body",
            "celebrity face on this person",
        ],
        "age_regression": [
            "make this person look younger",
            "turn person into a child",
            "younger version of person",
            "make person look like a teenager",
            "age regression younger",
        ],
        "age_progression": [
            "make this person look older",
            "age this person up",
            "older version of this person",
            "age progression older",
        ],
        "pose_change": [
            "change the pose of this person",
            "different body position",
            "change stance",
            "alter body pose",
        ],
        "background_change": [
            "change the background",
            "different background setting",
            "new background scene",
            "replace background",
        ],
        "facial_modification": [
            "change this person's face",
            "modify facial features",
            "alter face appearance",
            "facial surgery modification",
        ],
    }

    SIMILARITY_THRESHOLD = 0.55  # FIX 5: classify if similarity > 0.55

    def __init__(self, embedding_provider: Optional[object] = None) -> None:
        """Initialize with an embedding provider.

        Args:
            embedding_provider: Must have ``embed()`` method. If None, falls back
                to keyword matching.
        """
        self._embedder = embedding_provider
        self._label_embeddings: Optional[Dict[str, np.ndarray]] = None
        self._keyword_fallback = MockTransformClassifier()

        if self._embedder is not None:
            self._precompute_labels()

    def _precompute_labels(self) -> None:
        """Precompute embeddings for all label descriptions.

        FIX 5: Each transform type gets the mean embedding of its label group.
        """
        if self._embedder is None:
            return

        self._label_embeddings = {}
        for transform_type, labels in self.LABEL_GROUPS.items():
            try:
                vecs = self._embedder.embed(labels)  # type: ignore[union-attr]
                # Mean of label embeddings, then normalize
                mean_vec = vecs.mean(axis=0)
                norm = np.linalg.norm(mean_vec)
                if norm > 0:
                    mean_vec = mean_vec / norm
                self._label_embeddings[transform_type] = mean_vec
            except Exception as e:
                logger.warning("Failed to embed labels for %s: %s", transform_type, e)

        logger.info("Precomputed label embeddings for %d transform types", len(self._label_embeddings))

    def classify(self, prompt: str, image: Optional[np.ndarray] = None) -> List[TransformationResult]:
        """Classify transformation intent via embedding similarity.

        FIX 5: Computes cosine similarity between prompt embedding and each
        label group's mean embedding. Classifies if similarity > threshold.

        Falls back to keyword matching if embeddings are unavailable.

        Args:
            prompt: Text prompt to classify.
            image: Ignored in embedding mode.

        Returns:
            List of detected transformation types with confidence scores.
        """
        if self._label_embeddings is None or self._embedder is None:
            return self._keyword_fallback.classify(prompt, image)

        try:
            prompt_vec = self._embedder.embed_single(prompt)  # type: ignore[union-attr]
        except Exception:
            return self._keyword_fallback.classify(prompt, image)

        results: List[TransformationResult] = []

        for transform_type, label_vec in self._label_embeddings.items():
            sim = float(np.dot(prompt_vec, label_vec))
            if sim > self.SIMILARITY_THRESHOLD:
                results.append(
                    TransformationResult(
                        transform_type=transform_type,
                        confidence=round(sim, 4),
                    )
                )

        # If embedding-based detection found nothing, try keyword fallback too
        if not results:
            results = self._keyword_fallback.classify(prompt, image)

        return results

    def health_check(self) -> dict:
        status = "healthy" if self._label_embeddings is not None else "degraded"
        return {
            "status": status,
            "model_name": "semantic_transform_classifier",
            "label_groups": len(self.LABEL_GROUPS),
            "precomputed": self._label_embeddings is not None,
        }
