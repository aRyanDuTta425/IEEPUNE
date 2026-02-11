"""Privacy Consent Violation Predictor (PCVP).

Pipeline:
    1. Identity Extraction (face detection + age estimation)
    2. Transformation Intent Classification (prompt → transform types)
    3. Risk Matrix Lookup
    4. Compute privacy_score = max across all (identity, transform) combinations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from shield.config import settings
from shield.core.risk_matrix import lookup_risk
from shield.ml.base import (
    AgeEstimate,
    BaseAgeEstimator,
    BaseFaceDetector,
    BaseTransformClassifier,
    FaceDetection,
    TransformationResult,
)

logger = logging.getLogger(__name__)


@dataclass
class PrivacyPredictionResult:
    """Output of privacy consent violation prediction."""

    privacy_score: float
    identity_type: str  # "minor" | "public_figure" | "private_person" | "none"
    transformations: List[str]
    faces_detected: int
    is_violation: bool
    details: Dict[str, Any] = field(default_factory=dict)


class PrivacyPredictor:
    """Predict privacy consent violations for image editing requests.

    Uses abstract interfaces for face detection, age estimation, and
    transformation classification so that mock and real implementations
    are swappable via deployment mode.
    """

    def __init__(
        self,
        face_detector: BaseFaceDetector,
        age_estimator: BaseAgeEstimator,
        transform_classifier: BaseTransformClassifier,
    ) -> None:
        self._face_detector = face_detector
        self._age_estimator = age_estimator
        self._transform_classifier = transform_classifier

    def predict(
        self,
        prompt: str,
        image: Optional[np.ndarray] = None,
    ) -> PrivacyPredictionResult:
        """Run the full PCVP pipeline.

        Args:
            prompt: Text description of desired editing.
            image: Optional image as numpy array (H, W, C) in RGB.

        Returns:
            PrivacyPredictionResult with score and details.
        """
        # Step 1: Detect faces
        if image is not None:
            faces = self._face_detector.detect(image)
        else:
            faces = []

        # Step 2: Classify transformation intent
        transforms = self._transform_classifier.classify(prompt, image)

        # Step 3: If no faces, privacy_score = 0
        if not faces:
            return PrivacyPredictionResult(
                privacy_score=0.0,
                identity_type="none",
                transformations=[t.transform_type for t in transforms],
                faces_detected=0,
                is_violation=False,
                details={"reason": "no faces detected in image"},
            )

        # Step 4: For each face, determine identity type
        max_privacy_score = 0.0
        worst_identity = "private_person"
        all_risk_details: List[Dict[str, Any]] = []

        for face in faces:
            identity_type = self._classify_identity(face)

            # Step 5: Risk matrix lookup for each (identity, transform) pair
            for transform in transforms:
                risk_score = lookup_risk(identity_type, transform.transform_type)

                all_risk_details.append({
                    "identity_type": identity_type,
                    "transform_type": transform.transform_type,
                    "transform_confidence": transform.confidence,
                    "risk_score": risk_score,
                })

                if risk_score > max_privacy_score:
                    max_privacy_score = risk_score
                    worst_identity = identity_type

        is_violation = max_privacy_score >= settings.PRIVACY_THRESHOLD

        return PrivacyPredictionResult(
            privacy_score=max_privacy_score,
            identity_type=worst_identity,
            transformations=[t.transform_type for t in transforms],
            faces_detected=len(faces),
            is_violation=is_violation,
            details={
                "risk_lookups": all_risk_details,
                "threshold": settings.PRIVACY_THRESHOLD,
            },
        )

    def _classify_identity(self, face: FaceDetection) -> str:
        """Classify identity type from a face detection.

        Args:
            face: Face detection result.

        Returns:
            Identity type string.
        """
        # Estimate age
        face_crop = np.zeros((160, 160, 3), dtype=np.uint8)  # placeholder crop
        age_result = self._age_estimator.estimate(face_crop)

        if age_result.is_minor:
            return "minor"
        elif face.is_public_figure:
            return "public_figure"
        else:
            return "private_person"

    def health_check(self) -> dict:
        return {
            "face_detector": self._face_detector.health_check(),
            "age_estimator": self._age_estimator.health_check(),
            "transform_classifier": self._transform_classifier.health_check(),
        }
