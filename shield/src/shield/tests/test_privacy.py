"""Tests for Privacy Consent Violation Predictor."""

from __future__ import annotations

import numpy as np
import pytest

from shield.core.privacy_predictor import PrivacyPredictor
from shield.core.risk_matrix import PRIVACY_RISK_MATRIX, RISK_LEVEL_SCORES, lookup_risk
from shield.ml.mocks import MockAgeEstimator, MockFaceDetector, MockTransformClassifier


class TestRiskMatrix:
    """Tests for the 2D privacy risk matrix."""

    def test_all_combinations_covered(self) -> None:
        """Matrix must cover all identity × transform combinations."""
        identities = ["public_figure", "private_person", "minor"]
        transforms = [
            "clothing_removal", "sexualization", "pose_change", "face_swap",
            "age_regression", "age_progression", "background_change", "facial_modification",
        ]
        for identity in identities:
            for transform in transforms:
                score = lookup_risk(identity, transform)
                assert 0.0 <= score <= 1.0, f"({identity}, {transform}) → {score}"

    def test_score_conversion(self) -> None:
        """Risk levels should map to correct scores."""
        assert RISK_LEVEL_SCORES[0] == 0.0
        assert RISK_LEVEL_SCORES[1] == 0.5
        assert RISK_LEVEL_SCORES[2] == 0.8
        assert RISK_LEVEL_SCORES[3] == 0.95

    def test_minor_highest_risk(self) -> None:
        """Minor + most transforms should be critical (0.95)."""
        assert lookup_risk("minor", "clothing_removal") == 0.95
        assert lookup_risk("minor", "sexualization") == 0.95
        assert lookup_risk("minor", "face_swap") == 0.95

    def test_background_change_lowest(self) -> None:
        """Background change should have lowest risk for all identities."""
        assert lookup_risk("public_figure", "background_change") == 0.0
        assert lookup_risk("private_person", "background_change") == 0.5
        assert lookup_risk("minor", "background_change") == 0.5

    def test_unknown_identity_returns_zero(self) -> None:
        """Unknown identity type should return 0.0."""
        assert lookup_risk("unknown_type", "clothing_removal") == 0.0

    def test_unknown_transform_returns_zero(self) -> None:
        """Unknown transform type should return 0.0."""
        assert lookup_risk("private_person", "teleportation") == 0.0


class TestPrivacyPredictor:
    """Tests for the full PCVP pipeline."""

    def test_no_faces_zero_score(
        self, mock_face_detector_no_faces: MockFaceDetector, mock_age_estimator: MockAgeEstimator,
        mock_transform_classifier: MockTransformClassifier,
    ) -> None:
        """No faces detected → privacy_score = 0.0."""
        predictor = PrivacyPredictor(mock_face_detector_no_faces, mock_age_estimator, mock_transform_classifier)
        result = predictor.predict("Remove their clothes", image=np.zeros((224, 224, 3), dtype=np.uint8))
        assert result.privacy_score == 0.0
        assert result.faces_detected == 0
        assert not result.is_violation

    def test_private_person_sexualization_blocks(
        self, mock_face_detector: MockFaceDetector, mock_age_estimator: MockAgeEstimator,
        mock_transform_classifier: MockTransformClassifier, dummy_image: np.ndarray,
    ) -> None:
        """Private person + sexualization → high score → block."""
        predictor = PrivacyPredictor(mock_face_detector, mock_age_estimator, mock_transform_classifier)
        result = predictor.predict("Make this person wear a bikini", image=dummy_image)
        assert result.privacy_score == 0.95
        assert result.is_violation
        assert result.identity_type == "private_person"

    def test_minor_any_transform_critical(
        self, mock_face_detector: MockFaceDetector, mock_age_estimator_minor: MockAgeEstimator,
        mock_transform_classifier: MockTransformClassifier, dummy_image: np.ndarray,
    ) -> None:
        """Minor + sexualization → privacy_score = 0.95 → block."""
        predictor = PrivacyPredictor(mock_face_detector, mock_age_estimator_minor, mock_transform_classifier)
        result = predictor.predict("Make them sexy and seductive", image=dummy_image)
        assert result.privacy_score == 0.95
        assert result.is_violation
        assert result.identity_type == "minor"

    def test_no_image_no_privacy_issue(
        self, mock_face_detector: MockFaceDetector, mock_age_estimator: MockAgeEstimator,
        mock_transform_classifier: MockTransformClassifier,
    ) -> None:
        """No image provided → no face detection → score 0."""
        predictor = PrivacyPredictor(mock_face_detector, mock_age_estimator, mock_transform_classifier)
        result = predictor.predict("Remove their clothes")
        assert result.privacy_score == 0.0
        assert result.faces_detected == 0

    def test_safe_background_change(
        self, mock_face_detector: MockFaceDetector, mock_age_estimator: MockAgeEstimator,
        mock_transform_classifier: MockTransformClassifier, dummy_image: np.ndarray,
    ) -> None:
        """Background change on private person → low risk, not a violation."""
        predictor = PrivacyPredictor(mock_face_detector, mock_age_estimator, mock_transform_classifier)
        result = predictor.predict("Change the background to a beach", image=dummy_image)
        # Mock classifier may not detect "background" keyword → score 0.0, which is still safe
        assert result.privacy_score <= 0.5
        assert not result.is_violation  # Should not be a violation

    def test_health_check(
        self, mock_face_detector: MockFaceDetector, mock_age_estimator: MockAgeEstimator,
        mock_transform_classifier: MockTransformClassifier,
    ) -> None:
        """Health check should report status for all components."""
        predictor = PrivacyPredictor(mock_face_detector, mock_age_estimator, mock_transform_classifier)
        health = predictor.health_check()
        assert "face_detector" in health
        assert "age_estimator" in health
        assert "transform_classifier" in health
