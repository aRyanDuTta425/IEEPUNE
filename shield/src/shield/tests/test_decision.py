"""Tests for Decision Fusion."""

from __future__ import annotations

import pytest

from shield.core.decision_fusion import FusionInput, FusionResult, fuse
from shield.models.enums import Action


class TestDecisionFusion:
    """Tests for the decision fusion logic."""

    def test_max_score_selection(self) -> None:
        """Final score should be the max of all module scores."""
        inputs = FusionInput(jailbreak_score=0.5, intent_score=0.9, privacy_score=0.3)
        result = fuse(inputs)
        assert result.final_score == 0.9

    def test_block_threshold(self) -> None:
        """Score >= BLOCK_THRESHOLD should trigger block."""
        inputs = FusionInput(jailbreak_score=0.9)
        result = fuse(inputs)
        assert result.action == Action.BLOCK

    def test_review_threshold(self) -> None:
        """Score in [REVIEW, BLOCK) should trigger review."""
        inputs = FusionInput(jailbreak_score=0.45)
        result = fuse(inputs)
        assert result.action == Action.REVIEW

    def test_allow_below_review(self) -> None:
        """Score below REVIEW_THRESHOLD should allow."""
        inputs = FusionInput(jailbreak_score=0.1, intent_score=0.2, privacy_score=0.1)
        result = fuse(inputs)
        assert result.action == Action.ALLOW

    def test_multiple_violations(self) -> None:
        """All modules above thresholds should appear in violations."""
        inputs = FusionInput(
            jailbreak_score=0.8,
            intent_score=0.7,
            privacy_score=0.9,
            privacy_details={"identity_type": "private_person", "transformations": ["clothing_removal"]},
        )
        result = fuse(inputs)
        modules = {v.module for v in result.violations}
        assert "jailbreak_detector" in modules
        assert "intent_graph" in modules
        assert "privacy_predictor" in modules

    def test_no_violations_when_low_scores(self) -> None:
        """Low scores should produce empty violations list."""
        inputs = FusionInput(jailbreak_score=0.1, intent_score=0.1, privacy_score=0.1)
        result = fuse(inputs)
        assert len(result.violations) == 0

    def test_exact_block_threshold(self) -> None:
        """Score exactly at BLOCK_THRESHOLD should be block."""
        inputs = FusionInput(jailbreak_score=0.55)
        result = fuse(inputs)
        assert result.action == Action.BLOCK

    def test_exact_review_threshold(self) -> None:
        """Score exactly at REVIEW_THRESHOLD should be review."""
        inputs = FusionInput(intent_score=0.35)
        result = fuse(inputs)
        assert result.action == Action.REVIEW

    def test_scores_dict(self) -> None:
        """Result should contain per-module score breakdown."""
        inputs = FusionInput(jailbreak_score=0.5, intent_score=0.6, privacy_score=0.7)
        result = fuse(inputs)
        assert result.scores["jailbreak_score"] == 0.5
        assert result.scores["intent_score"] == 0.6
        assert result.scores["privacy_score"] == 0.7

    def test_violation_reason_contains_module_info(self) -> None:
        """Violation reasons should be human-readable."""
        inputs = FusionInput(
            jailbreak_score=0.9,
            privacy_score=0.95,
            privacy_details={"identity_type": "minor", "transformations": ["sexualization"]},
        )
        result = fuse(inputs)
        for v in result.violations:
            assert len(v.reason) > 0
            assert v.module in ["jailbreak_detector", "intent_graph", "privacy_predictor"]

    def test_all_zeros_allows(self) -> None:
        """Zero scores should allow."""
        inputs = FusionInput()
        result = fuse(inputs)
        assert result.action == Action.ALLOW
        assert result.final_score == 0.0
