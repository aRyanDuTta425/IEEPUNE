"""Decision Fusion — aggregates scores from all modules and determines final action.

Logic:
    final_score = max(jailbreak_score, intent_score, privacy_score)
    action = BLOCK if final_score >= BLOCK_THRESHOLD
           = REVIEW if final_score >= REVIEW_THRESHOLD
           = ALLOW  otherwise
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from shield.config import settings
from shield.models.enums import Action

logger = logging.getLogger(__name__)


@dataclass
class FusionInput:
    """Scores from individual detection modules."""

    jailbreak_score: float = 0.0
    jailbreak_details: Dict[str, Any] = field(default_factory=dict)

    intent_score: float = 0.0
    intent_details: Dict[str, Any] = field(default_factory=dict)

    privacy_score: float = 0.0
    privacy_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Violation:
    """A single violation record."""

    module: str
    score: float
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FusionResult:
    """Final decision after fusing all module scores."""

    action: Action
    final_score: float
    violations: List[Violation]
    scores: Dict[str, float]


def fuse(inputs: FusionInput) -> FusionResult:
    """Fuse scores from all detection modules into a final decision.

    Step 1: Compute final_score = max(jailbreak, intent, privacy)
    Step 2: Determine action based on thresholds
    Step 3: Collect violations from modules exceeding their thresholds

    Args:
        inputs: Scores from jailbreak detector, intent graph, and privacy predictor.

    Returns:
        FusionResult with final action, score, violations, and per-module breakdown.
    """
    jb = inputs.jailbreak_score
    intent = inputs.intent_score
    privacy = inputs.privacy_score

    # Step 1: Final score
    final_score = max(jb, intent, privacy)

    # Step 2: Action
    if final_score >= settings.BLOCK_THRESHOLD:
        action = Action.BLOCK
    elif final_score >= settings.REVIEW_THRESHOLD:
        action = Action.REVIEW
    else:
        action = Action.ALLOW

    # Step 3: Violations
    violations: List[Violation] = []

    if jb >= settings.JAILBREAK_THRESHOLD:
        violations.append(Violation(
            module="jailbreak_detector",
            score=jb,
            reason="High similarity to known jailbreak cluster",
            details=inputs.jailbreak_details,
        ))

    if intent >= settings.REVIEW_THRESHOLD:
        violations.append(Violation(
            module="intent_graph",
            score=intent,
            reason="Conversation escalation pattern detected",
            details=inputs.intent_details,
        ))

    if privacy >= settings.PRIVACY_THRESHOLD:
        identity = inputs.privacy_details.get("identity_type", "unknown")
        transforms = inputs.privacy_details.get("transformations", [])
        transform_str = ", ".join(transforms) if transforms else "unknown"
        violations.append(Violation(
            module="privacy_predictor",
            score=privacy,
            reason=f"Privacy violation risk: {identity} with {transform_str} transformation",
            details=inputs.privacy_details,
        ))

    logger.info(
        "Decision fusion: action=%s score=%.3f violations=%d (jb=%.3f intent=%.3f priv=%.3f)",
        action.value,
        final_score,
        len(violations),
        jb,
        intent,
        privacy,
    )

    return FusionResult(
        action=action,
        final_score=final_score,
        violations=violations,
        scores={
            "jailbreak_score": jb,
            "intent_score": intent,
            "privacy_score": privacy,
        },
    )
