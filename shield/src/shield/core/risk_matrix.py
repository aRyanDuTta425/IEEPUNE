"""2D Privacy Risk Matrix — maps (identity_type, transformation_type) → risk level.

Risk Levels:
    0 = low risk      → privacy_score: 0.0
    1 = moderate       → privacy_score: 0.5
    2 = high           → privacy_score: 0.8
    3 = critical       → privacy_score: 0.95
"""

from __future__ import annotations

from typing import Dict

# Exact risk matrix from specification
PRIVACY_RISK_MATRIX: Dict[str, Dict[str, int]] = {
    "public_figure": {
        "clothing_removal": 2,
        "sexualization": 2,
        "pose_change": 1,
        "face_swap": 2,
        "age_regression": 1,
        "age_progression": 1,
        "background_change": 0,
        "facial_modification": 1,
    },
    "private_person": {
        "clothing_removal": 3,
        "sexualization": 3,
        "pose_change": 2,
        "face_swap": 3,
        "age_regression": 2,
        "age_progression": 2,
        "background_change": 1,
        "facial_modification": 2,
    },
    "minor": {
        "clothing_removal": 3,
        "sexualization": 3,
        "pose_change": 3,
        "face_swap": 3,
        "age_regression": 3,
        "age_progression": 3,
        "background_change": 1,
        "facial_modification": 3,
    },
}

# Risk level → numeric score conversion
RISK_LEVEL_SCORES: Dict[int, float] = {
    0: 0.0,
    1: 0.5,
    2: 0.8,
    3: 0.95,
}


def lookup_risk(identity_type: str, transformation_type: str) -> float:
    """Look up the privacy risk score for a given identity and transformation combination.

    Args:
        identity_type: One of ``minor``, ``public_figure``, ``private_person``.
        transformation_type: One of the transformation types defined in ``TransformType`` enum.

    Returns:
        Privacy risk score in [0.0, 0.95].  Returns 0.0 if either key is unknown.
    """
    risk_level = PRIVACY_RISK_MATRIX.get(identity_type, {}).get(transformation_type, 0)
    return RISK_LEVEL_SCORES.get(risk_level, 0.0)
