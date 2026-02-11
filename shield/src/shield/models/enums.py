"""Enumerations used across SHIELD."""

from __future__ import annotations

from enum import Enum


class Action(str, Enum):
    """Final action decision."""

    BLOCK = "block"
    ALLOW = "allow"
    REVIEW = "review"


class IdentityType(str, Enum):
    """Classification of a detected person."""

    MINOR = "minor"
    PUBLIC_FIGURE = "public_figure"
    PRIVATE_PERSON = "private_person"
    NONE = "none"


class TransformType(str, Enum):
    """Types of image transformation intent."""

    CLOTHING_REMOVAL = "clothing_removal"
    SEXUALIZATION = "sexualization"
    POSE_CHANGE = "pose_change"
    FACE_SWAP = "face_swap"
    AGE_REGRESSION = "age_regression"
    AGE_PROGRESSION = "age_progression"
    BACKGROUND_CHANGE = "background_change"
    FACIAL_MODIFICATION = "facial_modification"


class JailbreakCategory(str, Enum):
    """Jailbreak prompt category."""

    ROLEPLAY = "roleplay"
    HYPOTHETICAL = "hypothetical"
    INSTRUCTION_INJECTION = "instruction_injection"
    MULTI_STEP = "multi_step"
    TOOL_EXPLOIT = "tool_exploit"
    CONTEXT_SMUGGLING = "context_smuggling"


class ConversationLabel(str, Enum):
    """Label for multi-turn conversation scenarios."""

    BENIGN = "benign"
    GRADUAL_ESCALATION = "gradual_escalation"
    SUDDEN_JAILBREAK = "sudden_jailbreak"
