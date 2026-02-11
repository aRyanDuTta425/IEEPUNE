"""Abstract interfaces for ML model components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class FaceDetection:
    """Result from face detection."""

    bbox: Tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    embedding: np.ndarray  # face embedding vector
    is_public_figure: bool = False


@dataclass
class AgeEstimate:
    """Result from age estimation."""

    age: int
    is_minor: bool  # age < 18
    confidence: float


@dataclass
class TransformationResult:
    """A detected transformation intent with confidence."""

    transform_type: str
    confidence: float


class BaseFaceDetector(ABC):
    """Abstract face detection interface."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        """Detect faces in an image.

        Args:
            image: Image as numpy array (H, W, C) in RGB.

        Returns:
            List of face detections.
        """
        ...

    @abstractmethod
    def health_check(self) -> dict:
        """Return health status."""
        ...


class BaseAgeEstimator(ABC):
    """Abstract age estimation interface."""

    @abstractmethod
    def estimate(self, face_crop: np.ndarray) -> AgeEstimate:
        """Estimate age from a cropped face image.

        Args:
            face_crop: Cropped face as numpy array.

        Returns:
            Age estimate.
        """
        ...

    @abstractmethod
    def health_check(self) -> dict:
        """Return health status."""
        ...


class BaseTransformClassifier(ABC):
    """Abstract transformation intent classifier."""

    @abstractmethod
    def classify(self, prompt: str, image: Optional[np.ndarray] = None) -> List[TransformationResult]:
        """Classify the transformation intent from a prompt.

        Args:
            prompt: Text prompt describing desired edit.
            image: Optional image for multimodal classification.

        Returns:
            List of transformation types with confidence scores.
        """
        ...

    @abstractmethod
    def health_check(self) -> dict:
        """Return health status."""
        ...
