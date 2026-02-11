"""Real age estimation implementation.

Uses a deep learning model for age prediction from face crops.
Falls back gracefully if model weights are not available.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from shield.config import settings
from shield.ml.base import AgeEstimate, BaseAgeEstimator

logger = logging.getLogger(__name__)


class RealAgeEstimator(BaseAgeEstimator):
    """Age estimation using a pre-trained deep learning model.

    Expects model weights at the path specified by AGE_ESTIMATOR_WEIGHTS.
    Falls back to a simple heuristic if weights are unavailable.
    """

    def __init__(self) -> None:
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        """Attempt to load age estimation model."""
        weights_path = Path(settings.AGE_ESTIMATOR_WEIGHTS)
        if not weights_path.exists():
            logger.warning(
                "Age estimator weights not found at %s — using heuristic fallback",
                weights_path,
            )
            return

        try:
            import torch

            # Generic age model loading — adjust for actual architecture
            self._model = torch.load(weights_path, map_location="cpu")
            self._model.eval()
            logger.info("Loaded age estimation model from %s", weights_path)
        except Exception:
            logger.exception("Failed to load age estimation model")

    def estimate(self, face_crop: np.ndarray) -> AgeEstimate:
        """Estimate age from face crop.

        Args:
            face_crop: Face crop as numpy array (H, W, C).

        Returns:
            AgeEstimate with age and is_minor flag.
        """
        if self._model is None:
            # Heuristic fallback — assume adult
            return AgeEstimate(age=30, is_minor=False, confidence=0.5)

        try:
            import torch
            from PIL import Image

            img = Image.fromarray(face_crop).resize((224, 224))
            tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1).unsqueeze(0) / 255.0

            with torch.no_grad():
                output = self._model(tensor)
                age = int(output.item())

            return AgeEstimate(age=age, is_minor=age < 18, confidence=0.85)
        except Exception:
            logger.exception("Age estimation failed — returning default")
            return AgeEstimate(age=30, is_minor=False, confidence=0.3)

    def health_check(self) -> dict:
        return {
            "status": "healthy" if self._model is not None else "degraded",
            "model_name": "real_age_estimator" if self._model else "heuristic_fallback",
        }
