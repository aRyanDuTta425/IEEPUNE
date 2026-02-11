"""Real CLIP-based transformation intent classifier.

Uses CLIP zero-shot classification to detect transformation types from text + image.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from shield.config import settings
from shield.ml.base import BaseTransformClassifier, TransformationResult

logger = logging.getLogger(__name__)

# Labels for zero-shot classification
CLIP_LABELS = [
    "removing clothing from a person",
    "sexualization of a person",
    "changing the pose of a person",
    "swapping someone's face",
    "making a person look younger",
    "making a person look older",
    "changing the background",
    "modifying facial features",
]

LABEL_TO_TRANSFORM = {
    "removing clothing from a person": "clothing_removal",
    "sexualization of a person": "sexualization",
    "changing the pose of a person": "pose_change",
    "swapping someone's face": "face_swap",
    "making a person look younger": "age_regression",
    "making a person look older": "age_progression",
    "changing the background": "background_change",
    "modifying facial features": "facial_modification",
}


class RealTransformClassifier(BaseTransformClassifier):
    """CLIP zero-shot transformation intent classifier.

    Uses OpenAI CLIP or open_clip for multimodal understanding.
    """

    def __init__(self) -> None:
        self._model = None
        self._preprocess = None
        self._tokenize = None
        self._load_model()

    def _load_model(self) -> None:
        """Load CLIP model."""
        try:
            import clip
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model, self._preprocess = clip.load(settings.CLIP_MODEL, device=self._device)
            self._tokenize = clip.tokenize
            logger.info("Loaded CLIP model: %s on %s", settings.CLIP_MODEL, self._device)
        except ImportError:
            try:
                import open_clip
                import torch

                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                    "ViT-B-32", pretrained="openai"
                )
                self._tokenize = open_clip.get_tokenizer("ViT-B-32")
                logger.info("Loaded open_clip ViT-B-32")
            except ImportError:
                logger.error("Neither clip nor open_clip installed")
                raise

    def classify(
        self, prompt: str, image: Optional[np.ndarray] = None
    ) -> List[TransformationResult]:
        """Classify transformation intent using CLIP.

        Args:
            prompt: Text prompt.
            image: Optional image array.

        Returns:
            Transformation results with confidence scores.
        """
        if self._model is None:
            return []

        try:
            import torch

            # Encode text labels
            text_tokens = self._tokenize(CLIP_LABELS).to(self._device)
            prompt_tokens = self._tokenize([prompt]).to(self._device)

            with torch.no_grad():
                text_features = self._model.encode_text(text_tokens)
                prompt_features = self._model.encode_text(prompt_tokens)

                # Normalize
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
                prompt_features = prompt_features / prompt_features.norm(dim=1, keepdim=True)

                # Cosine similarity
                similarity = (prompt_features @ text_features.T).squeeze(0)
                probs = similarity.softmax(dim=0).cpu().numpy()

            results: List[TransformationResult] = []
            for i, (label, prob) in enumerate(zip(CLIP_LABELS, probs)):
                if prob > 0.1:  # threshold
                    transform_type = LABEL_TO_TRANSFORM.get(label, "unknown")
                    results.append(
                        TransformationResult(transform_type=transform_type, confidence=float(prob))
                    )

            return sorted(results, key=lambda r: r.confidence, reverse=True)

        except Exception:
            logger.exception("CLIP classification failed")
            return []

    def health_check(self) -> dict:
        return {
            "status": "healthy" if self._model is not None else "unhealthy",
            "model_name": f"clip_{settings.CLIP_MODEL}",
        }
