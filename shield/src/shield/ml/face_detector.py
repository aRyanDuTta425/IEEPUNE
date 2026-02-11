"""Real face detection implementation using RetinaFace or facenet-pytorch.

Falls back gracefully if dependencies are not installed.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

from shield.config import settings
from shield.ml.base import BaseFaceDetector, FaceDetection

logger = logging.getLogger(__name__)


class RealFaceDetector(BaseFaceDetector):
    """Face detection using RetinaFace or facenet-pytorch.

    Automatically selects backend based on FACE_DETECTOR_MODEL config.
    """

    def __init__(self) -> None:
        self._model = None
        self._backend = settings.FACE_DETECTOR_MODEL
        self._load_model()

    def _load_model(self) -> None:
        """Load face detection model."""
        if self._backend == "facenet":
            try:
                from facenet_pytorch import MTCNN

                self._model = MTCNN(keep_all=True, device="cpu")
                logger.info("Loaded facenet-pytorch MTCNN face detector")
            except ImportError:
                logger.error("facenet-pytorch not installed")
                raise
        else:
            try:
                from retinaface import RetinaFace as RF

                self._model = RF
                logger.info("Loaded RetinaFace face detector")
            except ImportError:
                logger.error("retinaface not installed")
                raise

    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        """Detect faces in image.

        Args:
            image: RGB image as numpy array.

        Returns:
            List of face detections.
        """
        detections: List[FaceDetection] = []

        try:
            if self._backend == "facenet" and self._model is not None:
                from PIL import Image

                pil_img = Image.fromarray(image)
                boxes, probs = self._model.detect(pil_img)
                if boxes is not None:
                    for box, prob in zip(boxes, probs):
                        x1, y1, x2, y2 = [int(v) for v in box]
                        detections.append(
                            FaceDetection(
                                bbox=(x1, y1, x2 - x1, y2 - y1),
                                confidence=float(prob),
                                embedding=np.zeros(512, dtype=np.float32),
                                is_public_figure=False,
                            )
                        )
            else:
                resp = self._model.detect_faces(image)  # type: ignore[union-attr]
                for key, face_data in resp.items():
                    area = face_data["facial_area"]
                    detections.append(
                        FaceDetection(
                            bbox=(area[0], area[1], area[2] - area[0], area[3] - area[1]),
                            confidence=float(face_data["score"]),
                            embedding=np.zeros(512, dtype=np.float32),
                            is_public_figure=False,
                        )
                    )
        except Exception:
            logger.exception("Face detection failed")

        return detections

    def health_check(self) -> dict:
        return {
            "status": "healthy" if self._model is not None else "unhealthy",
            "model_name": f"real_{self._backend}",
        }
