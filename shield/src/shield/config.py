"""SHIELD configuration via pydantic BaseSettings with environment variable overrides."""

from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings


class ShieldSettings(BaseSettings):
    """All SHIELD configuration, overridable via environment variables.

    Attributes:
        SHIELD_MODE: Deployment mode — mock, lightweight, or full.
    """

    # --- Deployment ---
    SHIELD_MODE: Literal["mock", "lightweight", "full"] = "mock"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    LOG_FORMAT: Literal["json", "text"] = "json"

    # --- Safety Thresholds ---
    JAILBREAK_THRESHOLD: float = 0.75
    PRIVACY_THRESHOLD: float = 0.80
    BLOCK_THRESHOLD: float = 0.85
    REVIEW_THRESHOLD: float = 0.65

    # --- Intent Graph Parameters ---
    # FIX 4: Increased alpha (0.3→0.6) and K (3→5) for stronger propagation
    INTENT_GRAPH_ALPHA: float = 0.6
    INTENT_GRAPH_LAMBDA: float = 0.001
    INTENT_GRAPH_K_NEIGHBORS: int = 5
    INTENT_GRAPH_MAX_ITERATIONS: int = 10
    INTENT_GRAPH_CONVERGENCE: float = 0.001
    INTENT_SCORING_METHOD: Literal["max", "weighted_avg"] = "max"

    # --- Clustering ---
    HDBSCAN_MIN_CLUSTER_SIZE: int = 5
    HDBSCAN_MIN_SAMPLES: int = 3
    EMBEDDING_DIM: int = 384
    CLUSTER_METRIC: str = "euclidean"

    # --- Performance Limits ---
    MAX_CONVERSATION_LENGTH: int = 50
    MAX_JAILBREAK_CORPUS_SIZE: int = 100000
    REQUEST_TIMEOUT_SECONDS: int = 30
    IMAGE_DOWNLOAD_TIMEOUT: int = 5
    MAX_IMAGE_SIZE_MB: int = 10
    MAX_IMAGE_DIMENSION: int = 1024

    # --- File Paths ---
    JAILBREAK_CORPUS_PATH: str = "examples/sample_jailbreaks.json"
    CLUSTER_CACHE_PATH: str = "examples/jailbreak_clusters.pkl"
    MULTITURN_DATASET_PATH: str = "examples/sample_multiturn.json"

    # --- Model Configuration (Full Mode) ---
    FACE_DETECTOR_MODEL: Literal["retinaface", "facenet"] = "retinaface"
    AGE_ESTIMATOR_WEIGHTS: str = "age_net.pth"
    CLIP_MODEL: str = "ViT-B/32"
    SENTENCE_TRANSFORMER_MODEL: str = "all-MiniLM-L6-v2"

    # --- Feature Flags ---
    ENABLE_ASYNC_CLUSTERING: bool = True
    ENABLE_GRAPH_VISUALIZATION: bool = False
    ENABLE_METRICS: bool = True

    model_config = {"env_prefix": "", "env_file": ".env", "extra": "ignore"}


# Singleton instance — import this across the codebase
settings = ShieldSettings()
