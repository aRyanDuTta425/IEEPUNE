"""Shared utilities, metrics helpers, and timing decorators."""

from __future__ import annotations

import hashlib
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

import numpy as np


def timed(func: Callable) -> Callable:
    """Decorator that measures execution time in milliseconds.

    The wrapped function gains a ``_last_duration_ms`` attribute.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        wrapper._last_duration_ms = elapsed  # type: ignore[attr-defined]
        return result

    wrapper._last_duration_ms = 0.0  # type: ignore[attr-defined]
    return wrapper


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in [-1, 1].
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity between rows of two matrices.

    Args:
        a: Matrix of shape (N, D).
        b: Matrix of shape (M, D).

    Returns:
        Similarity matrix of shape (N, M).
    """
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return a_norm @ b_norm.T


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """L2 normalize each row vector.

    Args:
        vectors: Matrix of shape (N, D).

    Returns:
        Normalized matrix.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return vectors / norms


def deterministic_hash_vector(text: str, dim: int = 384, seed: int = 42) -> np.ndarray:
    """Generate a deterministic unit vector from text via hashing.

    Same text always produces the exact same vector.

    Args:
        text: Input text.
        dim: Output dimension.
        seed: Additional seed for variation.

    Returns:
        Unit vector of shape (dim,).
    """
    hash_bytes = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).digest()
    rng = np.random.RandomState(int.from_bytes(hash_bytes[:4], "big"))
    vec = rng.randn(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


# ---- Simple in-memory metrics counter ----


class MetricsCollector:
    """Thread-safe in-memory metrics collector for Prometheus-style output."""

    def __init__(self) -> None:
        self._counters: Dict[str, Dict[str, int]] = {}
        self._histograms: Dict[str, List[float]] = {}
        self._gauges: Dict[str, float] = {}

    def inc_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter by 1."""
        key = self._label_key(name, labels)
        self._counters.setdefault(name, {})
        self._counters[name][key] = self._counters[name].get(key, 0) + 1

    def observe_histogram(self, name: str, value: float) -> None:
        """Record a histogram observation."""
        self._histograms.setdefault(name, [])
        self._histograms[name].append(value)

    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge value."""
        self._gauges[name] = value

    def to_prometheus(self) -> str:
        """Export all metrics in Prometheus text format."""
        lines: List[str] = []

        for name, label_counts in self._counters.items():
            lines.append(f"# HELP {name} Counter metric")
            lines.append(f"# TYPE {name} counter")
            for label_key, count in label_counts.items():
                lines.append(f"{label_key} {count}")

        for name, values in self._histograms.items():
            lines.append(f"# HELP {name} Histogram metric")
            lines.append(f"# TYPE {name} histogram")
            if values:
                buckets = [0.1, 0.5, 1.0, 2.0, 5.0]
                for b in buckets:
                    c = sum(1 for v in values if v <= b)
                    lines.append(f'{name}_bucket{{le="{b}"}} {c}')
                lines.append(f'{name}_bucket{{le="+Inf"}} {len(values)}')
                lines.append(f"{name}_sum {sum(values):.3f}")
                lines.append(f"{name}_count {len(values)}")

        for name, value in self._gauges.items():
            lines.append(f"# HELP {name} Gauge metric")
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")

        return "\n".join(lines) + "\n"

    @staticmethod
    def _label_key(name: str, labels: Optional[Dict[str, str]] = None) -> str:
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"


# Global metrics singleton
metrics = MetricsCollector()
