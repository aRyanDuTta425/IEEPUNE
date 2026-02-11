"""Threshold calibration utilities for SHIELD.

Provides automatic threshold selection based on F1 optimization
and precision-constrained recall maximization.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def find_optimal_threshold(
    scores: List[float],
    labels: List[int],
    low: float = 0.05,
    high: float = 0.95,
    step: float = 0.01,
) -> Tuple[float, float]:
    """Find the threshold that maximizes F1 score.

    Sweeps thresholds from ``low`` to ``high`` and returns the one
    producing the highest F1.

    Args:
        scores: Predicted scores ∈ [0, 1].
        labels: Ground truth binary labels (1 = positive).
        low: Lower bound of sweep range.
        high: Upper bound of sweep range.
        step: Step size for sweep.

    Returns:
        Tuple of (best_threshold, best_f1).
    """
    scores_arr = np.asarray(scores, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int32)

    best_threshold = 0.5
    best_f1 = 0.0

    threshold = low
    while threshold <= high + 1e-9:
        preds = (scores_arr >= threshold).astype(np.int32)
        tp = int(np.sum((preds == 1) & (labels_arr == 1)))
        fp = int(np.sum((preds == 1) & (labels_arr == 0)))
        fn = int(np.sum((preds == 0) & (labels_arr == 1)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

        threshold += step

    logger.info(
        "Optimal threshold: %.4f (F1=%.4f)", best_threshold, best_f1
    )
    return round(best_threshold, 4), round(best_f1, 4)


def find_high_precision_threshold(
    scores: List[float],
    labels: List[int],
    min_precision: float = 0.95,
    low: float = 0.05,
    high: float = 0.95,
    step: float = 0.01,
) -> Tuple[float, float, float]:
    """Find the threshold that maximizes recall while maintaining precision ≥ ``min_precision``.

    Args:
        scores: Predicted scores ∈ [0, 1].
        labels: Ground truth binary labels (1 = positive).
        min_precision: Minimum precision constraint.
        low: Lower bound of sweep range.
        high: Upper bound of sweep range.
        step: Step size for sweep.

    Returns:
        Tuple of (best_threshold, precision_at_threshold, recall_at_threshold).
    """
    scores_arr = np.asarray(scores, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int32)

    best_threshold = high
    best_recall = 0.0
    best_precision = 1.0

    threshold = low
    while threshold <= high + 1e-9:
        preds = (scores_arr >= threshold).astype(np.int32)
        tp = int(np.sum((preds == 1) & (labels_arr == 1)))
        fp = int(np.sum((preds == 1) & (labels_arr == 0)))
        fn = int(np.sum((preds == 0) & (labels_arr == 1)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision >= min_precision and recall > best_recall:
            best_recall = recall
            best_precision = precision
            best_threshold = threshold

        threshold += step

    logger.info(
        "High-precision threshold: %.4f (P=%.4f, R=%.4f)",
        best_threshold, best_precision, best_recall,
    )
    return round(best_threshold, 4), round(best_precision, 4), round(best_recall, 4)


def compute_metrics_at_threshold(
    scores: List[float],
    labels: List[int],
    threshold: float,
) -> dict:
    """Compute precision, recall, F1 at a given threshold.

    Args:
        scores: Predicted scores.
        labels: Ground truth binary labels.
        threshold: Classification threshold.

    Returns:
        Dict with precision, recall, f1, tp, fp, fn, tn.
    """
    scores_arr = np.asarray(scores, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int32)
    preds = (scores_arr >= threshold).astype(np.int32)

    tp = int(np.sum((preds == 1) & (labels_arr == 1)))
    fp = int(np.sum((preds == 1) & (labels_arr == 0)))
    fn = int(np.sum((preds == 0) & (labels_arr == 1)))
    tn = int(np.sum((preds == 0) & (labels_arr == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "threshold": round(threshold, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
