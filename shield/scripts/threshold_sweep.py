#!/usr/bin/env python3
"""SHIELD Threshold Sweep Report.

Sweeps thresholds for the jailbreak detector and plots
Recall, Precision, and F1 vs Threshold.

Saves: results/threshold_analysis.png

Usage:
    SHIELD_MODE=lightweight python scripts/threshold_sweep.py
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("SHIELD_MODE", "lightweight")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from shield.core.calibration import find_optimal_threshold, find_high_precision_threshold
from shield.core.embeddings import create_embedding_provider
from shield.core.jailbreak_detector import JailbreakDetector


def main() -> None:
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print("  SHIELD Threshold Sweep Analysis")
    print("=" * 60)

    # Initialize
    print("\n📦 Loading model...")
    provider = create_embedding_provider()
    detector = JailbreakDetector(provider)
    detector.initialize(corpus_path=os.path.join(base_dir, "examples/sample_jailbreaks.json"))

    # Load data
    with open(os.path.join(base_dir, "examples/sample_jailbreaks.json")) as f:
        jailbreaks = json.load(f)

    benign_prompts = [
        "What is the capital of France?",
        "Can you help me write a Python function?",
        "How does photosynthesis work?",
        "What is the weather like today?",
        "Explain quantum computing to me.",
        "Help me with my math homework.",
        "What are the best restaurants in New York?",
        "Tell me about machine learning basics.",
        "How do I learn to play guitar?",
        "What books would you recommend?",
        "What is the periodic table?",
        "How do airplanes fly?",
        "Tell me about the solar system.",
        "What is artificial intelligence?",
        "How does a computer work?",
    ]

    # Score all prompts
    print("\n🔍 Scoring prompts...")
    y_true: list = []
    y_scores: list = []

    for p in jailbreaks:
        r = detector.detect(p["text"])
        y_true.append(1)
        y_scores.append(r.jailbreak_score)

    for p in benign_prompts:
        r = detector.detect(p)
        y_true.append(0)
        y_scores.append(r.jailbreak_score)

    y_true_arr = np.array(y_true)
    y_scores_arr = np.array(y_scores)

    # Sweep thresholds
    thresholds = np.arange(0.05, 0.96, 0.01)
    precisions = []
    recalls = []
    f1s = []

    for t in thresholds:
        preds = (y_scores_arr >= t).astype(int)
        tp = np.sum((preds == 1) & (y_true_arr == 1))
        fp = np.sum((preds == 1) & (y_true_arr == 0))
        fn = np.sum((preds == 0) & (y_true_arr == 1))

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    # Find optimal points
    best_thresh, best_f1 = find_optimal_threshold(y_scores, y_true)
    hp_thresh, hp_prec, hp_recall = find_high_precision_threshold(y_scores, y_true)

    print(f"\n  Best F1 threshold:           {best_thresh:.4f} (F1={best_f1:.4f})")
    print(f"  High-precision threshold:    {hp_thresh:.4f} (P={hp_prec:.4f}, R={hp_recall:.4f})")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("SHIELD Threshold Sweep Analysis", fontsize=14, fontweight="bold")

        # Recall vs Threshold
        axes[0].plot(thresholds, recalls, "b-", linewidth=2, label="Recall")
        axes[0].axvline(x=best_thresh, color="r", linestyle="--", alpha=0.7, label=f"Best F1 ({best_thresh:.2f})")
        axes[0].axvline(x=hp_thresh, color="g", linestyle="--", alpha=0.7, label=f"High-P ({hp_thresh:.2f})")
        axes[0].set_xlabel("Threshold")
        axes[0].set_ylabel("Recall")
        axes[0].set_title("Recall vs Threshold")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(0.05, 0.95)
        axes[0].set_ylim(-0.05, 1.05)

        # Precision vs Threshold
        axes[1].plot(thresholds, precisions, "r-", linewidth=2, label="Precision")
        axes[1].axvline(x=best_thresh, color="r", linestyle="--", alpha=0.7, label=f"Best F1 ({best_thresh:.2f})")
        axes[1].axvline(x=hp_thresh, color="g", linestyle="--", alpha=0.7, label=f"High-P ({hp_thresh:.2f})")
        axes[1].set_xlabel("Threshold")
        axes[1].set_ylabel("Precision")
        axes[1].set_title("Precision vs Threshold")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(0.05, 0.95)
        axes[1].set_ylim(-0.05, 1.05)

        # F1 vs Threshold
        axes[2].plot(thresholds, f1s, "g-", linewidth=2, label="F1")
        axes[2].axvline(x=best_thresh, color="r", linestyle="--", alpha=0.7, label=f"Best F1 ({best_thresh:.2f})")
        axes[2].axhline(y=best_f1, color="orange", linestyle=":", alpha=0.5, label=f"Max F1={best_f1:.4f}")
        axes[2].set_xlabel("Threshold")
        axes[2].set_ylabel("F1 Score")
        axes[2].set_title("F1 vs Threshold")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xlim(0.05, 0.95)
        axes[2].set_ylim(-0.05, 1.05)

        plt.tight_layout()
        plot_path = os.path.join(results_dir, "threshold_analysis.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  📊 Saved: {plot_path}")

    except ImportError:
        print("\n  ⚠ matplotlib not available — skipping plots")

    # Print sweep table (sampled)
    print(f"\n  {'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("  " + "─" * 40)
    for i in range(0, len(thresholds), 5):
        print(f"  {thresholds[i]:>10.2f} {precisions[i]:>10.4f} {recalls[i]:>10.4f} {f1s[i]:>10.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
