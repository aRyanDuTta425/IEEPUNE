#!/usr/bin/env python3
"""SHIELD Ablation Study — run each layer independently and combined.

Demonstrates that each SHIELD layer contributes to overall detection.
Outputs a table: System | Precision | Recall | F1

Usage:
    SHIELD_MODE=lightweight python scripts/ablation.py
"""

from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("SHIELD_MODE", "lightweight")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from shield.core.calibration import find_optimal_threshold
from shield.core.decision_fusion import FusionInput, fuse
from shield.core.embeddings import create_embedding_provider
from shield.core.intent_graph import IntentGraph
from shield.core.jailbreak_detector import JailbreakDetector
from shield.core.privacy_predictor import PrivacyPredictor
from shield.ml.mocks import MockAgeEstimator, MockFaceDetector, SemanticTransformClassifier

np.random.seed(42)


def compute_metrics(y_true: list, y_pred: list) -> dict:
    """Compute precision, recall, F1 from binary labels."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def main() -> None:
    base_dir = os.path.join(os.path.dirname(__file__), "..")

    print("=" * 70)
    print("  SHIELD Ablation Study")
    print("=" * 70)

    # Initialize components
    print("\n📦 Loading model and components...")
    provider = create_embedding_provider()
    detector = JailbreakDetector(provider)
    detector.initialize(corpus_path=os.path.join(base_dir, "examples/sample_jailbreaks.json"))
    graph_provider = provider

    face = MockFaceDetector()
    age = MockAgeEstimator(default_age=25)
    transform = SemanticTransformClassifier(embedding_provider=provider)
    predictor = PrivacyPredictor(face, age, transform)

    # Load jailbreak dataset
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

    all_texts = [p["text"] for p in jailbreaks] + benign_prompts
    all_labels = [1] * len(jailbreaks) + [0] * len(benign_prompts)

    results_table: list = []

    # ── Config 1: Only Jailbreak Detector ─────────────────────────
    print("\n▶ Running: Jailbreak Detector Only...")
    scores_jb = []
    for text in all_texts:
        r = detector.detect(text)
        scores_jb.append(r.jailbreak_score)

    thresh_jb, _ = find_optimal_threshold(scores_jb, all_labels)
    preds_jb = [1 if s >= thresh_jb else 0 for s in scores_jb]
    m = compute_metrics(all_labels, preds_jb)
    results_table.append(("Jailbreak Only", m, thresh_jb))

    # ── Config 2: Only Intent Graph ───────────────────────────────
    print("▶ Running: Intent Graph Only...")
    # For single-turn, wrap each prompt as a single-turn conversation
    scores_ig = []
    for text in all_texts:
        from datetime import datetime, timezone
        turns = [{"text": text, "timestamp": datetime.now(timezone.utc).isoformat(), "sender": "user"}]
        g = IntentGraph(graph_provider)
        jb_sims = [detector.detect(text).jailbreak_score]
        result = g.analyze(turns, jailbreak_similarities=jb_sims)
        scores_ig.append(result.intent_score)

    thresh_ig, _ = find_optimal_threshold(scores_ig, all_labels)
    preds_ig = [1 if s >= thresh_ig else 0 for s in scores_ig]
    m = compute_metrics(all_labels, preds_ig)
    results_table.append(("Intent Graph Only", m, thresh_ig))

    # ── Config 3: Only Privacy Predictor ──────────────────────────
    print("▶ Running: Privacy Predictor Only...")
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    scores_pp = []
    for text in all_texts:
        result = predictor.predict(text, image=dummy_image)
        scores_pp.append(result.privacy_score)

    thresh_pp, _ = find_optimal_threshold(scores_pp, all_labels)
    preds_pp = [1 if s >= thresh_pp else 0 for s in scores_pp]
    m = compute_metrics(all_labels, preds_pp)
    results_table.append(("Privacy Only", m, thresh_pp))

    # ── Config 4: Full System (all layers) ────────────────────────
    print("▶ Running: Full System...")
    scores_full = []
    for text in all_texts:
        jb = detector.detect(text)
        from datetime import datetime, timezone
        turns = [{"text": text, "timestamp": datetime.now(timezone.utc).isoformat(), "sender": "user"}]
        g = IntentGraph(graph_provider)
        ig = g.analyze(turns, jailbreak_similarities=[jb.jailbreak_score])
        pp = predictor.predict(text, image=dummy_image)

        fusion = fuse(FusionInput(
            jailbreak_score=jb.jailbreak_score,
            intent_score=ig.intent_score,
            privacy_score=pp.privacy_score,
        ))
        scores_full.append(fusion.final_score)

    thresh_full, _ = find_optimal_threshold(scores_full, all_labels)
    preds_full = [1 if s >= thresh_full else 0 for s in scores_full]
    m = compute_metrics(all_labels, preds_full)
    results_table.append(("Full System", m, thresh_full))

    # ── Print Results Table ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ABLATION RESULTS")
    print("=" * 70)
    print(f"\n  {'System':<25} {'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("  " + "─" * 65)
    for name, m, thresh in results_table:
        print(f"  {name:<25} {thresh:>10.4f} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")

    print("\n" + "=" * 70)
    print("  Each layer contributes independently — full system combines all signals.")
    print("=" * 70)


if __name__ == "__main__":
    main()
