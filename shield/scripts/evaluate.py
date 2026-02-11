#!/usr/bin/env python3
"""Evaluate SHIELD on test datasets.

Runs all multi-turn scenarios and image requests through the analysis pipeline
and compares predictions against expected outcomes.

Usage:
    python scripts/evaluate.py [--dataset multiturn|images|all]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from shield.config import settings
from shield.core.decision_fusion import FusionInput, fuse
from shield.core.embeddings import create_embedding_provider
from shield.core.intent_graph import IntentGraph
from shield.core.jailbreak_detector import JailbreakDetector
from shield.core.privacy_predictor import PrivacyPredictor
from shield.ml.mocks import MockAgeEstimator, MockFaceDetector, MockTransformClassifier


def evaluate_multiturn(detector: JailbreakDetector, provider) -> dict:
    """Evaluate on multi-turn dataset."""
    path = Path(settings.MULTITURN_DATASET_PATH)
    if not path.exists():
        print(f"  ⚠ Dataset not found: {path}")
        return {"total": 0, "correct": 0}

    with open(path) as f:
        scenarios = json.load(f)

    correct = 0
    total = len(scenarios)
    results = []

    for scenario in scenarios:
        graph = IntentGraph(provider)
        turns = scenario["conversation"]

        jb_sims = [detector.detect(t["text"]).jailbreak_score for t in turns]
        ig_result = graph.analyze(turns, jailbreak_similarities=jb_sims)

        latest = turns[-1]
        jb_result = detector.detect(latest["text"])

        fusion = fuse(FusionInput(
            jailbreak_score=jb_result.jailbreak_score,
            intent_score=ig_result.intent_score,
        ))

        predicted = fusion.action.value
        expected = scenario["expected_action"]
        match = predicted == expected

        if match:
            correct += 1

        results.append({
            "id": scenario["id"],
            "label": scenario["label"],
            "expected": expected,
            "predicted": predicted,
            "match": "✓" if match else "✗",
            "score": f"{fusion.final_score:.3f}",
        })

    print(f"\n  {'ID':<20} {'Label':<22} {'Expected':<10} {'Predicted':<10} {'Score':<8} {'Match'}")
    print(f"  {'─' * 90}")
    for r in results:
        print(f"  {r['id']:<20} {r['label']:<22} {r['expected']:<10} {r['predicted']:<10} {r['score']:<8} {r['match']}")

    return {"total": total, "correct": correct, "accuracy": correct / total if total else 0}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SHIELD on test datasets")
    parser.add_argument("--dataset", default="all", choices=["multiturn", "images", "all"])
    args = parser.parse_args()

    print(f"SHIELD Evaluation — Mode: {settings.SHIELD_MODE}")
    print("=" * 60)

    provider = create_embedding_provider()
    detector = JailbreakDetector(provider)
    detector.initialize()

    if args.dataset in ("multiturn", "all"):
        print("\n📊 Multi-turn Scenarios")
        mt = evaluate_multiturn(detector, provider)
        print(f"\n  Accuracy: {mt['correct']}/{mt['total']} ({mt.get('accuracy', 0) * 100:.0f}%)")

    print("\n" + "=" * 60)
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
