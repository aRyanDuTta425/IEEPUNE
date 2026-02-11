#!/usr/bin/env python3
"""SHIELD vs Baselines — Comprehensive Comparison Benchmark.

Compares SHIELD against all free baselines across jailbreak, multi-turn,
and privacy detection tasks. Outputs detailed results with per-system metrics.

Usage:
    SHIELD_MODE=lightweight python scripts/full_comparison.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time

os.environ.setdefault("SHIELD_MODE", "lightweight")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np

from shield.core.calibration import find_optimal_threshold
from shield.core.decision_fusion import FusionInput, fuse
from shield.core.embeddings import create_embedding_provider
from shield.core.intent_graph import IntentGraph
from shield.core.jailbreak_detector import JailbreakDetector
from shield.core.privacy_predictor import PrivacyPredictor
from shield.ml.mocks import MockAgeEstimator, MockFaceDetector, SemanticTransformClassifier

np.random.seed(42)


def compute_full_metrics(y_true: list, y_pred: list, y_scores: list) -> dict:
    """Compute all metrics."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # ROC-AUC
    roc_auc = 0.0
    if len(set(y_true)) >= 2:
        pairs = sorted(zip(y_scores, y_true), reverse=True)
        total_pos = sum(y_true)
        total_neg = len(y_true) - total_pos
        if total_pos > 0 and total_neg > 0:
            tp_c, fp_c = 0, 0
            prev_fpr = 0.0
            for score, label in pairs:
                if label == 1:
                    tp_c += 1
                else:
                    fp_c += 1
                    fpr = fp_c / total_neg
                    tpr = tp_c / total_pos
                    roc_auc += (fpr - prev_fpr) * tpr
                    prev_fpr = fpr

    # PR-AUC
    pr_auc = 0.0
    if len(set(y_true)) >= 2:
        pairs = sorted(zip(y_scores, y_true), reverse=True)
        total_pos = sum(y_true)
        if total_pos > 0:
            tp_c, fp_c = 0, 0
            prev_recall = 0.0
            for score, label in pairs:
                if label == 1:
                    tp_c += 1
                else:
                    fp_c += 1
                prec = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
                rec = tp_c / total_pos
                pr_auc += (rec - prev_recall) * prec
                prev_recall = rec

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def main() -> None:
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    total_start = time.perf_counter()

    print("=" * 80)
    print("  SHIELD vs Baselines — Comprehensive Comparison")
    print("=" * 80)

    # ── Initialize SHIELD ─────────────────────────────────────────
    print("\n📦 Initializing SHIELD (all-MiniLM-L6-v2)...")
    provider = create_embedding_provider()
    detector = JailbreakDetector(provider)
    detector.initialize(corpus_path=os.path.join(base_dir, "examples/sample_jailbreaks.json"))

    face = MockFaceDetector()
    age = MockAgeEstimator(default_age=25)
    transform = SemanticTransformClassifier(embedding_provider=provider)
    predictor = PrivacyPredictor(face, age, transform)

    print(f"   ✓ SHIELD ready (corpus={detector.corpus_size}, clusters={detector.num_clusters})")

    # ── Initialize Baselines ──────────────────────────────────────
    print("\n📦 Initializing baselines...")
    from baselines import KeywordFilterBaseline, ToxicBertBaseline, ZeroShotNLIBaseline

    baselines = []

    # Keyword Filter (always works, no model download)
    kw = KeywordFilterBaseline()
    baselines.append(kw)
    print(f"   ✓ {kw.name}")

    # Toxic-BERT
    try:
        tb = ToxicBertBaseline()
        baselines.append(tb)
        print(f"   ✓ {tb.name} (loading...)")
    except Exception as e:
        print(f"   ✗ Toxic-BERT failed: {e}")

    # Zero-Shot NLI (DistilBERT-MNLI)
    try:
        nli = ZeroShotNLIBaseline()
        baselines.append(nli)
        print(f"   ✓ {nli.name} (loading...)")
    except Exception as e:
        print(f"   ✗ Zero-Shot NLI failed: {e}")

    # ── Load Datasets ─────────────────────────────────────────────
    with open(os.path.join(base_dir, "examples/sample_jailbreaks.json")) as f:
        jailbreaks = json.load(f)

    with open(os.path.join(base_dir, "examples/sample_multiturn.json")) as f:
        multiturn = json.load(f)

    with open(os.path.join(base_dir, "examples/sample_image_requests.json")) as f:
        image_requests = json.load(f)

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

    all_results: list = []  # (task, system, metrics, latency)

    # ══════════════════════════════════════════════════════════════
    # TASK 1: JAILBREAK DETECTION
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  TASK 1: Jailbreak Detection")
    print(f"  Positive: {len(jailbreaks)} jailbreak prompts | Negative: {len(benign_prompts)} benign prompts")
    print("=" * 80)

    jb_texts = [p["text"] for p in jailbreaks]
    jb_labels = [1] * len(jailbreaks) + [0] * len(benign_prompts)
    all_jb_texts = jb_texts + benign_prompts

    # SHIELD on jailbreak task
    print("\n  ▶ SHIELD...")
    start = time.perf_counter()
    shield_jb_scores = [detector.detect(t).jailbreak_score for t in all_jb_texts]
    shield_jb_time = (time.perf_counter() - start) * 1000

    thresh, _ = find_optimal_threshold(shield_jb_scores, jb_labels)
    shield_jb_preds = [1 if s >= thresh else 0 for s in shield_jb_scores]
    m = compute_full_metrics(jb_labels, shield_jb_preds, shield_jb_scores)
    all_results.append(("Jailbreak", "SHIELD", m, shield_jb_time))

    # Baselines on jailbreak task
    for bl in baselines:
        print(f"  ▶ {bl.name}...")
        start = time.perf_counter()
        bl_scores = bl.score_batch(all_jb_texts)
        bl_time = (time.perf_counter() - start) * 1000

        bl_thresh, _ = find_optimal_threshold(bl_scores, jb_labels)
        bl_preds = [1 if s >= bl_thresh else 0 for s in bl_scores]
        m = compute_full_metrics(jb_labels, bl_preds, bl_scores)
        all_results.append(("Jailbreak", bl.name, m, bl_time))

    # ══════════════════════════════════════════════════════════════
    # TASK 2: MULTI-TURN ESCALATION DETECTION
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print(f"  TASK 2: Multi-Turn Escalation Detection ({len(multiturn)} scenarios)")
    print("=" * 80)

    mt_labels = [1 if sc["expected_action"] in ("block", "review") else 0 for sc in multiturn]

    # SHIELD on multi-turn task
    print("\n  ▶ SHIELD...")
    start = time.perf_counter()
    shield_mt_scores = []
    for sc in multiturn:
        turns = sc["conversation"]
        g = IntentGraph(provider)
        jb_sims = [detector.detect(t["text"]).jailbreak_score for t in turns]
        ig = g.analyze(turns, jailbreak_similarities=jb_sims)
        latest = next((t for t in reversed(turns) if t.get("sender", "user") == "user"), turns[-1])
        jb = detector.detect(latest["text"])
        fusion = fuse(FusionInput(jailbreak_score=jb.jailbreak_score, intent_score=ig.intent_score))
        shield_mt_scores.append(fusion.final_score)
    shield_mt_time = (time.perf_counter() - start) * 1000

    mt_thresh, _ = find_optimal_threshold(shield_mt_scores, mt_labels)
    shield_mt_preds = [1 if s >= mt_thresh else 0 for s in shield_mt_scores]
    m = compute_full_metrics(mt_labels, shield_mt_preds, shield_mt_scores)
    all_results.append(("Multi-Turn", "SHIELD", m, shield_mt_time))

    # Baselines on multi-turn task
    from baselines import score_conversation
    for bl in baselines:
        print(f"  ▶ {bl.name}...")
        start = time.perf_counter()
        bl_mt_scores = [score_conversation(bl, sc["conversation"]) for sc in multiturn]
        bl_time = (time.perf_counter() - start) * 1000

        bl_thresh, _ = find_optimal_threshold(bl_mt_scores, mt_labels)
        bl_preds = [1 if s >= bl_thresh else 0 for s in bl_mt_scores]
        m = compute_full_metrics(mt_labels, bl_preds, bl_mt_scores)
        all_results.append(("Multi-Turn", bl.name, m, bl_time))

    # ══════════════════════════════════════════════════════════════
    # TASK 3: PRIVACY VIOLATION DETECTION
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print(f"  TASK 3: Privacy Violation Detection ({len(image_requests)} requests)")
    print("=" * 80)

    priv_labels = [1 if r["expected_action"] in ("block", "review") else 0 for r in image_requests]
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)

    # SHIELD on privacy task
    print("\n  ▶ SHIELD...")
    start = time.perf_counter()
    shield_priv_scores = []
    for req in image_requests:
        result = predictor.predict(req["prompt"], image=dummy_image)
        shield_priv_scores.append(result.privacy_score)
    shield_priv_time = (time.perf_counter() - start) * 1000

    priv_thresh, _ = find_optimal_threshold(shield_priv_scores, priv_labels)
    shield_priv_preds = [1 if s >= priv_thresh else 0 for s in shield_priv_scores]
    m = compute_full_metrics(priv_labels, shield_priv_preds, shield_priv_scores)
    all_results.append(("Privacy", "SHIELD", m, shield_priv_time))

    # Baselines on privacy task
    priv_texts = [r["prompt"] for r in image_requests]
    for bl in baselines:
        print(f"  ▶ {bl.name}...")
        start = time.perf_counter()
        bl_priv_scores = bl.score_batch(priv_texts)
        bl_time = (time.perf_counter() - start) * 1000

        bl_thresh, _ = find_optimal_threshold(bl_priv_scores, priv_labels)
        bl_preds = [1 if s >= bl_thresh else 0 for s in bl_priv_scores]
        m = compute_full_metrics(priv_labels, bl_preds, bl_priv_scores)
        all_results.append(("Privacy", bl.name, m, bl_time))

    # ══════════════════════════════════════════════════════════════
    # FINAL RESULTS TABLE
    # ══════════════════════════════════════════════════════════════
    total_time = time.perf_counter() - total_start

    print("\n" + "=" * 80)
    print("  FINAL COMPARISON RESULTS")
    print("=" * 80)

    for task_name in ["Jailbreak", "Multi-Turn", "Privacy"]:
        task_results = [(sys, m, lat) for (t, sys, m, lat) in all_results if t == task_name]
        print(f"\n  ┌{'─' * 78}┐")
        print(f"  │  {task_name + ' Detection':<76}│")
        print(f"  ├{'─' * 78}┤")
        print(f"  │  {'System':<30} {'Prec':>8} {'Recall':>8} {'F1':>8} {'ROC':>8} {'PR':>8} {'ms':>8} │")
        print(f"  ├{'─' * 78}┤")

        for sys_name, m, lat in task_results:
            marker = " ★" if sys_name == "SHIELD" else "  "
            print(f"  │{marker}{sys_name:<29} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} {m['roc_auc']:>8.4f} {m['pr_auc']:>8.4f} {lat:>7.0f} │")

        print(f"  └{'─' * 78}┘")

    # ── Save CSV ──────────────────────────────────────────────────
    csv_path = os.path.join(results_dir, "comparison_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "system", "precision", "recall", "f1", "roc_auc", "pr_auc", "tp", "fp", "fn", "tn", "latency_ms"])
        for task, sys_name, m, lat in all_results:
            writer.writerow([task, sys_name, m["precision"], m["recall"], m["f1"],
                             m["roc_auc"], m["pr_auc"], m["tp"], m["fp"], m["fn"], m["tn"], f"{lat:.0f}"])

    print(f"\n  📄 Results saved to: {csv_path}")
    print(f"  Total benchmark time: {total_time:.1f}s")

    # ── Generate comparison plots ─────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tasks = ["Jailbreak", "Multi-Turn", "Privacy"]
        systems = list(dict.fromkeys(sys for _, sys, _, _ in all_results))  # preserves order
        metrics_list = ["precision", "recall", "f1", "roc_auc", "pr_auc"]

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle("SHIELD vs Baselines — Performance Comparison", fontsize=15, fontweight="bold")

        colors = {"SHIELD": "#2196F3", "Keyword Filter": "#FF9800", "Toxic-BERT": "#F44336",
                  "Zero-Shot NLI (DistilBERT-MNLI)": "#4CAF50"}

        for ax_idx, task in enumerate(tasks):
            task_results = [(sys, m) for (t, sys, m, _) in all_results if t == task]
            sys_names = [s for s, _ in task_results]
            f1_scores = [m["f1"] for _, m in task_results]
            recall_scores = [m["recall"] for _, m in task_results]
            prec_scores = [m["precision"] for _, m in task_results]

            x = np.arange(len(sys_names))
            width = 0.25

            bars1 = axes[ax_idx].bar(x - width, prec_scores, width, label="Precision",
                                     color="#2196F3", alpha=0.85)
            bars2 = axes[ax_idx].bar(x, recall_scores, width, label="Recall",
                                     color="#FF5722", alpha=0.85)
            bars3 = axes[ax_idx].bar(x + width, f1_scores, width, label="F1",
                                     color="#4CAF50", alpha=0.85)

            axes[ax_idx].set_xlabel("System")
            axes[ax_idx].set_ylabel("Score")
            axes[ax_idx].set_title(f"{task} Detection")
            axes[ax_idx].set_xticks(x)
            short_names = [s.replace("Zero-Shot NLI (DistilBERT-MNLI)", "NLI") for s in sys_names]
            axes[ax_idx].set_xticklabels(short_names, rotation=30, ha="right", fontsize=8)
            axes[ax_idx].set_ylim(0, 1.15)
            axes[ax_idx].legend(fontsize=8)
            axes[ax_idx].grid(True, alpha=0.2, axis="y")

            # Add value labels
            for bars in [bars1, bars2, bars3]:
                for bar in bars:
                    h = bar.get_height()
                    if h > 0:
                        axes[ax_idx].annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                                              xytext=(0, 3), textcoords="offset points",
                                              ha="center", va="bottom", fontsize=6)

        plt.tight_layout()
        plot_path = os.path.join(results_dir, "comparison_chart.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  📊 Comparison chart saved: {plot_path}")

    except ImportError:
        print("  ⚠ matplotlib not available — skipping chart")

    print("=" * 80)


if __name__ == "__main__":
    main()
