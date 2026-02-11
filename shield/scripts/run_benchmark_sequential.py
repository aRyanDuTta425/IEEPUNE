#!/usr/bin/env python3
"""Memory-efficient SHIELD benchmark — runs baselines ONE AT A TIME.

On systems with limited RAM, loading BART-MNLI + toxic-bert + DistilBERT
simultaneously causes Bus error / OOM. This script:
  1. Runs SHIELD on all datasets first (keeps SHIELD loaded).
  2. Unloads SHIELD.
  3. Runs each HuggingFace baseline one at a time, collecting & freeing.
  4. Runs keyword filter last (no model).
  5. Merges all results at the end.

Usage:
    python scripts/run_benchmark_sequential.py [--threshold 0.5]
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ── Path setup ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

SEED = 42
np.random.seed(SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark_seq")

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════════
#  Metrics & plotting (same as benchmark_all_free.py)
# ═════════════════════════════════════════════════════════════════════════════════


def compute_metrics(y_true: np.ndarray, y_scores: np.ndarray, threshold: float) -> Dict[str, Any]:
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

    y_pred = (y_scores >= threshold).astype(int)
    if len(np.unique(y_true)) < 2:
        auc = float("nan")
    else:
        try:
            auc = roc_auc_score(y_true, y_scores)
        except ValueError:
            auc = float("nan")

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "auc": round(auc, 4) if not np.isnan(auc) else "N/A",
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def plot_roc_curves(results, title, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    plt.figure(figsize=(10, 7))
    for sys_name, (y_true, y_scores) in results.items():
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{sys_name} (AUC={roc_auc:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Saved ROC: %s", output_path)


def plot_pr_curves(results, title, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score

    plt.figure(figsize=(10, 7))
    for sys_name, (y_true, y_scores) in results.items():
        if len(np.unique(y_true)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        plt.plot(recall, precision, label=f"{sys_name} (AP={ap:.3f})", linewidth=2)
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Saved PR: %s", output_path)


def plot_confusion_matrix(y_true, y_scores, threshold, sys_name, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    y_pred = (y_scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Safe", "Unsafe"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix — {sys_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ═════════════════════════════════════════════════════════════════════════════════
#  Dataset loaders
# ═════════════════════════════════════════════════════════════════════════════════


def load_jailbreak_dataset():
    with open(DATA_DIR / "jailbreak_large.json") as f:
        data = json.load(f)
    texts = [item["text"] for item in data]
    labels = np.array([1 if item["label"] == "jailbreak" else 0 for item in data])
    return texts, labels


def load_multiturn_dataset():
    with open(DATA_DIR / "multiturn_large.json") as f:
        data = json.load(f)
    labels = np.array([0 if item["label"] == "benign" else 1 for item in data])
    return data, labels


def load_privacy_dataset():
    with open(DATA_DIR / "privacy_large.json") as f:
        data = json.load(f)
    texts = [item["prompt"] for item in data]
    labels = np.array([1 if item["label"] in ("unsafe", "critical_minor") else 0 for item in data])
    return texts, labels


# ═════════════════════════════════════════════════════════════════════════════════
#  Unload helpers
# ═════════════════════════════════════════════════════════════════════════════════


def force_gc():
    """Aggressively free memory."""
    gc.collect()
    gc.collect()
    gc.collect()


def unload_baseline(baseline):
    """Release a baseline's pipeline."""
    if hasattr(baseline, "_pipeline") and baseline._pipeline is not None:
        del baseline._pipeline
        baseline._pipeline = None
    del baseline
    force_gc()
    logger.info("Model unloaded, memory freed.")


# ═════════════════════════════════════════════════════════════════════════════════
#  Score conversation helper (from baselines.py)
# ═════════════════════════════════════════════════════════════════════════════════


def score_conversation_baseline(baseline, turns):
    user_texts = [t["text"] for t in turns if t.get("sender") == "user"]
    if not user_texts:
        return 0.0
    individual_scores = baseline.score_batch(user_texts)
    max_individual = max(individual_scores)
    concatenated = " ".join(user_texts)
    concat_score = baseline.score_single(concatenated)
    return max(max_individual, concat_score)


# ═════════════════════════════════════════════════════════════════════════════════
#  SHIELD evaluator
# ═════════════════════════════════════════════════════════════════════════════════


def evaluate_shield(threshold):
    """Run SHIELD on all 3 datasets, return results, then delete SHIELD."""
    logger.info("=" * 70)
    logger.info("Evaluating SHIELD on all datasets...")
    logger.info("=" * 70)

    os.environ["SHIELD_MODE"] = "lightweight"
    from shield.config import ShieldSettings
    settings = ShieldSettings()
    settings.SHIELD_MODE = "lightweight"

    from shield.core.embeddings import create_embedding_provider
    provider = create_embedding_provider()

    from shield.core.jailbreak_detector import JailbreakDetector
    detector = JailbreakDetector(provider)
    corpus_path = ROOT / "examples" / "sample_jailbreaks.json"
    if corpus_path.exists():
        detector.initialize(str(corpus_path))
    else:
        detector.initialize()

    from shield.core.intent_graph import IntentGraph
    from shield.core.decision_fusion import FusionInput, fuse
    from shield.ml.mocks import MockAgeEstimator, MockFaceDetector, SemanticTransformClassifier
    from shield.core.privacy_predictor import PrivacyPredictor

    face = MockFaceDetector()
    age = MockAgeEstimator(default_age=25)
    transform_classifier = SemanticTransformClassifier(embedding_provider=provider)
    privacy_pred = PrivacyPredictor(face, age, transform_classifier)

    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    results = {}

    # ── Jailbreak ─────────────────────────────────────────────────────────────
    texts_jb, y_true_jb = load_jailbreak_dataset()
    logger.info("Jailbreak: %d prompts (%d jb, %d benign)", len(texts_jb), y_true_jb.sum(), (1 - y_true_jb).sum())

    t0 = time.time()
    jb_scores = []
    iterator = tqdm(texts_jb, desc="SHIELD jailbreak") if has_tqdm else texts_jb
    for text in iterator:
        result = detector.detect(text)
        jb_scores.append(result.jailbreak_score)
    jb_time = (time.time() - t0) * 1000 / len(texts_jb)
    jb_scores_arr = np.array(jb_scores)

    m = compute_metrics(y_true_jb, jb_scores_arr, threshold)
    m["latency_ms"] = round(jb_time, 2)
    results["jailbreak"] = {"SHIELD": {"metrics": m, "y_true": y_true_jb, "y_scores": jb_scores_arr}}
    logger.info("  SHIELD jailbreak: P=%.3f R=%.3f F1=%.3f AUC=%s", m["precision"], m["recall"], m["f1"], m["auc"])

    # ── Multi-turn ────────────────────────────────────────────────────────────
    conversations, y_true_mt = load_multiturn_dataset()
    logger.info("Multi-turn: %d conversations (%d unsafe, %d benign)", len(conversations), y_true_mt.sum(), (1 - y_true_mt).sum())

    t0 = time.time()
    mt_scores = []
    iterator = tqdm(conversations, desc="SHIELD multi-turn") if has_tqdm else conversations
    for conv in iterator:
        graph = IntentGraph(provider)
        turns = conv["conversation"]
        jb_sims = [detector.detect(t["text"]).jailbreak_score for t in turns]
        ig_result = graph.analyze(turns, jailbreak_similarities=jb_sims)
        latest_user = next((t for t in reversed(turns) if t.get("sender") == "user"), turns[-1])
        jb_result = detector.detect(latest_user["text"])
        fusion = fuse(FusionInput(jailbreak_score=jb_result.jailbreak_score, intent_score=ig_result.intent_score))
        mt_scores.append(fusion.final_score)
    mt_time = (time.time() - t0) * 1000 / len(conversations)
    mt_scores_arr = np.array(mt_scores)

    m = compute_metrics(y_true_mt, mt_scores_arr, threshold)
    m["latency_ms"] = round(mt_time, 2)
    results["multiturn"] = {"SHIELD": {"metrics": m, "y_true": y_true_mt, "y_scores": mt_scores_arr}}
    logger.info("  SHIELD multi-turn: P=%.3f R=%.3f F1=%.3f AUC=%s", m["precision"], m["recall"], m["f1"], m["auc"])

    # ── Privacy ───────────────────────────────────────────────────────────────
    texts_pr, y_true_pr = load_privacy_dataset()
    logger.info("Privacy: %d prompts (%d unsafe, %d safe)", len(texts_pr), y_true_pr.sum(), (1 - y_true_pr).sum())

    HIGH_RISK = {"clothing_removal", "sexualization", "face_swap"}
    MEDIUM_RISK = {"age_regression", "pose_change"}

    t0 = time.time()
    pr_scores = []
    iterator = tqdm(texts_pr, desc="SHIELD privacy") if has_tqdm else texts_pr
    for text in iterator:
        transforms = transform_classifier.classify(text, image=None)
        transform_score = 0.0
        for t in transforms:
            if t.transform_type in HIGH_RISK:
                transform_score = max(transform_score, t.confidence)
            elif t.transform_type in MEDIUM_RISK:
                transform_score = max(transform_score, t.confidence * 0.6)
        jb = detector.detect(text)
        pr_scores.append(max(transform_score, jb.jailbreak_score))
    pr_time = (time.time() - t0) * 1000 / len(texts_pr)
    pr_scores_arr = np.array(pr_scores)

    m = compute_metrics(y_true_pr, pr_scores_arr, threshold)
    m["latency_ms"] = round(pr_time, 2)
    results["privacy"] = {"SHIELD": {"metrics": m, "y_true": y_true_pr, "y_scores": pr_scores_arr}}
    logger.info("  SHIELD privacy: P=%.3f R=%.3f F1=%.3f AUC=%s", m["precision"], m["recall"], m["f1"], m["auc"])

    # ── Cleanup SHIELD ────────────────────────────────────────────────────────
    del detector, provider, graph, transform_classifier, privacy_pred, face, age
    force_gc()
    logger.info("SHIELD unloaded.")

    return results


# ═════════════════════════════════════════════════════════════════════════════════
#  Baseline evaluator (one at a time)
# ═════════════════════════════════════════════════════════════════════════════════


def evaluate_baseline(baseline_cls, baseline_kwargs, threshold, all_results):
    """Evaluate one baseline on all datasets, then unload."""
    baseline = baseline_cls(**baseline_kwargs)
    bname = baseline.name
    logger.info("=" * 70)
    logger.info("Evaluating: %s", bname)
    logger.info("=" * 70)

    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    # ── Jailbreak ─────────────────────────────────────────────────────────────
    texts_jb, y_true_jb = load_jailbreak_dataset()
    t0 = time.time()
    scores = []
    batch_size = 32
    if has_tqdm:
        for i in tqdm(range(0, len(texts_jb), batch_size), desc=f"{bname} jailbreak"):
            batch = texts_jb[i:i + batch_size]
            scores.extend(baseline.score_batch(batch))
    else:
        scores = baseline.score_batch(texts_jb)
    elapsed = (time.time() - t0) * 1000 / len(texts_jb)
    scores_arr = np.array(scores)

    m = compute_metrics(y_true_jb, scores_arr, threshold)
    m["latency_ms"] = round(elapsed, 2)
    all_results["jailbreak"][bname] = {"metrics": m, "y_true": y_true_jb, "y_scores": scores_arr}
    logger.info("  %s jailbreak: P=%.3f R=%.3f F1=%.3f AUC=%s", bname, m["precision"], m["recall"], m["f1"], m["auc"])

    # ── Multi-turn ────────────────────────────────────────────────────────────
    conversations, y_true_mt = load_multiturn_dataset()
    t0 = time.time()
    mt_scores = []
    iterator = tqdm(conversations, desc=f"{bname} multi-turn") if has_tqdm else conversations
    for conv in iterator:
        score = score_conversation_baseline(baseline, conv["conversation"])
        mt_scores.append(score)
    elapsed = (time.time() - t0) * 1000 / len(conversations)
    mt_scores_arr = np.array(mt_scores)

    m = compute_metrics(y_true_mt, mt_scores_arr, threshold)
    m["latency_ms"] = round(elapsed, 2)
    all_results["multiturn"][bname] = {"metrics": m, "y_true": y_true_mt, "y_scores": mt_scores_arr}
    logger.info("  %s multi-turn: P=%.3f R=%.3f F1=%.3f AUC=%s", bname, m["precision"], m["recall"], m["f1"], m["auc"])

    # ── Privacy ───────────────────────────────────────────────────────────────
    texts_pr, y_true_pr = load_privacy_dataset()
    t0 = time.time()
    scores = []
    if has_tqdm:
        for i in tqdm(range(0, len(texts_pr), batch_size), desc=f"{bname} privacy"):
            batch = texts_pr[i:i + batch_size]
            scores.extend(baseline.score_batch(batch))
    else:
        scores = baseline.score_batch(texts_pr)
    elapsed = (time.time() - t0) * 1000 / len(texts_pr)
    scores_arr = np.array(scores)

    m = compute_metrics(y_true_pr, scores_arr, threshold)
    m["latency_ms"] = round(elapsed, 2)
    all_results["privacy"][bname] = {"metrics": m, "y_true": y_true_pr, "y_scores": scores_arr}
    logger.info("  %s privacy: P=%.3f R=%.3f F1=%.3f AUC=%s", bname, m["precision"], m["recall"], m["f1"], m["auc"])

    # ── Unload ────────────────────────────────────────────────────────────────
    unload_baseline(baseline)
    return all_results


# ═════════════════════════════════════════════════════════════════════════════════
#  Output helpers
# ═════════════════════════════════════════════════════════════════════════════════


def save_csv(dataset_name, dataset_results):
    path = RESULTS_DIR / f"{dataset_name}_results.csv"
    fieldnames = ["system", "precision", "recall", "f1", "auc", "latency_ms", "tp", "fp", "fn", "tn"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sys_name, data in dataset_results.items():
            row = {"system": sys_name}
            row.update(data["metrics"])
            writer.writerow(row)
    logger.info("Saved: %s", path)


def print_table(title, dataset_results):
    print(f"\n{'=' * 95}")
    print(f"  {title}")
    print(f"{'=' * 95}")
    header = f"  {'System':<35} {'Precision':>9} {'Recall':>8} {'F1':>8} {'AUC':>8} {'Latency(ms)':>12} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6}"
    print(header)
    print(f"  {'-' * 91}")
    for sys_name, data in dataset_results.items():
        m = data["metrics"]
        auc_str = f"{m['auc']:.4f}" if isinstance(m["auc"], float) else str(m["auc"])
        print(f"  {sys_name:<35} {m['precision']:>9.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} {auc_str:>8} {m['latency_ms']:>12.2f} {m['tp']:>6} {m['fp']:>6} {m['fn']:>6} {m['tn']:>6}")
    print(f"{'=' * 95}\n")


def generate_plots(all_results, threshold):
    """Generate ROC, PR, and CM plots for all datasets."""
    for dataset_name, dataset_results in all_results.items():
        # Build curve_data
        curve_data = {}
        for sys_name, data in dataset_results.items():
            curve_data[sys_name] = (data["y_true"], data["y_scores"])

        plot_roc_curves(
            curve_data,
            f"{dataset_name.title()} Detection — ROC Curves",
            RESULTS_DIR / f"roc_{dataset_name}.png",
        )
        plot_pr_curves(
            curve_data,
            f"{dataset_name.title()} Detection — PR Curves",
            RESULTS_DIR / f"pr_{dataset_name}.png",
        )
        for sys_name, (yt, ys) in curve_data.items():
            safe_name = sys_name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
            plot_confusion_matrix(
                yt, ys, threshold, sys_name,
                RESULTS_DIR / f"cm_{dataset_name}_{safe_name}.png",
            )


# ═════════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="SHIELD Sequential Benchmark")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--skip-nli", action="store_true", help="Skip Zero-Shot NLI (largest model)")
    args = parser.parse_args()

    print("=" * 95)
    print("  SHIELD BENCHMARK SUITE — Memory-Efficient Sequential Runner")
    print(f"  Threshold: {args.threshold} | Seed: {SEED}")
    print("=" * 95)

    # Step 1: SHIELD on all datasets
    all_results = evaluate_shield(args.threshold)

    # Step 2: Baselines one at a time
    from baselines import ZeroShotNLIBaseline, ToxicBertBaseline, SentimentHeuristicBaseline, KeywordFilterBaseline

    baselines_to_run = []
    if not args.skip_nli:
        baselines_to_run.append((ZeroShotNLIBaseline, {"batch_size": 4}))  # smaller batch for memory
    baselines_to_run.append((ToxicBertBaseline, {"batch_size": 16}))
    baselines_to_run.append((SentimentHeuristicBaseline, {"batch_size": 16}))
    baselines_to_run.append((KeywordFilterBaseline, {}))

    for cls, kwargs in baselines_to_run:
        try:
            all_results = evaluate_baseline(cls, kwargs, args.threshold, all_results)
        except Exception as e:
            logger.error("Failed to evaluate %s: %s", cls.__name__, e)
            import traceback
            traceback.print_exc()
            continue

    # Step 3: Save results
    for dataset_name in ["jailbreak", "multiturn", "privacy"]:
        save_csv(dataset_name, all_results[dataset_name])
        print_table(f"{dataset_name.upper()} RESULTS", all_results[dataset_name])

    # Step 4: Plots
    generate_plots(all_results, args.threshold)

    print("\n" + "=" * 95)
    print("  BENCHMARK COMPLETE")
    print(f"  Results saved to: {RESULTS_DIR}")
    print("=" * 95)


if __name__ == "__main__":
    main()
