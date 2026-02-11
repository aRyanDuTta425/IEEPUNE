#!/usr/bin/env python3
"""Full benchmark suite: SHIELD vs. free offline baselines.

Evaluates SHIELD and all baselines on:
  - Jailbreak detection (data/jailbreak_large.json)
  - Multi-turn escalation (data/multiturn_large.json)
  - Privacy violation (data/privacy_large.json)

All models run 100% offline, CPU-compatible, no API keys.

Outputs:
  results/jailbreak_results.csv
  results/multiturn_results.csv
  results/privacy_results.csv
  results/roc_*.png, results/pr_*.png, results/cm_*.png

Usage:
    python scripts/benchmark_all_free.py [--skip-heavy] [--threshold 0.5]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")

# ── Lazy imports (after path setup) ──────────────────────────────────────────
from baselines import (
    BaselineModel,
    KeywordFilterBaseline,
    SentimentHeuristicBaseline,
    ToxicBertBaseline,
    ZeroShotNLIBaseline,
    score_conversation,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  SHIELD wrapper (uses the real SHIELD pipeline, no core changes)
# ═══════════════════════════════════════════════════════════════════════════════


class SHIELDSystem:
    """Wrapper around the full SHIELD pipeline for benchmark evaluation."""

    def __init__(self) -> None:
        os.environ["SHIELD_MODE"] = "lightweight"
        from shield.config import ShieldSettings

        self._settings = ShieldSettings()
        self._settings.SHIELD_MODE = "lightweight"

        from shield.core.embeddings import create_embedding_provider

        self._provider = create_embedding_provider()

        from shield.core.jailbreak_detector import JailbreakDetector

        self._detector = JailbreakDetector(self._provider)

        # Initialize with the existing sample jailbreaks corpus
        corpus_path = ROOT / "examples" / "sample_jailbreaks.json"
        if corpus_path.exists():
            self._detector.initialize(str(corpus_path))
        else:
            self._detector.initialize()

        from shield.core.intent_graph import IntentGraph

        self._graph_cls = IntentGraph

        from shield.core.decision_fusion import FusionInput, fuse
        from shield.ml.mocks import MockAgeEstimator, MockFaceDetector, SemanticTransformClassifier
        from shield.core.privacy_predictor import PrivacyPredictor

        self._fuse = fuse
        self._FusionInput = FusionInput

        face = MockFaceDetector()
        age = MockAgeEstimator(default_age=25)
        self._transform_classifier = SemanticTransformClassifier(embedding_provider=self._provider)
        self._privacy = PrivacyPredictor(face, age, self._transform_classifier)

    @property
    def name(self) -> str:
        return "SHIELD"

    def score_jailbreak(self, text: str) -> float:
        """Score a single prompt for jailbreak."""
        result = self._detector.detect(text)
        return result.jailbreak_score

    def score_jailbreak_batch(self, texts: List[str]) -> List[float]:
        """Score batch of prompts for jailbreak."""
        return [self.score_jailbreak(t) for t in texts]

    def score_conversation(self, turns: List[Dict[str, Any]]) -> float:
        """Score a multi-turn conversation using the full SHIELD pipeline."""
        graph = self._graph_cls(self._provider)

        jb_sims = [self._detector.detect(t["text"]).jailbreak_score for t in turns]
        ig_result = graph.analyze(turns, jailbreak_similarities=jb_sims)

        latest_user = next(
            (t for t in reversed(turns) if t.get("sender") == "user"), turns[-1]
        )
        jb_result = self._detector.detect(latest_user["text"])

        fusion = self._fuse(
            self._FusionInput(
                jailbreak_score=jb_result.jailbreak_score,
                intent_score=ig_result.intent_score,
            )
        )
        return fusion.final_score

    def score_privacy(self, prompt: str) -> float:
        """Score a privacy prompt (text-only, no image).

        Since SHIELD's privacy predictor requires images (face detection),
        text-only privacy evaluation uses the semantic transform classifier
        to detect harmful intent (clothing_removal, sexualization, etc.)
        and the jailbreak detector as secondary signal.
        """
        # Use the cached transform classifier
        transforms = self._transform_classifier.classify(prompt, image=None)

        # High-risk transform types
        HIGH_RISK = {"clothing_removal", "sexualization", "face_swap"}
        MEDIUM_RISK = {"age_regression", "pose_change"}

        transform_score = 0.0
        for t in transforms:
            if t.transform_type in HIGH_RISK:
                transform_score = max(transform_score, t.confidence)
            elif t.transform_type in MEDIUM_RISK:
                transform_score = max(transform_score, t.confidence * 0.6)

        # Also check jailbreak similarity
        jb = self._detector.detect(prompt)

        return max(transform_score, jb.jailbreak_score)


# ═══════════════════════════════════════════════════════════════════════════════
#  Metrics computation
# ═══════════════════════════════════════════════════════════════════════════════


def compute_metrics(
    y_true: np.ndarray, y_scores: np.ndarray, threshold: float
) -> Dict[str, Any]:
    """Compute precision, recall, F1, AUC, TP, FP, FN, TN."""
    from sklearn.metrics import (
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix,
    )

    y_pred = (y_scores >= threshold).astype(int)

    # Handle edge cases
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
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════════════════


def plot_roc_curves(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str,
    output_path: Path,
) -> None:
    """Plot ROC curves for all systems."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    plt.figure(figsize=(8, 6))
    for sys_name, (y_true, y_scores) in results.items():
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{sys_name} (AUC={roc_auc:.3f})", linewidth=2)

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Saved ROC plot: %s", output_path)


def plot_pr_curves(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str,
    output_path: Path,
) -> None:
    """Plot Precision-Recall curves for all systems."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score

    plt.figure(figsize=(8, 6))
    for sys_name, (y_true, y_scores) in results.items():
        if len(np.unique(y_true)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        plt.plot(recall, precision, label=f"{sys_name} (AP={ap:.3f})", linewidth=2)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Saved PR plot: %s", output_path)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float,
    sys_name: str,
    output_path: Path,
) -> None:
    """Plot confusion matrix for a single system."""
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


# ═══════════════════════════════════════════════════════════════════════════════
#  Dataset loaders
# ═══════════════════════════════════════════════════════════════════════════════

DATA_DIR = ROOT / "data"


def load_jailbreak_dataset() -> Tuple[List[str], np.ndarray]:
    """Load jailbreak dataset. Returns (texts, labels) where 1=jailbreak, 0=benign."""
    path = DATA_DIR / "jailbreak_large.json"
    if not path.exists():
        logger.error("Jailbreak dataset not found at %s — run data/build_large_datasets.py first", path)
        sys.exit(1)
    with open(path) as f:
        data = json.load(f)
    texts = [item["text"] for item in data]
    labels = np.array([1 if item["label"] == "jailbreak" else 0 for item in data])
    return texts, labels


def load_multiturn_dataset() -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Load multi-turn dataset. Returns (conversations, labels) where 1=unsafe."""
    path = DATA_DIR / "multiturn_large.json"
    if not path.exists():
        logger.error("Multi-turn dataset not found at %s — run data/build_large_datasets.py first", path)
        sys.exit(1)
    with open(path) as f:
        data = json.load(f)
    labels = np.array([0 if item["label"] == "benign" else 1 for item in data])
    return data, labels


def load_privacy_dataset() -> Tuple[List[str], np.ndarray]:
    """Load privacy dataset. Returns (prompts, labels) where 1=unsafe."""
    path = DATA_DIR / "privacy_large.json"
    if not path.exists():
        logger.error("Privacy dataset not found at %s — run data/build_large_datasets.py first", path)
        sys.exit(1)
    with open(path) as f:
        data = json.load(f)
    texts = [item["prompt"] for item in data]
    # unsafe + critical_minor = unsafe (1), safe + moderate = safe (0)
    labels = np.array([1 if item["label"] in ("unsafe", "critical_minor") else 0 for item in data])
    return texts, labels


# ═══════════════════════════════════════════════════════════════════════════════
#  Benchmark runner
# ═══════════════════════════════════════════════════════════════════════════════


def benchmark_jailbreak(
    shield: SHIELDSystem,
    baselines: List[BaselineModel],
    threshold: float,
    results_dir: Path,
) -> None:
    """Run jailbreak benchmark."""
    logger.info("=" * 70)
    logger.info("JAILBREAK BENCHMARK")
    logger.info("=" * 70)

    texts, y_true = load_jailbreak_dataset()
    logger.info("Loaded %d prompts (%d jailbreak, %d benign)", len(texts), y_true.sum(), (1 - y_true).sum())

    all_results: Dict[str, Dict[str, Any]] = {}
    curve_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    # ── SHIELD ────────────────────────────────────────────────────────────────
    logger.info("Evaluating SHIELD...")
    try:
        from tqdm import tqdm

        has_tqdm = True
    except ImportError:
        has_tqdm = False

    t0 = time.time()
    shield_scores = []
    if has_tqdm:
        for text in tqdm(texts, desc="SHIELD jailbreak"):
            shield_scores.append(shield.score_jailbreak(text))
    else:
        for i, text in enumerate(texts):
            shield_scores.append(shield.score_jailbreak(text))
            if (i + 1) % 200 == 0:
                logger.info("  SHIELD: %d/%d", i + 1, len(texts))
    shield_time = (time.time() - t0) * 1000 / len(texts)
    shield_scores_arr = np.array(shield_scores)

    metrics = compute_metrics(y_true, shield_scores_arr, threshold)
    metrics["latency_ms"] = round(shield_time, 2)
    all_results["SHIELD"] = metrics
    curve_data["SHIELD"] = (y_true, shield_scores_arr)
    logger.info("  SHIELD: P=%.3f R=%.3f F1=%.3f AUC=%s Latency=%.1fms",
                metrics["precision"], metrics["recall"], metrics["f1"], metrics["auc"], metrics["latency_ms"])

    # ── Baselines ─────────────────────────────────────────────────────────────
    for baseline in baselines:
        logger.info("Evaluating %s...", baseline.name)
        t0 = time.time()
        if has_tqdm:
            scores = []
            for i in tqdm(range(0, len(texts), 32), desc=baseline.name):
                batch = texts[i : i + 32]
                scores.extend(baseline.score_batch(batch))
        else:
            scores = baseline.score_batch(texts)
        elapsed = (time.time() - t0) * 1000 / len(texts)
        scores_arr = np.array(scores)

        m = compute_metrics(y_true, scores_arr, threshold)
        m["latency_ms"] = round(elapsed, 2)
        all_results[baseline.name] = m
        curve_data[baseline.name] = (y_true, scores_arr)
        logger.info("  %s: P=%.3f R=%.3f F1=%.3f AUC=%s Latency=%.1fms",
                    baseline.name, m["precision"], m["recall"], m["f1"], m["auc"], m["latency_ms"])

    # ── Save results ──────────────────────────────────────────────────────────
    _save_csv(all_results, results_dir / "jailbreak_results.csv")
    _print_table("JAILBREAK RESULTS", all_results)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_roc_curves(curve_data, "Jailbreak Detection — ROC Curves", results_dir / "roc_jailbreak.png")
    plot_pr_curves(curve_data, "Jailbreak Detection — PR Curves", results_dir / "pr_jailbreak.png")
    for sys_name, (yt, ys) in curve_data.items():
        safe_name = sys_name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        plot_confusion_matrix(yt, ys, threshold, sys_name, results_dir / f"cm_jailbreak_{safe_name}.png")


def benchmark_multiturn(
    shield: SHIELDSystem,
    baselines: List[BaselineModel],
    threshold: float,
    results_dir: Path,
) -> None:
    """Run multi-turn benchmark."""
    logger.info("=" * 70)
    logger.info("MULTI-TURN ESCALATION BENCHMARK")
    logger.info("=" * 70)

    conversations, y_true = load_multiturn_dataset()
    logger.info("Loaded %d conversations (%d unsafe, %d benign)",
                len(conversations), y_true.sum(), (1 - y_true).sum())

    all_results: Dict[str, Dict[str, Any]] = {}
    curve_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    try:
        from tqdm import tqdm

        has_tqdm = True
    except ImportError:
        has_tqdm = False

    # ── SHIELD ────────────────────────────────────────────────────────────────
    logger.info("Evaluating SHIELD on multi-turn...")
    t0 = time.time()
    shield_scores = []
    iterator = tqdm(conversations, desc="SHIELD multi-turn") if has_tqdm else conversations
    for conv in iterator:
        score = shield.score_conversation(conv["conversation"])
        shield_scores.append(score)
    shield_time = (time.time() - t0) * 1000 / len(conversations)
    shield_scores_arr = np.array(shield_scores)

    metrics = compute_metrics(y_true, shield_scores_arr, threshold)
    metrics["latency_ms"] = round(shield_time, 2)
    all_results["SHIELD"] = metrics
    curve_data["SHIELD"] = (y_true, shield_scores_arr)
    logger.info("  SHIELD: P=%.3f R=%.3f F1=%.3f AUC=%s Latency=%.1fms",
                metrics["precision"], metrics["recall"], metrics["f1"], metrics["auc"], metrics["latency_ms"])

    # ── Baselines ─────────────────────────────────────────────────────────────
    for baseline in baselines:
        logger.info("Evaluating %s on multi-turn...", baseline.name)
        t0 = time.time()
        scores = []
        iterator = tqdm(conversations, desc=baseline.name) if has_tqdm else conversations
        for conv in iterator:
            score = score_conversation(baseline, conv["conversation"])
            scores.append(score)
        elapsed = (time.time() - t0) * 1000 / len(conversations)
        scores_arr = np.array(scores)

        m = compute_metrics(y_true, scores_arr, threshold)
        m["latency_ms"] = round(elapsed, 2)
        all_results[baseline.name] = m
        curve_data[baseline.name] = (y_true, scores_arr)
        logger.info("  %s: P=%.3f R=%.3f F1=%.3f AUC=%s Latency=%.1fms",
                    baseline.name, m["precision"], m["recall"], m["f1"], m["auc"], m["latency_ms"])

    # ── Save results ──────────────────────────────────────────────────────────
    _save_csv(all_results, results_dir / "multiturn_results.csv")
    _print_table("MULTI-TURN RESULTS", all_results)

    plot_roc_curves(curve_data, "Multi-Turn Escalation — ROC Curves", results_dir / "roc_multiturn.png")
    plot_pr_curves(curve_data, "Multi-Turn Escalation — PR Curves", results_dir / "pr_multiturn.png")
    for sys_name, (yt, ys) in curve_data.items():
        safe_name = sys_name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        plot_confusion_matrix(yt, ys, threshold, sys_name, results_dir / f"cm_multiturn_{safe_name}.png")


def benchmark_privacy(
    shield: SHIELDSystem,
    baselines: List[BaselineModel],
    threshold: float,
    results_dir: Path,
) -> None:
    """Run privacy benchmark."""
    logger.info("=" * 70)
    logger.info("PRIVACY VIOLATION BENCHMARK")
    logger.info("=" * 70)

    texts, y_true = load_privacy_dataset()
    logger.info("Loaded %d prompts (%d unsafe, %d safe)", len(texts), y_true.sum(), (1 - y_true).sum())

    all_results: Dict[str, Dict[str, Any]] = {}
    curve_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    try:
        from tqdm import tqdm

        has_tqdm = True
    except ImportError:
        has_tqdm = False

    # ── SHIELD ────────────────────────────────────────────────────────────────
    logger.info("Evaluating SHIELD on privacy...")
    t0 = time.time()
    shield_scores = []
    if has_tqdm:
        for text in tqdm(texts, desc="SHIELD privacy"):
            shield_scores.append(shield.score_privacy(text))
    else:
        for i, text in enumerate(texts):
            shield_scores.append(shield.score_privacy(text))
            if (i + 1) % 100 == 0:
                logger.info("  SHIELD: %d/%d", i + 1, len(texts))
    shield_time = (time.time() - t0) * 1000 / len(texts)
    shield_scores_arr = np.array(shield_scores)

    metrics = compute_metrics(y_true, shield_scores_arr, threshold)
    metrics["latency_ms"] = round(shield_time, 2)
    all_results["SHIELD"] = metrics
    curve_data["SHIELD"] = (y_true, shield_scores_arr)
    logger.info("  SHIELD: P=%.3f R=%.3f F1=%.3f AUC=%s Latency=%.1fms",
                metrics["precision"], metrics["recall"], metrics["f1"], metrics["auc"], metrics["latency_ms"])

    # ── Baselines ─────────────────────────────────────────────────────────────
    for baseline in baselines:
        logger.info("Evaluating %s on privacy...", baseline.name)
        t0 = time.time()
        if has_tqdm:
            scores = []
            for i in tqdm(range(0, len(texts), 32), desc=baseline.name):
                batch = texts[i : i + 32]
                scores.extend(baseline.score_batch(batch))
        else:
            scores = baseline.score_batch(texts)
        elapsed = (time.time() - t0) * 1000 / len(texts)
        scores_arr = np.array(scores)

        m = compute_metrics(y_true, scores_arr, threshold)
        m["latency_ms"] = round(elapsed, 2)
        all_results[baseline.name] = m
        curve_data[baseline.name] = (y_true, scores_arr)
        logger.info("  %s: P=%.3f R=%.3f F1=%.3f AUC=%s Latency=%.1fms",
                    baseline.name, m["precision"], m["recall"], m["f1"], m["auc"], m["latency_ms"])

    # ── Save results ──────────────────────────────────────────────────────────
    _save_csv(all_results, results_dir / "privacy_results.csv")
    _print_table("PRIVACY RESULTS", all_results)

    plot_roc_curves(curve_data, "Privacy Violation — ROC Curves", results_dir / "roc_privacy.png")
    plot_pr_curves(curve_data, "Privacy Violation — PR Curves", results_dir / "pr_privacy.png")
    for sys_name, (yt, ys) in curve_data.items():
        safe_name = sys_name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        plot_confusion_matrix(yt, ys, threshold, sys_name, results_dir / f"cm_privacy_{safe_name}.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════════════════


def _save_csv(results: Dict[str, Dict[str, Any]], path: Path) -> None:
    """Save results dict to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["system", "precision", "recall", "f1", "auc", "latency_ms", "tp", "fp", "fn", "tn"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sys_name, metrics in results.items():
            row = {"system": sys_name}
            row.update(metrics)
            writer.writerow(row)
    logger.info("Saved results: %s", path)


def _print_table(title: str, results: Dict[str, Dict[str, Any]]) -> None:
    """Print formatted results table."""
    print(f"\n{'=' * 90}")
    print(f"  {title}")
    print(f"{'=' * 90}")
    header = f"  {'System':<35} {'Precision':>9} {'Recall':>8} {'F1':>8} {'AUC':>8} {'Latency(ms)':>12}"
    print(header)
    print(f"  {'-' * 83}")
    for sys_name, m in results.items():
        auc_str = f"{m['auc']:.4f}" if isinstance(m["auc"], float) else str(m["auc"])
        print(f"  {sys_name:<35} {m['precision']:>9.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} {auc_str:>8} {m['latency_ms']:>12.2f}")
    print(f"{'=' * 90}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(description="SHIELD Benchmark Suite — Free Offline Baselines")
    parser.add_argument("--skip-heavy", action="store_true",
                        help="Skip HuggingFace models (only run keyword baseline)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold (default: 0.5)")
    parser.add_argument("--dataset", default="all", choices=["jailbreak", "multiturn", "privacy", "all"],
                        help="Which dataset to benchmark")
    args = parser.parse_args()

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("  SHIELD BENCHMARK SUITE — Free Offline Baselines")
    print(f"  Threshold: {args.threshold} | Seed: {SEED} | Skip heavy: {args.skip_heavy}")
    print("=" * 90)

    # ── Check datasets exist ──────────────────────────────────────────────────
    datasets_exist = all(
        (DATA_DIR / f).exists()
        for f in ["jailbreak_large.json", "multiturn_large.json", "privacy_large.json"]
    )
    if not datasets_exist:
        logger.info("Datasets not found — generating now...")
        sys.path.insert(0, str(DATA_DIR))
        from build_large_datasets import main as build_main

        build_main()
        logger.info("Datasets generated.")

    # ── Initialize systems ────────────────────────────────────────────────────
    logger.info("Initializing SHIELD...")
    shield = SHIELDSystem()

    logger.info("Initializing baselines...")
    baselines: List[BaselineModel] = []
    if not args.skip_heavy:
        baselines.append(ZeroShotNLIBaseline())
        baselines.append(ToxicBertBaseline())
        baselines.append(SentimentHeuristicBaseline())
    baselines.append(KeywordFilterBaseline())

    logger.info("Systems ready: SHIELD + %d baselines", len(baselines))

    # ── Run benchmarks ────────────────────────────────────────────────────────
    if args.dataset in ("jailbreak", "all"):
        benchmark_jailbreak(shield, baselines, args.threshold, results_dir)

    if args.dataset in ("multiturn", "all"):
        benchmark_multiturn(shield, baselines, args.threshold, results_dir)

    if args.dataset in ("privacy", "all"):
        benchmark_privacy(shield, baselines, args.threshold, results_dir)

    print("\n" + "=" * 90)
    print("  BENCHMARK COMPLETE")
    print(f"  Results saved to: {results_dir}")
    print("=" * 90)


if __name__ == "__main__":
    main()
