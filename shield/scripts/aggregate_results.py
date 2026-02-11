#!/usr/bin/env python3
"""Aggregate .npz score files and generate final results + plots.

Usage:
    python scripts/aggregate_results.py [--threshold 0.5]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCORES_DIR = ROOT / "results" / "scores"
RESULTS_DIR = ROOT / "results"

SYSTEM_NAMES = {
    "shield": "SHIELD",
    "nli": "Zero-Shot NLI (BART-MNLI)",
    "toxic_bert": "Toxic-BERT",
    "distilbert": "DistilBERT Sentiment Heuristic",
    "keyword": "Keyword Filter",
}

DATASET_NAMES = ["jailbreak", "multiturn", "privacy"]


def compute_metrics(y_true, y_scores, threshold):
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

    y_pred = (y_scores >= threshold).astype(int)
    if len(np.unique(y_true)) < 2:
        auc = float("nan")
    else:
        try:
            auc = roc_auc_score(y_true, y_scores)
        except ValueError:
            auc = float("nan")

    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
        "auc": round(auc, 4) if not np.isnan(auc) else "N/A",
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    all_data = {}  # {dataset: {sys_name: {metrics, y_true, y_scores}}}

    for ds in DATASET_NAMES:
        all_data[ds] = {}
        for key, display_name in SYSTEM_NAMES.items():
            path = SCORES_DIR / f"{key}_{ds}.npz"
            if not path.exists():
                print(f"  [SKIP] {path.name} not found")
                continue
            data = np.load(path)
            scores = data["scores"]
            labels = data["labels"]
            latency = float(data["latency"])

            m = compute_metrics(labels, scores, args.threshold)
            m["latency_ms"] = round(latency, 2)
            all_data[ds][display_name] = {"metrics": m, "y_true": labels, "y_scores": scores}

    # Save CSVs + print tables
    for ds in DATASET_NAMES:
        if not all_data[ds]:
            continue

        # CSV
        path = RESULTS_DIR / f"{ds}_results.csv"
        fieldnames = ["system", "precision", "recall", "f1", "auc", "latency_ms", "tp", "fp", "fn", "tn"]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for sys_name, d in all_data[ds].items():
                row = {"system": sys_name}
                row.update(d["metrics"])
                writer.writerow(row)
        print(f"Saved: {path}")

        # Table
        print(f"\n{'=' * 110}")
        print(f"  {ds.upper()} RESULTS")
        print(f"{'=' * 110}")
        header = f"  {'System':<35} {'Precision':>9} {'Recall':>8} {'F1':>8} {'AUC':>8} {'Latency(ms)':>12} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6}"
        print(header)
        print(f"  {'-' * 106}")
        for sys_name, d in all_data[ds].items():
            m = d["metrics"]
            auc_str = f"{m['auc']:.4f}" if isinstance(m["auc"], float) else str(m["auc"])
            print(f"  {sys_name:<35} {m['precision']:>9.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} {auc_str:>8} {m['latency_ms']:>12.2f} {m['tp']:>6} {m['fp']:>6} {m['fn']:>6} {m['tn']:>6}")
        print(f"{'=' * 110}\n")

    # Plots
    for ds in DATASET_NAMES:
        if not all_data[ds]:
            continue

        curve_data = {n: (d["y_true"], d["y_scores"]) for n, d in all_data[ds].items()}

        # ROC
        plt.figure(figsize=(10, 7))
        for sys_name, (yt, ys) in curve_data.items():
            if len(np.unique(yt)) < 2:
                continue
            fpr, tpr, _ = roc_curve(yt, ys)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{sys_name} (AUC={roc_auc:.3f})", linewidth=2)
        plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title(f"{ds.title()} Detection — ROC Curves", fontsize=14)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"roc_{ds}.png", dpi=150)
        plt.close()

        # PR
        plt.figure(figsize=(10, 7))
        for sys_name, (yt, ys) in curve_data.items():
            if len(np.unique(yt)) < 2:
                continue
            p, r, _ = precision_recall_curve(yt, ys)
            ap = average_precision_score(yt, ys)
            plt.plot(r, p, label=f"{sys_name} (AP={ap:.3f})", linewidth=2)
        plt.xlabel("Recall", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
        plt.title(f"{ds.title()} Detection — PR Curves", fontsize=14)
        plt.legend(loc="lower left", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"pr_{ds}.png", dpi=150)
        plt.close()

        # Confusion matrices
        for sys_name, (yt, ys) in curve_data.items():
            y_pred = (ys >= args.threshold).astype(int)
            cm = confusion_matrix(yt, y_pred, labels=[0, 1])
            disp = ConfusionMatrixDisplay(cm, display_labels=["Safe", "Unsafe"])
            fig, ax = plt.subplots(figsize=(5, 4))
            disp.plot(ax=ax, cmap="Blues", values_format="d")
            ax.set_title(f"Confusion Matrix — {sys_name}")
            plt.tight_layout()
            safe_name = sys_name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
            plt.savefig(RESULTS_DIR / f"cm_{ds}_{safe_name}.png", dpi=150)
            plt.close()

    print(f"\nAll plots saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
