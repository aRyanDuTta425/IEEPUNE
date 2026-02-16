#!/usr/bin/env python3
"""Run ALL remaining experiments on large datasets:
  1. Ablation study (JB only, IG only, Privacy only, JB+IG, Full)
  2. Threshold sweep on large jailbreak dataset
  3. Keyword Filter baseline on large datasets
  4. Toxic-BERT baseline on large datasets
  
Saves all results to results/ directory.
"""

from __future__ import annotations

import csv
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

os.environ["SHIELD_MODE"] = "lightweight"
SEED = 42
np.random.seed(SEED)

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT / "data"


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_scores, threshold):
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    y_pred = (y_scores >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    # AUC
    auc = 0.0
    if len(set(y_true)) >= 2:
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(y_true, y_scores)
        except ValueError:
            auc = 0.0
    return {
        "precision": round(prec, 4), "recall": round(rec, 4),
        "f1": round(f1, 4), "auc": round(auc, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def find_optimal_threshold(y_true, y_scores, min_precision=0.85):
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.01):
        y_pred = (y_scores >= t).astype(int)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if prec >= min_precision and f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1


# ── Dataset loaders ───────────────────────────────────────────────────────────

def load_jailbreak_large():
    with open(DATA_DIR / "jailbreak_large.json") as f:
        data = json.load(f)
    texts = [d["text"] for d in data]
    labels = [1 if d["label"] == "jailbreak" else 0 for d in data]
    return texts, np.array(labels)

def load_multiturn_large():
    with open(DATA_DIR / "multiturn_large.json") as f:
        data = json.load(f)
    labels = [0 if d["label"] == "benign" else 1 for d in data]
    return data, np.array(labels)

def load_privacy_large():
    with open(DATA_DIR / "privacy_large.json") as f:
        data = json.load(f)
    texts = [d["prompt"] for d in data]
    labels = [1 if d["label"] in ("unsafe", "critical_minor") else 0 for d in data]
    return texts, np.array(labels)


# ══════════════════════════════════════════════════════════════════════════════
#  PART 1: ABLATION STUDY ON LARGE DATASETS
# ══════════════════════════════════════════════════════════════════════════════

def run_ablation():
    print("\n" + "=" * 80)
    print("  ABLATION STUDY — Large Datasets")
    print("=" * 80)

    from shield.core.embeddings import create_embedding_provider
    from shield.core.jailbreak_detector import JailbreakDetector
    from shield.core.intent_graph import IntentGraph
    from shield.core.privacy_predictor import PrivacyPredictor
    from shield.core.decision_fusion import FusionInput, fuse
    from shield.ml.mocks import MockAgeEstimator, MockFaceDetector, SemanticTransformClassifier

    provider = create_embedding_provider()
    detector = JailbreakDetector(provider)
    detector.initialize(str(ROOT / "examples" / "sample_jailbreaks.json"))

    face = MockFaceDetector()
    age = MockAgeEstimator(default_age=25)
    transform = SemanticTransformClassifier(embedding_provider=provider)
    predictor = PrivacyPredictor(face, age, transform)

    # Calibrate
    texts_jb, y_true_jb = load_jailbreak_large()
    benign_texts = [texts_jb[i] for i in range(len(texts_jb)) if y_true_jb[i] == 0]
    detector.calibrate_from_distributions(benign_texts[:50])

    HIGH_RISK = {"clothing_removal", "sexualization", "face_swap"}
    MEDIUM_RISK = {"age_regression", "pose_change"}

    results = []

    # ── Ablation configs ──────────────────────────────────────────────────────
    configs = [
        ("JB Only", True, False, False),
        ("IG Only", False, True, False),
        ("Privacy Only", False, False, True),
        ("JB + IG", True, True, False),
        ("JB + Privacy", True, False, True),
        ("Full System", True, True, True),
    ]

    for config_name, use_jb, use_ig, use_priv in configs:
        print(f"\n  ▶ Config: {config_name}")

        # ── JAILBREAK TASK ────────────────────────────────────────────────────
        print(f"    Jailbreak ({len(texts_jb)} samples)...", end=" ", flush=True)
        t0 = time.time()
        scores = []
        for i, text in enumerate(texts_jb):
            jb_score = detector.detect(text).jailbreak_score if use_jb else 0.0
            ig_score = 0.0
            if use_ig:
                from datetime import datetime, timezone
                turns = [{"text": text, "timestamp": datetime.now(timezone.utc).isoformat(), "sender": "user"}]
                g = IntentGraph(provider)
                jb_sim = [detector.detect(text).jailbreak_score]
                ig_result = g.analyze(turns, jailbreak_similarities=jb_sim)
                ig_score = ig_result.intent_score
            pr_score = 0.0
            if use_priv:
                transforms = transform.classify(text, image=None)
                for t in transforms:
                    if t.transform_type in HIGH_RISK:
                        pr_score = max(pr_score, t.confidence)
                    elif t.transform_type in MEDIUM_RISK:
                        pr_score = max(pr_score, t.confidence * 0.6)

            fusion = fuse(FusionInput(jailbreak_score=jb_score, intent_score=ig_score, privacy_score=pr_score))
            scores.append(fusion.final_score)

        elapsed_jb = time.time() - t0
        scores_arr = np.array(scores)
        thresh, _ = find_optimal_threshold(y_true_jb, scores_arr)
        m_jb = compute_metrics(y_true_jb, scores_arr, thresh)
        m_jb["threshold"] = thresh
        print(f"F1={m_jb['f1']:.4f} ({elapsed_jb:.1f}s)")

        # ── MULTITURN TASK ────────────────────────────────────────────────────
        conversations, y_true_mt = load_multiturn_large()
        print(f"    Multi-turn ({len(conversations)} samples)...", end=" ", flush=True)
        t0 = time.time()
        mt_scores = []
        for conv in conversations:
            turns = conv["conversation"]
            jb_sims = []
            for t in turns:
                jb_sims.append(detector.detect(t["text"]).jailbreak_score if use_jb else 0.0)
            
            ig_score = 0.0
            if use_ig:
                g = IntentGraph(provider)
                ig_result = g.analyze(turns, jailbreak_similarities=jb_sims)
                ig_score = ig_result.intent_score

            latest_user = next((t for t in reversed(turns) if t.get("sender") == "user"), turns[-1])
            jb_score = detector.detect(latest_user["text"]).jailbreak_score if use_jb else 0.0
            pr_score = 0.0
            if use_priv:
                transforms = transform.classify(latest_user["text"], image=None)
                for t in transforms:
                    if t.transform_type in HIGH_RISK:
                        pr_score = max(pr_score, t.confidence)

            fusion = fuse(FusionInput(jailbreak_score=jb_score, intent_score=ig_score, privacy_score=pr_score))
            mt_scores.append(fusion.final_score)

        elapsed_mt = time.time() - t0
        mt_scores_arr = np.array(mt_scores)
        thresh_mt, _ = find_optimal_threshold(y_true_mt, mt_scores_arr)
        m_mt = compute_metrics(y_true_mt, mt_scores_arr, thresh_mt)
        m_mt["threshold"] = thresh_mt
        print(f"F1={m_mt['f1']:.4f} ({elapsed_mt:.1f}s)")

        # ── PRIVACY TASK ──────────────────────────────────────────────────────
        texts_pr, y_true_pr = load_privacy_large()
        print(f"    Privacy ({len(texts_pr)} samples)...", end=" ", flush=True)
        t0 = time.time()
        pr_scores = []
        for text in texts_pr:
            jb_score = detector.detect(text).jailbreak_score if use_jb else 0.0
            ig_score = 0.0
            priv_score = 0.0
            if use_priv:
                transforms = transform.classify(text, image=None)
                for t in transforms:
                    if t.transform_type in HIGH_RISK:
                        priv_score = max(priv_score, t.confidence)
                    elif t.transform_type in MEDIUM_RISK:
                        priv_score = max(priv_score, t.confidence * 0.6)

            fusion = fuse(FusionInput(jailbreak_score=jb_score, intent_score=ig_score, privacy_score=priv_score))
            pr_scores.append(max(fusion.final_score, priv_score))

        elapsed_pr = time.time() - t0
        pr_scores_arr = np.array(pr_scores)
        thresh_pr, _ = find_optimal_threshold(y_true_pr, pr_scores_arr)
        m_pr = compute_metrics(y_true_pr, pr_scores_arr, thresh_pr)
        m_pr["threshold"] = thresh_pr
        print(f"F1={m_pr['f1']:.4f} ({elapsed_pr:.1f}s)")

        results.append({
            "config": config_name,
            "jailbreak": m_jb,
            "multiturn": m_mt,
            "privacy": m_pr,
        })

    # ── Save ablation results ─────────────────────────────────────────────────
    csv_path = RESULTS_DIR / "ablation_large.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "task", "precision", "recall", "f1", "auc", "threshold", "tp", "fp", "fn", "tn"])
        for r in results:
            for task in ["jailbreak", "multiturn", "privacy"]:
                m = r[task]
                writer.writerow([r["config"], task, m["precision"], m["recall"], m["f1"], m["auc"], m.get("threshold", ""), m["tp"], m["fp"], m["fn"], m["tn"]])
    print(f"\n  Saved: {csv_path}")

    # ── Print ablation table ──────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("  ABLATION RESULTS SUMMARY (Large Datasets)")
    print("=" * 100)
    for task in ["jailbreak", "multiturn", "privacy"]:
        print(f"\n  {task.upper()} DETECTION:")
        print(f"  {'Config':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
        print(f"  {'─' * 60}")
        for r in results:
            m = r[task]
            print(f"  {r['config']:<20} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['auc']:>10.4f}")
    print("=" * 100)

    # Cleanup
    del detector, provider, transform, predictor, face, age
    gc.collect()

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  PART 2: THRESHOLD SWEEP ON LARGE DATASET
# ══════════════════════════════════════════════════════════════════════════════

def run_threshold_sweep():
    print("\n" + "=" * 80)
    print("  THRESHOLD SWEEP — Large Jailbreak Dataset (2548 samples)")
    print("=" * 80)

    from shield.core.embeddings import create_embedding_provider
    from shield.core.jailbreak_detector import JailbreakDetector

    provider = create_embedding_provider()
    detector = JailbreakDetector(provider)
    detector.initialize(str(ROOT / "examples" / "sample_jailbreaks.json"))

    texts, y_true = load_jailbreak_large()
    benign_texts = [texts[i] for i in range(len(texts)) if y_true[i] == 0]
    detector.calibrate_from_distributions(benign_texts[:50])

    print("  Scoring all prompts...")
    scores = []
    for text in texts:
        r = detector.detect(text)
        scores.append(r.jailbreak_score)
    y_scores = np.array(scores)

    # Sweep
    thresholds = np.arange(0.05, 0.96, 0.01)
    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        preds = (y_scores >= t).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    best_idx = np.argmax(f1s)
    best_t = thresholds[best_idx]
    best_f1 = f1s[best_idx]

    # High precision point
    hp_thresh, hp_f1 = find_optimal_threshold(y_true, y_scores, min_precision=0.95)

    print(f"\n  Best F1 threshold:        {best_t:.4f} (F1={best_f1:.4f})")
    print(f"  High-precision threshold: {hp_thresh:.4f}")

    # Save sweep CSV
    csv_path = RESULTS_DIR / "threshold_sweep_large.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "precision", "recall", "f1"])
        for i, t in enumerate(thresholds):
            writer.writerow([f"{t:.2f}", f"{precisions[i]:.4f}", f"{recalls[i]:.4f}", f"{f1s[i]:.4f}"])
    print(f"  Saved: {csv_path}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("SHIELD Threshold Sweep — Large Dataset (N=2548)", fontsize=14, fontweight="bold")

        axes[0].plot(thresholds, recalls, "b-", linewidth=2, label="Recall")
        axes[0].axvline(x=best_t, color="r", linestyle="--", alpha=0.7, label=f"Best F1 ({best_t:.2f})")
        axes[0].set_xlabel("Threshold"); axes[0].set_ylabel("Recall")
        axes[0].set_title("Recall vs Threshold"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].plot(thresholds, precisions, "r-", linewidth=2, label="Precision")
        axes[1].axvline(x=best_t, color="r", linestyle="--", alpha=0.7, label=f"Best F1 ({best_t:.2f})")
        axes[1].set_xlabel("Threshold"); axes[1].set_ylabel("Precision")
        axes[1].set_title("Precision vs Threshold"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

        axes[2].plot(thresholds, f1s, "g-", linewidth=2, label="F1")
        axes[2].axvline(x=best_t, color="r", linestyle="--", alpha=0.7, label=f"Best F1 ({best_t:.2f})")
        axes[2].axhline(y=best_f1, color="orange", linestyle=":", alpha=0.5, label=f"Max F1={best_f1:.4f}")
        axes[2].set_xlabel("Threshold"); axes[2].set_ylabel("F1 Score")
        axes[2].set_title("F1 vs Threshold"); axes[2].legend(); axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = RESULTS_DIR / "threshold_sweep_large.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {plot_path}")
    except ImportError:
        print("  matplotlib not available — skipping plots")

    # Print table (sampled)
    print(f"\n  {'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'─' * 40}")
    for i in range(0, len(thresholds), 5):
        marker = " <-- best" if abs(thresholds[i] - best_t) < 0.015 else ""
        print(f"  {thresholds[i]:>10.2f} {precisions[i]:>10.4f} {recalls[i]:>10.4f} {f1s[i]:>10.4f}{marker}")

    del detector, provider
    gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
#  PART 3: BASELINES ON LARGE DATASETS
# ══════════════════════════════════════════════════════════════════════════════

def run_baselines():
    print("\n" + "=" * 80)
    print("  BASELINES ON LARGE DATASETS")
    print("=" * 80)

    from baselines import KeywordFilterBaseline, ToxicBertBaseline

    all_baseline_results = []

    # ── Keyword Filter (instant, no model) ────────────────────────────────────
    print("\n  ▶ Keyword Filter")
    kw = KeywordFilterBaseline()

    texts_jb, y_true_jb = load_jailbreak_large()
    t0 = time.time()
    kw_jb_scores = kw.score_batch(texts_jb)
    kw_jb_time = (time.time() - t0) * 1000 / len(texts_jb)
    kw_jb_arr = np.array(kw_jb_scores)
    thresh_kw_jb, _ = find_optimal_threshold(y_true_jb, kw_jb_arr, min_precision=0.5)
    m_jb = compute_metrics(y_true_jb, kw_jb_arr, thresh_kw_jb)
    m_jb["latency_ms"] = round(kw_jb_time, 2)
    print(f"    Jailbreak: P={m_jb['precision']:.4f} R={m_jb['recall']:.4f} F1={m_jb['f1']:.4f}")

    conversations, y_true_mt = load_multiturn_large()
    t0 = time.time()
    kw_mt_scores = []
    for conv in conversations:
        user_texts = [t["text"] for t in conv["conversation"] if t.get("sender") == "user"]
        if not user_texts:
            kw_mt_scores.append(0.0)
            continue
        indiv = kw.score_batch(user_texts)
        concat_score = kw.score_single(" ".join(user_texts))
        kw_mt_scores.append(max(max(indiv), concat_score))
    kw_mt_time = (time.time() - t0) * 1000 / len(conversations)
    kw_mt_arr = np.array(kw_mt_scores)
    thresh_kw_mt, _ = find_optimal_threshold(y_true_mt, kw_mt_arr, min_precision=0.5)
    m_mt = compute_metrics(y_true_mt, kw_mt_arr, thresh_kw_mt)
    m_mt["latency_ms"] = round(kw_mt_time, 2)
    print(f"    Multi-turn: P={m_mt['precision']:.4f} R={m_mt['recall']:.4f} F1={m_mt['f1']:.4f}")

    texts_pr, y_true_pr = load_privacy_large()
    t0 = time.time()
    kw_pr_scores = kw.score_batch(texts_pr)
    kw_pr_time = (time.time() - t0) * 1000 / len(texts_pr)
    kw_pr_arr = np.array(kw_pr_scores)
    thresh_kw_pr, _ = find_optimal_threshold(y_true_pr, kw_pr_arr, min_precision=0.5)
    m_pr = compute_metrics(y_true_pr, kw_pr_arr, thresh_kw_pr)
    m_pr["latency_ms"] = round(kw_pr_time, 2)
    print(f"    Privacy: P={m_pr['precision']:.4f} R={m_pr['recall']:.4f} F1={m_pr['f1']:.4f}")

    all_baseline_results.append(("Keyword Filter", m_jb, m_mt, m_pr))
    del kw

    # ── Toxic-BERT ────────────────────────────────────────────────────────────
    print("\n  ▶ Toxic-BERT")
    try:
        tb = ToxicBertBaseline(batch_size=16)

        t0 = time.time()
        tb_jb_scores = tb.score_batch(texts_jb)
        tb_jb_time = (time.time() - t0) * 1000 / len(texts_jb)
        tb_jb_arr = np.array(tb_jb_scores)
        thresh_tb_jb, _ = find_optimal_threshold(y_true_jb, tb_jb_arr, min_precision=0.5)
        m_jb_tb = compute_metrics(y_true_jb, tb_jb_arr, thresh_tb_jb)
        m_jb_tb["latency_ms"] = round(tb_jb_time, 2)
        print(f"    Jailbreak: P={m_jb_tb['precision']:.4f} R={m_jb_tb['recall']:.4f} F1={m_jb_tb['f1']:.4f}")

        t0 = time.time()
        tb_mt_scores = []
        for conv in conversations:
            user_texts = [t["text"] for t in conv["conversation"] if t.get("sender") == "user"]
            if not user_texts:
                tb_mt_scores.append(0.0)
                continue
            indiv = tb.score_batch(user_texts)
            concat_score = tb.score_single(" ".join(user_texts))
            tb_mt_scores.append(max(max(indiv), concat_score))
        tb_mt_time = (time.time() - t0) * 1000 / len(conversations)
        tb_mt_arr = np.array(tb_mt_scores)
        thresh_tb_mt, _ = find_optimal_threshold(y_true_mt, tb_mt_arr, min_precision=0.5)
        m_mt_tb = compute_metrics(y_true_mt, tb_mt_arr, thresh_tb_mt)
        m_mt_tb["latency_ms"] = round(tb_mt_time, 2)
        print(f"    Multi-turn: P={m_mt_tb['precision']:.4f} R={m_mt_tb['recall']:.4f} F1={m_mt_tb['f1']:.4f}")

        t0 = time.time()
        tb_pr_scores = tb.score_batch(texts_pr)
        tb_pr_time = (time.time() - t0) * 1000 / len(texts_pr)
        tb_pr_arr = np.array(tb_pr_scores)
        thresh_tb_pr, _ = find_optimal_threshold(y_true_pr, tb_pr_arr, min_precision=0.5)
        m_pr_tb = compute_metrics(y_true_pr, tb_pr_arr, thresh_tb_pr)
        m_pr_tb["latency_ms"] = round(tb_pr_time, 2)
        print(f"    Privacy: P={m_pr_tb['precision']:.4f} R={m_pr_tb['recall']:.4f} F1={m_pr_tb['f1']:.4f}")

        all_baseline_results.append(("Toxic-BERT", m_jb_tb, m_mt_tb, m_pr_tb))

        # Unload
        if hasattr(tb, '_pipeline') and tb._pipeline is not None:
            del tb._pipeline
        del tb
        gc.collect()
    except Exception as e:
        print(f"    Toxic-BERT failed: {e}")

    # ── Save baseline results ─────────────────────────────────────────────────
    csv_path = RESULTS_DIR / "baselines_large.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["system", "task", "precision", "recall", "f1", "auc", "latency_ms", "tp", "fp", "fn", "tn"])
        for sys_name, m_j, m_m, m_p in all_baseline_results:
            for task, m in [("jailbreak", m_j), ("multiturn", m_m), ("privacy", m_p)]:
                writer.writerow([sys_name, task, m["precision"], m["recall"], m["f1"], m["auc"], m.get("latency_ms", ""), m["tp"], m["fp"], m["fn"], m["tn"]])
    print(f"\n  Saved: {csv_path}")

    return all_baseline_results


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("  SHIELD — Complete Experiment Suite (Large Datasets)")
    print("  Jailbreak: 2548 | Multi-turn: 255 | Privacy: 400")
    print("=" * 80)

    t_total = time.time()

    # Part 1: Ablation
    ablation_results = run_ablation()

    # Part 2: Threshold Sweep
    run_threshold_sweep()

    # Part 3: Baselines
    baseline_results = run_baselines()

    # ── Final comprehensive summary ───────────────────────────────────────────
    total_time = time.time() - t_total

    print("\n\n" + "=" * 100)
    print("  COMPLETE RESULTS SUMMARY")
    print("=" * 100)

    # Read SHIELD large results
    print("\n  ── SHIELD RESULTS (Large Dataset, Auto-Calibrated) ──")
    print(f"  {'Task':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
    print(f"  {'─' * 55}")
    for csv_name, task_name in [("jailbreak_results.csv", "Jailbreak"), ("multiturn_results.csv", "Multi-turn"), ("privacy_results.csv", "Privacy")]:
        path = RESULTS_DIR / csv_name
        if path.exists():
            with open(path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["system"] == "SHIELD":
                        print(f"  {task_name:<15} {float(row['precision']):>10.4f} {float(row['recall']):>10.4f} {float(row['f1']):>10.4f} {row['auc']:>10}")

    # Baselines
    print("\n  ── BASELINE COMPARISON (Large Dataset) ──")
    print(f"  {'System':<22} {'Task':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
    print(f"  {'─' * 74}")
    for sys_name, m_j, m_m, m_p in baseline_results:
        for task, m in [("Jailbreak", m_j), ("Multi-turn", m_m), ("Privacy", m_p)]:
            print(f"  {sys_name:<22} {task:<12} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['auc']:>10.4f}")

    # Ablation
    print("\n  ── ABLATION STUDY (Large Dataset) ──")
    for task in ["jailbreak", "multiturn", "privacy"]:
        print(f"\n  {task.upper()}:")
        print(f"  {'Config':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
        print(f"  {'─' * 60}")
        for r in ablation_results:
            m = r[task]
            print(f"  {r['config']:<20} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['auc']:>10.4f}")

    print(f"\n  Total experiment time: {total_time:.0f}s")
    print("=" * 100)


if __name__ == "__main__":
    main()
