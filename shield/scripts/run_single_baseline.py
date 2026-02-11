#!/usr/bin/env python3
"""Run ONE baseline model in isolation. Saves raw scores to a .npz file.

Usage:
    python scripts/run_single_baseline.py --baseline <name> --output-dir /path/to/tmp
    
Baseline names: shield, toxic-bert, distilbert, keyword, nli
"""

from __future__ import annotations

import argparse
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

SEED = 42
np.random.seed(SEED)

DATA_DIR = ROOT / "data"


def load_jailbreak():
    with open(DATA_DIR / "jailbreak_large.json") as f:
        data = json.load(f)
    return [item["text"] for item in data], np.array([1 if item["label"] == "jailbreak" else 0 for item in data])


def load_multiturn():
    with open(DATA_DIR / "multiturn_large.json") as f:
        data = json.load(f)
    return data, np.array([0 if item["label"] == "benign" else 1 for item in data])


def load_privacy():
    with open(DATA_DIR / "privacy_large.json") as f:
        data = json.load(f)
    return [item["prompt"] for item in data], np.array([1 if item["label"] in ("unsafe", "critical_minor") else 0 for item in data])


def score_conv(baseline, turns):
    user_texts = [t["text"] for t in turns if t.get("sender") == "user"]
    if not user_texts:
        return 0.0
    individual = baseline.score_batch(user_texts)
    concat = baseline.score_single(" ".join(user_texts))
    return max(max(individual), concat)


def run_shield(output_dir):
    """Evaluate SHIELD on all datasets."""
    from tqdm import tqdm
    
    os.environ["SHIELD_MODE"] = "lightweight"
    from shield.config import ShieldSettings
    settings = ShieldSettings()
    settings.SHIELD_MODE = "lightweight"
    
    from shield.core.embeddings import create_embedding_provider
    provider = create_embedding_provider()
    from shield.core.jailbreak_detector import JailbreakDetector
    detector = JailbreakDetector(provider)
    corpus_path = ROOT / "examples" / "sample_jailbreaks.json"
    detector.initialize(str(corpus_path) if corpus_path.exists() else None)
    
    from shield.core.intent_graph import IntentGraph
    from shield.core.decision_fusion import FusionInput, fuse
    from shield.ml.mocks import MockAgeEstimator, MockFaceDetector, SemanticTransformClassifier
    
    transform_classifier = SemanticTransformClassifier(embedding_provider=provider)
    HIGH_RISK = {"clothing_removal", "sexualization", "face_swap"}
    MEDIUM_RISK = {"age_regression", "pose_change"}
    
    # Jailbreak
    texts_jb, y_true_jb = load_jailbreak()
    t0 = time.time()
    jb_scores = [detector.detect(t).jailbreak_score for t in tqdm(texts_jb, desc="SHIELD jailbreak")]
    jb_lat = (time.time() - t0) * 1000 / len(texts_jb)
    np.savez(output_dir / "shield_jailbreak.npz", scores=np.array(jb_scores), labels=y_true_jb, latency=jb_lat)
    
    # Multi-turn
    conversations, y_true_mt = load_multiturn()
    t0 = time.time()
    mt_scores = []
    for conv in tqdm(conversations, desc="SHIELD multi-turn"):
        graph = IntentGraph(provider)
        turns = conv["conversation"]
        jb_sims = [detector.detect(t["text"]).jailbreak_score for t in turns]
        ig_result = graph.analyze(turns, jailbreak_similarities=jb_sims)
        latest = next((t for t in reversed(turns) if t.get("sender") == "user"), turns[-1])
        jb_result = detector.detect(latest["text"])
        fusion = fuse(FusionInput(jailbreak_score=jb_result.jailbreak_score, intent_score=ig_result.intent_score))
        mt_scores.append(fusion.final_score)
    mt_lat = (time.time() - t0) * 1000 / len(conversations)
    np.savez(output_dir / "shield_multiturn.npz", scores=np.array(mt_scores), labels=y_true_mt, latency=mt_lat)
    
    # Privacy
    texts_pr, y_true_pr = load_privacy()
    t0 = time.time()
    pr_scores = []
    for t in tqdm(texts_pr, desc="SHIELD privacy"):
        transforms = transform_classifier.classify(t, image=None)
        tscore = 0.0
        for tr in transforms:
            if tr.transform_type in HIGH_RISK:
                tscore = max(tscore, tr.confidence)
            elif tr.transform_type in MEDIUM_RISK:
                tscore = max(tscore, tr.confidence * 0.6)
        jb = detector.detect(t)
        pr_scores.append(max(tscore, jb.jailbreak_score))
    pr_lat = (time.time() - t0) * 1000 / len(texts_pr)
    np.savez(output_dir / "shield_privacy.npz", scores=np.array(pr_scores), labels=y_true_pr, latency=pr_lat)
    
    print("SHIELD done.")


def run_hf_baseline(baseline_cls, baseline_kwargs, name_key, output_dir):
    """Run a HuggingFace baseline on all datasets."""
    from tqdm import tqdm
    baseline = baseline_cls(**baseline_kwargs)
    bname = name_key
    
    for ds_name, loader in [("jailbreak", load_jailbreak), ("multiturn", load_multiturn), ("privacy", load_privacy)]:
        if ds_name == "multiturn":
            conversations, y_true = loader()
            t0 = time.time()
            scores = [score_conv(baseline, conv["conversation"]) for conv in tqdm(conversations, desc=f"{bname} {ds_name}")]
            lat = (time.time() - t0) * 1000 / len(conversations)
        else:
            texts, y_true = loader()
            t0 = time.time()
            scores = []
            for i in tqdm(range(0, len(texts), 32), desc=f"{bname} {ds_name}"):
                scores.extend(baseline.score_batch(texts[i:i+32]))
            lat = (time.time() - t0) * 1000 / len(texts)
        
        np.savez(output_dir / f"{name_key}_{ds_name}.npz", scores=np.array(scores), labels=y_true, latency=lat)
    
    print(f"{bname} done.")


def run_keyword(output_dir):
    from baselines import KeywordFilterBaseline
    run_hf_baseline(KeywordFilterBaseline, {}, "keyword", output_dir)


def run_toxic_bert(output_dir):
    from baselines import ToxicBertBaseline
    run_hf_baseline(ToxicBertBaseline, {"batch_size": 16}, "toxic_bert", output_dir)


def run_distilbert(output_dir):
    from baselines import SentimentHeuristicBaseline
    run_hf_baseline(SentimentHeuristicBaseline, {"batch_size": 16}, "distilbert", output_dir)


def run_nli(output_dir):
    from baselines import ZeroShotNLIBaseline
    run_hf_baseline(ZeroShotNLIBaseline, {"batch_size": 4}, "nli", output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, choices=["shield", "toxic-bert", "distilbert", "keyword", "nli"])
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "scores"))
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    runners = {
        "shield": run_shield,
        "toxic-bert": run_toxic_bert,
        "distilbert": run_distilbert,
        "keyword": run_keyword,
        "nli": run_nli,
    }
    runners[args.baseline](output_dir)


if __name__ == "__main__":
    main()
