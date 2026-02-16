
import sys
import os
import time
import numpy as np
from pathlib import Path

# Setup paths
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_benchmark_sequential import (
    load_multiturn_dataset, load_privacy_dataset, 
    compute_metrics, score_conversation_baseline,
    print_table
)
from scripts.baselines import ZeroShotNLIBaseline, KeywordFilterBaseline

def main():
    print("Recovering NLI and Keyword results for Multi-turn/Privacy...")
    
    # Run Keyword Filter (instant)
    kw = KeywordFilterBaseline()
    
    # Run NLI (slower)
    nli = ZeroShotNLIBaseline(batch_size=8)
    
    # ── Multi-turn ────────────────────────────────────────────────────────────
    conversations, y_true_mt = load_multiturn_dataset()
    print(f"Multi-turn: {len(conversations)}")

    # Keyword
    t0 = time.time()
    kw_scores_mt = []
    for conv in conversations:
        kw_scores_mt.append(score_conversation_baseline(kw, conv["conversation"]))
    t_kw_mt = (time.time() - t0) * 1000 / len(conversations)

    m = compute_metrics(y_true_mt, np.array(kw_scores_mt), 0.5)
    m["latency_ms"] = t_kw_mt
    print_table("KEYWORD MULTI-TURN", {"Keyword": {"metrics": m}})

    # NLI
    t0 = time.time()
    nli_scores_mt = []
    for conv in conversations:
        nli_scores_mt.append(score_conversation_baseline(nli, conv["conversation"]))
    t_nli_mt = (time.time() - t0) * 1000 / len(conversations)

    m = compute_metrics(y_true_mt, np.array(nli_scores_mt), 0.5)
    m["latency_ms"] = t_nli_mt
    print_table("NLI MULTI-TURN", {"NLI": {"metrics": m}})

    # ── Privacy ───────────────────────────────────────────────────────────────
    texts_pr, y_true_pr = load_privacy_dataset()
    print(f"Privacy: {len(texts_pr)}")

    # Keyword
    t0 = time.time()
    kw_scores_pr = kw.score_batch(texts_pr)
    t_kw_pr = (time.time() - t0) * 1000 / len(texts_pr)

    m = compute_metrics(y_true_pr, np.array(kw_scores_pr), 0.5)
    m["latency_ms"] = t_kw_pr
    print_table("KEYWORD PRIVACY", {"Keyword": {"metrics": m}})

    # NLI
    t0 = time.time()
    nli_scores_pr = nli.score_batch(texts_pr)
    t_nli_pr = (time.time() - t0) * 1000 / len(texts_pr)

    m = compute_metrics(y_true_pr, np.array(nli_scores_pr), 0.5)
    m["latency_ms"] = t_nli_pr
    print_table("NLI PRIVACY", {"NLI": {"metrics": m}})

if __name__ == "__main__":
    main()
