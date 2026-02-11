#!/usr/bin/env python3
"""Offline cluster training script.

Embeds the jailbreak corpus, runs HDBSCAN clustering, and saves results.

Usage:
    python scripts/train_clusters.py [--corpus PATH] [--min-cluster-size N]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from shield.config import settings
from shield.core.clustering import save_clusters
from shield.core.embeddings import create_embedding_provider
from shield.core.jailbreak_detector import JailbreakDetector


def main() -> None:
    parser = argparse.ArgumentParser(description="Train jailbreak clusters offline")
    parser.add_argument("--corpus", default=settings.JAILBREAK_CORPUS_PATH, help="Path to jailbreak corpus JSON")
    parser.add_argument("--min-cluster-size", type=int, default=settings.HDBSCAN_MIN_CLUSTER_SIZE)
    parser.add_argument("--output", default=settings.CLUSTER_CACHE_PATH, help="Output path for cluster pickle")
    args = parser.parse_args()

    print(f"Mode: {settings.SHIELD_MODE}")
    print(f"Corpus: {args.corpus}")
    print(f"Min cluster size: {args.min_cluster_size}")

    provider = create_embedding_provider()
    detector = JailbreakDetector(provider)
    detector.initialize(corpus_path=args.corpus)

    print(f"Corpus loaded: {detector.corpus_size} prompts")

    start = time.perf_counter()
    result = detector.refresh_clusters(min_cluster_size=args.min_cluster_size, save_to_disk=True)
    elapsed = time.perf_counter() - start

    print(f"\n✓ Clustering complete in {elapsed:.2f}s")
    print(f"  Clusters found: {result.num_clusters}")
    print(f"  Noise points: {result.noise_count}")
    print(f"  Cluster sizes: {result.cluster_sizes}")
    print(f"  Saved to: {args.output}")


if __name__ == "__main__":
    main()
