#!/usr/bin/env python3
"""Performance benchmarking for SHIELD endpoints.

Usage:
    python scripts/benchmark.py [--mode mock] [--requests 100]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SHIELD performance")
    parser.add_argument("--mode", default="mock", choices=["mock", "lightweight", "full"])
    parser.add_argument("--requests", type=int, default=50, help="Number of requests to send")
    parser.add_argument("--host", default="http://localhost:8000", help="SHIELD server URL")
    args = parser.parse_args()

    try:
        import httpx
    except ImportError:
        print("Error: httpx required. Install with: pip install httpx")
        sys.exit(1)

    client = httpx.Client(base_url=args.host, timeout=30)

    # Check health
    print(f"Benchmarking {args.host} ({args.requests} requests)")
    try:
        health = client.get("/health")
        print(f"Server mode: {health.json().get('mode', 'unknown')}")
    except Exception as e:
        print(f"Error: Cannot connect to server at {args.host}: {e}")
        sys.exit(1)

    # Benchmark /analyze conversation
    payloads = [
        {
            "type": "conversation",
            "conversation": [
                {"sender": "user", "text": f"Test message {i}", "timestamp": "2026-02-10T10:00:00Z"},
            ],
        }
        for i in range(args.requests)
    ]

    times = []
    errors = 0
    for payload in payloads:
        start = time.perf_counter()
        try:
            resp = client.post("/analyze", json=payload)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
            if resp.status_code != 200:
                errors += 1
        except Exception:
            errors += 1

    if times:
        print(f"\n=== /analyze Benchmark ({len(times)} requests) ===")
        print(f"  Mean:    {statistics.mean(times):.1f} ms")
        print(f"  Median:  {statistics.median(times):.1f} ms")
        print(f"  P95:     {sorted(times)[int(len(times) * 0.95)]:.1f} ms")
        print(f"  P99:     {sorted(times)[int(len(times) * 0.99)]:.1f} ms")
        print(f"  Min:     {min(times):.1f} ms")
        print(f"  Max:     {max(times):.1f} ms")
        print(f"  Errors:  {errors}")
        print(f"  QPS:     {len(times) / (sum(times) / 1000):.1f}")
    else:
        print("No successful requests!")

    client.close()


if __name__ == "__main__":
    main()
