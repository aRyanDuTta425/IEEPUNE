#!/usr/bin/env python3
"""Export intent graphs to JSON for visualization.

Usage:
    python scripts/visualize_graph.py --input examples/sample_multiturn.json --output graph_output.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

os.environ.setdefault("ENABLE_GRAPH_VISUALIZATION", "true")

from shield.config import settings
from shield.core.embeddings import create_embedding_provider
from shield.core.intent_graph import IntentGraph


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize intent graphs")
    parser.add_argument("--input", default=settings.MULTITURN_DATASET_PATH, help="Multi-turn dataset path")
    parser.add_argument("--output", default="graph_output.json", help="Output JSON file")
    parser.add_argument("--scenario", type=int, default=0, help="Scenario index to visualize")
    args = parser.parse_args()

    with open(args.input) as f:
        scenarios = json.load(f)

    if args.scenario >= len(scenarios):
        print(f"Error: scenario index {args.scenario} out of range (max {len(scenarios) - 1})")
        sys.exit(1)

    scenario = scenarios[args.scenario]
    print(f"Analyzing scenario: {scenario['id']} ({scenario.get('label', 'unknown')})")

    provider = create_embedding_provider()
    graph = IntentGraph(provider)

    result = graph.analyze(scenario["conversation"])

    output = {
        "scenario_id": scenario["id"],
        "label": scenario.get("label", "unknown"),
        "intent_score": result.intent_score,
        "iterations": result.iterations,
        "converged": result.converged,
        "node_risks": result.node_risks,
        "base_risks": result.base_risks,
        "graph_data": result.graph_data,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n✓ Graph exported to {args.output}")
    print(f"  Intent score: {result.intent_score:.4f}")
    print(f"  Iterations: {result.iterations} (converged: {result.converged})")
    print(f"  Node risks: {[f'{r:.3f}' for r in result.node_risks]}")


if __name__ == "__main__":
    main()
