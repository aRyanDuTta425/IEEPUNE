
import sys
import os
from pathlib import Path

# Setup paths
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from scripts.run_benchmark_sequential import evaluate_shield, save_csv, print_table

def main():
    print("Running SHIELD-only evaluation on large datasets...")
    results, threshold_map = evaluate_shield(0.5)  # Threshold arg is ignored by auto-calib

    # Save results
    for dataset in ["jailbreak", "multiturn", "privacy"]:
        save_csv(dataset, results[dataset])
        print_table(dataset.upper(), results[dataset])

if __name__ == "__main__":
    main()
