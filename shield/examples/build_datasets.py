#!/usr/bin/env python3
"""Generate all SHIELD example datasets.

Usage:
    python examples/build_datasets.py

Outputs:
    examples/sample_jailbreaks.json
    examples/sample_multiturn.json
    examples/sample_image_requests.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure shield package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from shield.data.builders import (
    build_image_requests_dataset,
    build_jailbreak_corpus,
    build_multiturn_dataset,
)


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Jailbreak corpus
    jailbreaks = build_jailbreak_corpus()
    jailbreak_path = output_dir / "sample_jailbreaks.json"
    with open(jailbreak_path, "w") as f:
        json.dump(jailbreaks, f, indent=2)
    print(f"✓ Generated {len(jailbreaks)} jailbreak prompts → {jailbreak_path}")

    # Multiturn scenarios
    multiturn = build_multiturn_dataset()
    multiturn_path = output_dir / "sample_multiturn.json"
    with open(multiturn_path, "w") as f:
        json.dump(multiturn, f, indent=2)
    print(f"✓ Generated {len(multiturn)} multi-turn scenarios → {multiturn_path}")

    # Image requests
    image_reqs = build_image_requests_dataset()
    image_path = output_dir / "sample_image_requests.json"
    with open(image_path, "w") as f:
        json.dump(image_reqs, f, indent=2)
    print(f"✓ Generated {len(image_reqs)} image requests → {image_path}")

    print("\nAll datasets generated successfully!")


if __name__ == "__main__":
    main()
