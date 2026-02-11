"""Data loading utilities for SHIELD datasets."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from shield.config import settings

logger = logging.getLogger(__name__)


def load_jailbreak_corpus(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load jailbreak prompts from JSON file.

    Args:
        path: Override file path.

    Returns:
        List of prompt dicts with ``text``, ``category``, ``severity``.
    """
    filepath = Path(path or settings.JAILBREAK_CORPUS_PATH)
    if not filepath.exists():
        logger.warning("Jailbreak corpus not found at %s", filepath)
        return []
    with open(filepath) as f:
        data = json.load(f)
    logger.info("Loaded %d jailbreak prompts from %s", len(data), filepath)
    return data


def load_multiturn_dataset(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load multi-turn conversation scenarios.

    Args:
        path: Override file path.

    Returns:
        List of scenario dicts.
    """
    filepath = Path(path or settings.MULTITURN_DATASET_PATH)
    if not filepath.exists():
        logger.warning("Multi-turn dataset not found at %s", filepath)
        return []
    with open(filepath) as f:
        data = json.load(f)
    logger.info("Loaded %d multi-turn scenarios from %s", len(data), filepath)
    return data


def load_image_requests(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load image editing request dataset.

    Args:
        path: Override file path.

    Returns:
        List of image request dicts.
    """
    filepath = Path(path or "examples/sample_image_requests.json")
    if not filepath.exists():
        logger.warning("Image requests dataset not found at %s", filepath)
        return []
    with open(filepath) as f:
        data = json.load(f)
    logger.info("Loaded %d image requests from %s", len(data), filepath)
    return data
