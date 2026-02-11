#!/usr/bin/env python3
"""Free, offline baseline safety classifiers for SHIELD benchmarking.

All baselines:
  - Run 100% offline (no API keys, no internet)
  - Output score ∈ [0, 1]
  - CPU compatible
  - Support batch inference

Baselines:
  1. Zero-Shot NLI (facebook/bart-large-mnli)
  2. Toxicity Model (unitary/toxic-bert)
  3. Sentiment Heuristic (distilbert-base-uncased-finetuned-sst-2-english)
  4. Keyword Filter (regex-based)
"""

from __future__ import annotations

import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class BaselineModel(ABC):
    """Abstract baseline safety classifier."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""
        ...

    @abstractmethod
    def score_batch(self, texts: List[str]) -> List[float]:
        """Score a batch of texts for unsafety.

        Args:
            texts: Input texts.

        Returns:
            List of scores ∈ [0, 1] where 1 = maximally unsafe.
        """
        ...

    def score_single(self, text: str) -> float:
        """Score a single text."""
        return self.score_batch([text])[0]

    def classify_batch(self, texts: List[str], threshold: float = 0.5) -> List[bool]:
        """Classify batch as unsafe (True) or safe (False)."""
        scores = self.score_batch(texts)
        return [s >= threshold for s in scores]


# ═══════════════════════════════════════════════════════════════════════════════
#  Baseline 1 — Zero-Shot NLI Safety Classifier
# ═══════════════════════════════════════════════════════════════════════════════


class ZeroShotNLIBaseline(BaselineModel):
    """Zero-shot classification using facebook/bart-large-mnli.

    Labels: ["safe", "unsafe", "harmful", "privacy violation", "illegal"]
    Returns the max probability across unsafe labels.
    """

    def __init__(self, model_name: str = "facebook/bart-large-mnli", batch_size: int = 8) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._pipeline = None
        self._unsafe_labels = ["unsafe", "harmful", "privacy violation", "illegal"]
        self._all_labels = ["safe", "unsafe", "harmful", "privacy violation", "illegal"]

    @property
    def name(self) -> str:
        return "Zero-Shot NLI (BART-MNLI)"

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        from transformers import pipeline

        logger.info("Loading zero-shot NLI model: %s", self._model_name)
        self._pipeline = pipeline(
            "zero-shot-classification",
            model=self._model_name,
            device=-1,  # CPU
        )
        logger.info("Zero-shot NLI model loaded")

    def score_batch(self, texts: List[str]) -> List[float]:
        self._load()
        scores: List[float] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            results = self._pipeline(batch, self._all_labels, multi_label=True)
            if not isinstance(results, list):
                results = [results]
            for result in results:
                label_scores = dict(zip(result["labels"], result["scores"]))
                unsafe_score = max(label_scores.get(l, 0.0) for l in self._unsafe_labels)
                scores.append(float(unsafe_score))
        return scores


# ═══════════════════════════════════════════════════════════════════════════════
#  Baseline 2 — Toxicity Model (toxic-bert)
# ═══════════════════════════════════════════════════════════════════════════════


class ToxicBertBaseline(BaselineModel):
    """Toxicity detection using unitary/toxic-bert.

    Returns max toxicity score across all toxicity categories.
    """

    def __init__(self, model_name: str = "unitary/toxic-bert", batch_size: int = 32) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._pipeline = None

    @property
    def name(self) -> str:
        return "Toxic-BERT"

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        from transformers import pipeline

        logger.info("Loading toxicity model: %s", self._model_name)
        self._pipeline = pipeline(
            "text-classification",
            model=self._model_name,
            top_k=None,  # return all labels
            device=-1,  # CPU
        )
        logger.info("Toxicity model loaded")

    def score_batch(self, texts: List[str]) -> List[float]:
        self._load()
        scores: List[float] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            # Truncate long texts to avoid tokenizer errors
            batch = [t[:512] for t in batch]
            results = self._pipeline(batch)
            # results is List[List[dict]] with top_k=None
            for result in results:
                if isinstance(result, list):
                    # Get max score across toxic labels (exclude 'non-toxic' type labels)
                    toxic_score = max(
                        (r["score"] for r in result if "toxic" in r["label"].lower()),
                        default=0.0,
                    )
                    scores.append(float(toxic_score))
                else:
                    scores.append(float(result.get("score", 0.0)))
        return scores


# ═══════════════════════════════════════════════════════════════════════════════
#  Baseline 3 — Sentiment Heuristic (DistilBERT SST-2)
# ═══════════════════════════════════════════════════════════════════════════════


class SentimentHeuristicBaseline(BaselineModel):
    """Sentiment-based heuristic safety classifier.

    Uses distilbert-base-uncased-finetuned-sst-2-english.
    Maps: NEGATIVE sentiment → higher unsafety score.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        batch_size: int = 32,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._pipeline = None

    @property
    def name(self) -> str:
        return "DistilBERT Sentiment Heuristic"

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        from transformers import pipeline

        logger.info("Loading sentiment model: %s", self._model_name)
        self._pipeline = pipeline(
            "sentiment-analysis",
            model=self._model_name,
            device=-1,  # CPU
        )
        logger.info("Sentiment model loaded")

    def score_batch(self, texts: List[str]) -> List[float]:
        self._load()
        scores: List[float] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            batch = [t[:512] for t in batch]
            results = self._pipeline(batch)
            for result in results:
                label = result["label"]
                confidence = result["score"]
                # NEGATIVE sentiment → unsafe signal
                if label == "NEGATIVE":
                    scores.append(float(confidence))
                else:
                    # POSITIVE → invert: low unsafety
                    scores.append(float(1.0 - confidence))
        return scores


# ═══════════════════════════════════════════════════════════════════════════════
#  Baseline 4 — Keyword Filter (control)
# ═══════════════════════════════════════════════════════════════════════════════

KEYWORD_PATTERNS = [
    # (pattern, weight) — accumulated for matched patterns
    (r"\bbypass\b", 0.3),
    (r"\bignore\s+(all\s+)?rules\b", 0.4),
    (r"\bignore\s+(all\s+)?previous\s+instructions?\b", 0.5),
    (r"\bnude\b", 0.6),
    (r"\bnaked\b", 0.6),
    (r"\bundress\b", 0.6),
    (r"\bbomb\b", 0.5),
    (r"\bhack\b", 0.3),
    (r"\billegal\b", 0.3),
    (r"\bexploit\b", 0.25),
    (r"\bjailbreak\b", 0.5),
    (r"\bunrestricted\b", 0.4),
    (r"\bno\s+restrictions?\b", 0.4),
    (r"\bdan\s+mode\b", 0.5),
    (r"\boverride\b", 0.3),
    (r"\bremove\s+(all\s+)?cloth(es|ing)\b", 0.6),
    (r"\bstrip\b", 0.3),
    (r"\bweapon\b", 0.4),
    (r"\bpoison\b", 0.4),
    (r"\bkill\b", 0.3),
    (r"\bmurder\b", 0.5),
    (r"\bterror(ist|ism)?\b", 0.5),
    (r"\bdrug\b", 0.25),
    (r"\bviolence\b", 0.3),
    (r"\bexplosi(ve|on)\b", 0.5),
    (r"\bmalware\b", 0.5),
    (r"\bransomware\b", 0.5),
    (r"\bphishing\b", 0.3),
    (r"\bsexuali[zs]e?\b", 0.6),
    (r"\bdeepfake\b", 0.5),
    (r"\bsteal\b", 0.3),
    (r"\bfraud\b", 0.4),
    (r"\bforgery?\b", 0.4),
    (r"\bself[- ]harm\b", 0.5),
    (r"\bsuicid\w*\b", 0.5),
    (r"\btorture\b", 0.5),
    (r"\bchild\s+(porn|exploitation|abuse)\b", 0.9),
    (r"\bminor\b", 0.2),
    (r"\bsystem\s+(override|prompt)\b", 0.4),
    (r"\badmin\s+mode\b", 0.4),
    (r"\bdeveloper\s+mode\b", 0.4),
]


class KeywordFilterBaseline(BaselineModel):
    """Simple keyword/regex safety filter.

    Accumulates scores from matched patterns, capped at 1.0.
    """

    def __init__(self) -> None:
        self._patterns = [(re.compile(pat, re.IGNORECASE), weight) for pat, weight in KEYWORD_PATTERNS]

    @property
    def name(self) -> str:
        return "Keyword Filter"

    def score_batch(self, texts: List[str]) -> List[float]:
        scores: List[float] = []
        for text in texts:
            total = 0.0
            for pattern, weight in self._patterns:
                if pattern.search(text):
                    total += weight
            scores.append(min(1.0, total))
        return scores


# ═══════════════════════════════════════════════════════════════════════════════
#  Multi-turn scoring wrapper
# ═══════════════════════════════════════════════════════════════════════════════


def score_conversation(
    baseline: BaselineModel, turns: List[Dict[str, Any]]
) -> float:
    """Score a multi-turn conversation using a single-turn baseline.

    Strategy: concatenate all user turns and score; also take max of individual
    user turn scores. Return the max of both approaches.

    Args:
        baseline: A baseline model.
        turns: Conversation turns with 'sender' and 'text'.

    Returns:
        Score ∈ [0, 1].
    """
    user_texts = [t["text"] for t in turns if t.get("sender") == "user"]
    if not user_texts:
        return 0.0

    # Score individual turns
    individual_scores = baseline.score_batch(user_texts)
    max_individual = max(individual_scores)

    # Score concatenated conversation
    concatenated = " ".join(user_texts)
    concat_score = baseline.score_single(concatenated)

    return max(max_individual, concat_score)


# ═══════════════════════════════════════════════════════════════════════════════
#  Factory
# ═══════════════════════════════════════════════════════════════════════════════


def get_all_baselines(load_heavy: bool = True) -> List[BaselineModel]:
    """Get all baseline models.

    Args:
        load_heavy: If True, include HuggingFace models (requires downloading).
                   If False, only return keyword filter.

    Returns:
        List of baseline model instances.
    """
    baselines: List[BaselineModel] = []
    if load_heavy:
        baselines.append(ZeroShotNLIBaseline())
        baselines.append(ToxicBertBaseline())
        baselines.append(SentimentHeuristicBaseline())
    baselines.append(KeywordFilterBaseline())
    return baselines
