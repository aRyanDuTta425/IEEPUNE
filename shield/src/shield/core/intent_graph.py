"""Adversarial Intent Graph — models conversation as a directed graph with risk propagation.

Edge weight formula:  w = sim * exp(-λ * Δt)
Risk propagation:     new_risk[i] = (1-α)*base_risk[i] + α * Σ(risk[j] * w[j→i]) / Σ(w[j→i])

FIX 3: Strengthened base risk formula using max(jb_sim, kw_score, combo).
FIX 4: Increased graph connectivity with K_NEIGHBORS=5, ALPHA=0.6.

Detects gradual escalation attacks across multi-turn conversations.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from shield.config import settings
from shield.core.embeddings import EmbeddingProvider
from shield.core.utils import cosine_similarity

logger = logging.getLogger(__name__)

# Suspicious keyword patterns with individual risk contributions
# FIX 3: Increased keyword weights for stronger signal
SUSPICIOUS_PATTERNS: List[Tuple[str, float]] = [
    (r"\bignore\s+(all\s+)?previous\b", 0.15),
    (r"\bbypass\b", 0.15),
    (r"\bsecret\b", 0.08),
    (r"\bunrestricted\b", 0.12),
    (r"\bpretend\s+you\s+are\b", 0.12),
    (r"\bjailbreak\b", 0.18),
    (r"\bno\s+restrictions\b", 0.12),
    (r"\boverride\b", 0.12),
    (r"\bforget\s+(all\s+)?rules\b", 0.15),
    (r"\bdan\s+mode\b", 0.15),
    (r"\bsystem\s+prompt\b", 0.10),
    (r"\broleplay\s+as\b", 0.10),
    (r"\bact\s+as\s+if\b", 0.10),
    (r"\bhypothetical(ly)?\b", 0.08),
    (r"\bfor\s+educational\s+purposes\b", 0.08),
    (r"\bexploit\b", 0.10),
    (r"\bhack\b", 0.10),
    (r"\bharmful\b", 0.08),
    (r"\billegal\b", 0.10),
    (r"\bdangerous\b", 0.08),
]


@dataclass
class IntentGraphResult:
    """Output of intent graph analysis."""

    intent_score: float
    node_risks: List[float]
    base_risks: List[float]
    iterations: int
    converged: bool
    graph_data: Optional[Dict[str, Any]] = None


def _compute_keyword_score(text: str) -> float:
    """Compute keyword-based risk contribution from suspicious patterns.

    FIX 3: Expanded pattern set with increased weights.

    Args:
        text: Input text to scan.

    Returns:
        Keyword score (cumulative, uncapped here — capping happens at base_risk level).
    """
    score = 0.0
    lower = text.lower()
    for pattern, weight in SUSPICIOUS_PATTERNS:
        if re.search(pattern, lower):
            score += weight
    return score


def _compute_length_penalty(text: str, threshold: int = 500) -> float:
    """Penalize very long prompts.

    Args:
        text: Input text.
        threshold: Character count threshold.

    Returns:
        0.05 if text exceeds threshold, else 0.0.
    """
    return 0.05 if len(text) > threshold else 0.0


def compute_base_risk(text: str, jailbreak_similarity: float = 0.0) -> float:
    """Compute base risk score for a single conversation turn.

    FIX 3: Replaced weak weighted-sum with max-based formula:
        base_risk = max(jailbreak_sim, keyword_score, combo + length_penalty)
    where combo = jailbreak_sim*0.6 + keyword_score*0.4

    This produces a stronger initial risk signal so the graph
    can propagate risk more effectively.

    Args:
        text: Turn text.
        jailbreak_similarity: Similarity to nearest jailbreak cluster (calibrated).

    Returns:
        Base risk in [0.0, 1.0].
    """
    keyword = _compute_keyword_score(text)
    length = _compute_length_penalty(text)

    # FIX 3: max-based formula for stronger signal
    combo = jailbreak_similarity * 0.6 + keyword * 0.4 + length
    risk = max(jailbreak_similarity, keyword, min(1.0, combo))
    return min(1.0, risk)


class IntentGraph:
    """Adversarial intent graph for multi-turn conversation analysis.

    Constructs a directed graph where:
    - Each conversation turn is a node
    - Edges connect each turn to the previous K turns
    - Edge weight = similarity * exp(-λ * time_delta)
    - Risk propagates through edges with damping factor α

    FIX 4: Increased K_NEIGHBORS to 5 and ALPHA to 0.6 for stronger propagation.
    """

    def __init__(self, embedding_provider: EmbeddingProvider) -> None:
        self._embedder = embedding_provider
        self._graph = nx.DiGraph()

    def analyze(
        self,
        turns: List[Dict[str, Any]],
        jailbreak_similarities: Optional[List[float]] = None,
    ) -> IntentGraphResult:
        """Analyze a conversation for intent escalation.

        Args:
            turns: List of conversation turns (must have ``text`` and ``timestamp``).
            jailbreak_similarities: Per-turn jailbreak similarity scores (optional).

        Returns:
            IntentGraphResult with final score and per-node risk details.
        """
        n = len(turns)
        if n == 0:
            return IntentGraphResult(
                intent_score=0.0, node_risks=[], base_risks=[], iterations=0, converged=True
            )

        # Embed all turns
        texts = [t["text"] for t in turns]
        embeddings = self._embedder.embed(texts)

        # Parse timestamps
        timestamps = []
        for t in turns:
            ts = t["timestamp"]
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            timestamps.append(ts)

        # Default jailbreak similarities
        if jailbreak_similarities is None:
            jailbreak_similarities = [0.0] * n

        # Compute base risks
        base_risks = [
            compute_base_risk(texts[i], jailbreak_similarities[i]) for i in range(n)
        ]

        # Build graph
        self._graph = nx.DiGraph()
        for i in range(n):
            self._graph.add_node(
                i,
                text=texts[i],
                timestamp=timestamps[i],
                base_risk=base_risks[i],
                risk=base_risks[i],
            )

        # FIX 4: Use increased K_NEIGHBORS for more connectivity
        k = settings.INTENT_GRAPH_K_NEIGHBORS
        lam = settings.INTENT_GRAPH_LAMBDA

        for i in range(1, n):
            for j in range(max(0, i - k), i):
                sim = cosine_similarity(embeddings[i], embeddings[j])
                dt = (timestamps[i] - timestamps[j]).total_seconds()
                weight = sim * math.exp(-lam * abs(dt))
                self._graph.add_edge(j, i, weight=max(0.0, weight))

        # Propagate risk
        # FIX 4: Use increased ALPHA for stronger propagation
        risks = list(base_risks)
        alpha = settings.INTENT_GRAPH_ALPHA
        max_iter = settings.INTENT_GRAPH_MAX_ITERATIONS
        convergence = settings.INTENT_GRAPH_CONVERGENCE

        converged = False
        iterations = 0
        for iteration in range(max_iter):
            iterations = iteration + 1
            new_risks = list(base_risks)

            for i in range(n):
                incoming_weighted = 0.0
                incoming_total_weight = 0.0
                for j in self._graph.predecessors(i):
                    w = self._graph[j][i]["weight"]
                    incoming_weighted += risks[j] * w
                    incoming_total_weight += w
                if incoming_total_weight > 0:
                    # FIX 4: Normalized propagation with damping
                    propagated = incoming_weighted / incoming_total_weight
                    new_risks[i] = min(1.0, (1 - alpha) * base_risks[i] + alpha * propagated)
                    # Ensure risk never decreases below base
                    new_risks[i] = max(new_risks[i], base_risks[i])

            max_change = max(abs(new_risks[i] - risks[i]) for i in range(n))
            risks = new_risks

            if max_change < convergence:
                converged = True
                break

        # Update graph node risks
        for i in range(n):
            self._graph.nodes[i]["risk"] = risks[i]

        # Compute final intent score
        intent_score = self._compute_final_score(risks, timestamps)

        # Optional graph visualization data
        graph_data = None
        if settings.ENABLE_GRAPH_VISUALIZATION:
            graph_data = self._export_graph_data(risks, base_risks, iterations, converged, intent_score)

        return IntentGraphResult(
            intent_score=intent_score,
            node_risks=risks,
            base_risks=base_risks,
            iterations=iterations,
            converged=converged,
            graph_data=graph_data,
        )

    def _compute_final_score(self, risks: List[float], timestamps: List[datetime]) -> float:
        """Compute the final intent score from node risks.

        Args:
            risks: Per-node risk values.
            timestamps: Per-node timestamps.

        Returns:
            Final intent score.
        """
        if not risks:
            return 0.0

        method = settings.INTENT_SCORING_METHOD

        if method == "max":
            return max(risks)

        elif method == "weighted_avg":
            # Weight by recency (more recent = higher weight)
            if len(risks) == 1:
                return risks[0]
            latest = max(timestamps)
            weights = []
            for ts in timestamps:
                dt = (latest - ts).total_seconds()
                weights.append(math.exp(-0.01 * dt))
            total_weight = sum(weights)
            if total_weight == 0:
                return max(risks)
            return sum(r * w for r, w in zip(risks, weights)) / total_weight

        else:
            return max(risks)

    def _export_graph_data(
        self,
        risks: List[float],
        base_risks: List[float],
        iterations: int,
        converged: bool,
        intent_score: float,
    ) -> Dict[str, Any]:
        """Export graph structure for visualization.

        Returns:
            JSON-serializable graph data.
        """
        nodes = []
        for i in self._graph.nodes:
            data = self._graph.nodes[i]
            nodes.append({
                "id": i,
                "text": data.get("text", ""),
                "timestamp": data.get("timestamp", "").isoformat() if hasattr(data.get("timestamp", ""), "isoformat") else str(data.get("timestamp", "")),
                "risk": risks[i] if i < len(risks) else 0.0,
                "base_risk": base_risks[i] if i < len(base_risks) else 0.0,
            })

        edges = []
        for u, v, data in self._graph.edges(data=True):
            edges.append({
                "source": u,
                "target": v,
                "weight": data.get("weight", 0.0),
            })

        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "iterations": iterations,
                "converged": converged,
                "final_intent_score": intent_score,
            },
        }

    @property
    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()
