"""Pydantic response schemas for the SHIELD API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# --- /analyze ---


class Violation(BaseModel):
    """A single violation flagged by a detection module."""

    module: str
    score: float
    reason: str
    details: Dict[str, Any] = Field(default_factory=dict)


class ScoreBreakdown(BaseModel):
    """Per-module score breakdown."""

    jailbreak_score: float = 0.0
    intent_score: float = 0.0
    privacy_score: float = 0.0


class ResponseMetadata(BaseModel):
    """Metadata about the analysis."""

    mode: str
    models_used: List[str] = Field(default_factory=list)
    conversation_length: Optional[int] = None
    graph_nodes: Optional[int] = None
    graph_edges: Optional[int] = None
    jailbreak_corpus_size: int = 0
    active_clusters: int = 0
    faces_detected: Optional[int] = None


class AnalyzeResponse(BaseModel):
    """Full response from the /analyze endpoint."""

    request_id: str
    timestamp: datetime
    final_action: Literal["block", "allow", "review"]
    final_score: float
    processing_time_ms: int
    violations: List[Violation] = Field(default_factory=list)
    scores: ScoreBreakdown
    metadata: ResponseMetadata


# --- /health ---


class ComponentHealth(BaseModel):
    """Health status of a single component."""

    status: Literal["healthy", "degraded", "unhealthy", "mock"]
    details: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """System health check response."""

    status: Literal["healthy", "degraded", "unhealthy"]
    timestamp: datetime
    version: str
    mode: str
    uptime_seconds: int
    components: Dict[str, ComponentHealth] = Field(default_factory=dict)


# --- /ingest ---


class IngestResponse(BaseModel):
    """Response from jailbreak ingestion."""

    added_count: int
    total_corpus_size: int
    message: str
    next_action: str = "Call POST /train/refresh_clusters to update"


# --- /train ---


class ClusteringJobResult(BaseModel):
    """Result of a clustering job."""

    clusters_found: int
    noise_points: int
    duration_seconds: float


class RefreshClusterResponse(BaseModel):
    """Response from cluster refresh trigger."""

    job_id: str
    status: Literal["started", "queued"]
    estimated_duration_seconds: int
    corpus_size: int


class JobStatusResponse(BaseModel):
    """Status of a background clustering job."""

    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    progress_percent: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[ClusteringJobResult] = None
    error: Optional[str] = None


# --- Errors ---


class ErrorResponse(BaseModel):
    """Structured error response."""

    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime
