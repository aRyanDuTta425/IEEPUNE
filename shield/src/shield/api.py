"""FastAPI application — all SHIELD API endpoints."""

from __future__ import annotations

import asyncio
import base64
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Dict, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from shield import __version__
from shield.config import settings
from shield.core.decision_fusion import FusionInput, fuse
from shield.core.embeddings import EmbeddingProvider, create_embedding_provider
from shield.core.intent_graph import IntentGraph
from shield.core.jailbreak_detector import JailbreakDetector
from shield.core.privacy_predictor import PrivacyPredictor
from shield.core.utils import metrics
from shield.ml.mocks import MockAgeEstimator, MockFaceDetector, MockTransformClassifier, SemanticTransformClassifier
from shield.models.requests import (
    AnalyzeRequest,
    ConversationRequest,
    ImageEditRequest,
    IngestRequest,
    RefreshClusterRequest,
)
from shield.models.responses import (
    AnalyzeResponse,
    ClusteringJobResult,
    ComponentHealth,
    ErrorResponse,
    HealthResponse,
    IngestResponse,
    JobStatusResponse,
    RefreshClusterResponse,
    ResponseMetadata,
    ScoreBreakdown,
    Violation,
)

logger = logging.getLogger(__name__)

# ── Global state ──────────────────────────────────────────────────────────────
_start_time: float = 0.0
_embedding_provider: Optional[EmbeddingProvider] = None
_jailbreak_detector: Optional[JailbreakDetector] = None
_intent_graph: Optional[IntentGraph] = None
_privacy_predictor: Optional[PrivacyPredictor] = None
_background_jobs: Dict[str, Dict[str, Any]] = {}


def _create_ml_models() -> tuple:
    """Create ML model instances based on deployment mode.

    FIX 5: lightweight mode now uses SemanticTransformClassifier
    with the embedding provider for pseudo-CLIP behavior.
    """
    mode = settings.SHIELD_MODE
    if mode == "mock":
        face = MockFaceDetector()
        age = MockAgeEstimator(default_age=25)
        transform = MockTransformClassifier()
    elif mode == "lightweight":
        # FIX 5: Use semantic classifier with real embeddings
        face = MockFaceDetector()
        age = MockAgeEstimator(default_age=25)
        transform = SemanticTransformClassifier(embedding_provider=_embedding_provider)
    else:
        # Full mode — attempt real models, fall back to mock
        try:
            from shield.ml.face_detector import RealFaceDetector
            face = RealFaceDetector()
        except Exception:
            logger.warning("Failed to load real face detector — using mock")
            face = MockFaceDetector()
        try:
            from shield.ml.age_estimator import RealAgeEstimator
            age = RealAgeEstimator()
        except Exception:
            logger.warning("Failed to load real age estimator — using mock")
            age = MockAgeEstimator()
        try:
            from shield.ml.transform_classifier import RealTransformClassifier
            transform = RealTransformClassifier()
        except Exception:
            logger.warning("Failed to load real transform classifier — using mock")
            transform = MockTransformClassifier()

    return face, age, transform


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    """Initialize all components on startup, clean up on shutdown."""
    global _start_time, _embedding_provider, _jailbreak_detector, _intent_graph, _privacy_predictor

    _start_time = time.time()
    logger.info("SHIELD starting in %s mode (v%s)", settings.SHIELD_MODE, __version__)

    # Embedding provider
    _embedding_provider = create_embedding_provider()

    # Jailbreak detector
    _jailbreak_detector = JailbreakDetector(_embedding_provider)
    _jailbreak_detector.initialize()

    # Intent graph
    _intent_graph = IntentGraph(_embedding_provider)

    # Privacy predictor
    face, age, transform = _create_ml_models()
    _privacy_predictor = PrivacyPredictor(face, age, transform)

    logger.info("SHIELD ready — all components initialized")
    yield
    logger.info("SHIELD shutting down")


app = FastAPI(
    title="SHIELD",
    description="Secure Hierarchical Intent Evaluation and Leak Defense",
    version=__version__,
    lifespan=lifespan,
)


# ── Error Handlers ────────────────────────────────────────────────────────────


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=str(exc.status_code),
            message=exc.detail,
            timestamp=datetime.now(timezone.utc),
        ).model_dump(mode="json"),
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_server_error",
            message="An internal server error occurred",
            details={"type": type(exc).__name__},
            timestamp=datetime.now(timezone.utc),
        ).model_dump(mode="json"),
    )


# ── POST /analyze ─────────────────────────────────────────────────────────────


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: ConversationRequest | ImageEditRequest) -> AnalyzeResponse:
    """Analyze conversation or image request for safety violations."""
    start = time.perf_counter()
    request_id = str(uuid.uuid4())

    assert _jailbreak_detector is not None
    assert _intent_graph is not None
    assert _privacy_predictor is not None

    if isinstance(request, ConversationRequest):
        return await _analyze_conversation(request, request_id, start)
    else:
        return await _analyze_image(request, request_id, start)


async def _analyze_conversation(
    request: ConversationRequest, request_id: str, start: float
) -> AnalyzeResponse:
    """Handle conversation analysis."""
    # Validate length
    if len(request.conversation) > settings.MAX_CONVERSATION_LENGTH:
        raise HTTPException(
            status_code=413,
            detail=f"Conversation exceeds maximum length of {settings.MAX_CONVERSATION_LENGTH}",
        )

    turns = [
        {"text": t.text, "timestamp": t.timestamp, "sender": t.sender}
        for t in request.conversation
    ]

    # Run jailbreak detection on the latest user turn
    latest_user = next(
        (t for t in reversed(turns) if t["sender"] == "user"), turns[-1]
    )
    jb_result = _jailbreak_detector.detect(latest_user["text"])  # type: ignore[union-attr]

    # Run jailbreak on all turns for intent graph
    jb_sims = [_jailbreak_detector.detect(t["text"]).jailbreak_score for t in turns]  # type: ignore[union-attr]

    # Run intent graph
    ig_result = _intent_graph.analyze(turns, jailbreak_similarities=jb_sims)  # type: ignore[union-attr]

    # Decision fusion
    fusion_input = FusionInput(
        jailbreak_score=jb_result.jailbreak_score,
        jailbreak_details=jb_result.details,
        intent_score=ig_result.intent_score,
        intent_details={
            "node_risks": ig_result.node_risks,
            "iterations": ig_result.iterations,
            "converged": ig_result.converged,
        },
        privacy_score=0.0,  # no image in conversation mode
    )
    fusion_result = fuse(fusion_input)

    elapsed_ms = int((time.perf_counter() - start) * 1000)

    # Metrics
    metrics.inc_counter("shield_requests_total", {"action": fusion_result.action.value})
    metrics.observe_histogram("shield_processing_duration_seconds", elapsed_ms / 1000)

    return AnalyzeResponse(
        request_id=request_id,
        timestamp=datetime.now(timezone.utc),
        final_action=fusion_result.action.value,
        final_score=fusion_result.final_score,
        processing_time_ms=elapsed_ms,
        violations=[
            Violation(module=v.module, score=v.score, reason=v.reason, details=v.details)
            for v in fusion_result.violations
        ],
        scores=ScoreBreakdown(
            jailbreak_score=fusion_result.scores["jailbreak_score"],
            intent_score=fusion_result.scores["intent_score"],
            privacy_score=fusion_result.scores["privacy_score"],
        ),
        metadata=ResponseMetadata(
            mode=settings.SHIELD_MODE,
            models_used=_get_models_used(),
            conversation_length=len(turns),
            graph_nodes=ig_result.node_risks.__len__() if ig_result.node_risks else 0,
            graph_edges=_intent_graph.edge_count if _intent_graph else 0,  # type: ignore[union-attr]
            jailbreak_corpus_size=_jailbreak_detector.corpus_size,  # type: ignore[union-attr]
            active_clusters=_jailbreak_detector.num_clusters,  # type: ignore[union-attr]
        ),
    )


async def _analyze_image(
    request: ImageEditRequest, request_id: str, start: float
) -> AnalyzeResponse:
    """Handle image edit analysis."""
    # Resolve image
    image_array: Optional[np.ndarray] = None
    if request.image_base64:
        try:
            from PIL import Image

            img_bytes = base64.b64decode(request.image_base64)
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            # Resize if needed
            if max(img.size) > settings.MAX_IMAGE_DIMENSION:
                ratio = settings.MAX_IMAGE_DIMENSION / max(img.size)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size)
            image_array = np.array(img)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Invalid image format: {e}")

    elif request.image_url:
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    request.image_url, timeout=aiohttp.ClientTimeout(total=settings.IMAGE_DOWNLOAD_TIMEOUT)
                ) as resp:
                    if resp.status != 200:
                        raise HTTPException(
                            status_code=422,
                            detail=f"Failed to download image from URL (HTTP {resp.status})",
                        )
                    data = await resp.read()
                    size_mb = len(data) / (1024 * 1024)
                    if size_mb > settings.MAX_IMAGE_SIZE_MB:
                        raise HTTPException(
                            status_code=413,
                            detail=f"Image size ({size_mb:.1f}MB) exceeds {settings.MAX_IMAGE_SIZE_MB}MB limit",
                        )
                    from PIL import Image

                    img = Image.open(BytesIO(data)).convert("RGB")
                    image_array = np.array(img)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Failed to download image from URL: {e}")

    # Run jailbreak on the prompt
    jb_result = _jailbreak_detector.detect(request.prompt)  # type: ignore[union-attr]

    # Run privacy predictor
    priv_result = _privacy_predictor.predict(request.prompt, image_array)  # type: ignore[union-attr]

    # Decision fusion
    fusion_input = FusionInput(
        jailbreak_score=jb_result.jailbreak_score,
        jailbreak_details=jb_result.details,
        intent_score=0.0,  # no conversation context for single image request
        privacy_score=priv_result.privacy_score,
        privacy_details={
            "identity_type": priv_result.identity_type,
            "transformations": priv_result.transformations,
            **priv_result.details,
        },
    )
    fusion_result = fuse(fusion_input)

    elapsed_ms = int((time.perf_counter() - start) * 1000)

    metrics.inc_counter("shield_requests_total", {"action": fusion_result.action.value})
    metrics.observe_histogram("shield_processing_duration_seconds", elapsed_ms / 1000)

    return AnalyzeResponse(
        request_id=request_id,
        timestamp=datetime.now(timezone.utc),
        final_action=fusion_result.action.value,
        final_score=fusion_result.final_score,
        processing_time_ms=elapsed_ms,
        violations=[
            Violation(module=v.module, score=v.score, reason=v.reason, details=v.details)
            for v in fusion_result.violations
        ],
        scores=ScoreBreakdown(
            jailbreak_score=fusion_result.scores["jailbreak_score"],
            intent_score=fusion_result.scores["intent_score"],
            privacy_score=fusion_result.scores["privacy_score"],
        ),
        metadata=ResponseMetadata(
            mode=settings.SHIELD_MODE,
            models_used=_get_models_used(),
            jailbreak_corpus_size=_jailbreak_detector.corpus_size,  # type: ignore[union-attr]
            active_clusters=_jailbreak_detector.num_clusters,  # type: ignore[union-attr]
            faces_detected=priv_result.faces_detected,
        ),
    )


# ── GET /health ───────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """System health check and component status."""
    components: Dict[str, ComponentHealth] = {}

    # Embedding provider
    if _embedding_provider:
        eh = _embedding_provider.health_check()
        components["embedding_provider"] = ComponentHealth(
            status=eh.get("status", "unhealthy"), details=eh
        )

    # Vector index / jailbreak detector
    if _jailbreak_detector:
        jh = _jailbreak_detector.health_check()
        components["jailbreak_detector"] = ComponentHealth(
            status=jh.get("status", "unhealthy"), details=jh
        )

    # Privacy predictor components
    if _privacy_predictor:
        ph = _privacy_predictor.health_check()
        for comp_name, comp_health in ph.items():
            components[comp_name] = ComponentHealth(
                status=comp_health.get("status", "unhealthy"), details=comp_health
            )

    # Determine overall status
    statuses = [c.status for c in components.values()]
    if any(s == "unhealthy" for s in statuses):
        overall_status = "unhealthy"
    elif any(s == "degraded" for s in statuses):
        overall_status = "degraded"
    else:
        overall_status = "healthy"

    uptime = int(time.time() - _start_time) if _start_time else 0

    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now(timezone.utc),
        version=__version__,
        mode=settings.SHIELD_MODE,
        uptime_seconds=uptime,
        components=components,
    )


# ── POST /ingest/jailbreaks ──────────────────────────────────────────────────


@app.post("/ingest/jailbreaks", response_model=IngestResponse)
async def ingest_jailbreaks(request: IngestRequest) -> IngestResponse:
    """Add new jailbreak prompts to the corpus."""
    assert _jailbreak_detector is not None

    prompts = [p.model_dump() for p in request.prompts]
    added = _jailbreak_detector.add_prompts(prompts)

    return IngestResponse(
        added_count=added,
        total_corpus_size=_jailbreak_detector.corpus_size,
        message=f"Successfully added {added} prompts to corpus",
        next_action="Call POST /train/refresh_clusters to update cluster index",
    )


# ── POST /train/refresh_clusters ─────────────────────────────────────────────


@app.post("/train/refresh_clusters", response_model=RefreshClusterResponse)
async def refresh_clusters(request: RefreshClusterRequest) -> RefreshClusterResponse:
    """Trigger re-clustering of jailbreak corpus."""
    assert _jailbreak_detector is not None

    job_id = str(uuid.uuid4())
    corpus_size = _jailbreak_detector.corpus_size

    # Run clustering in background
    _background_jobs[job_id] = {
        "status": "running",
        "progress_percent": 0,
        "started_at": datetime.now(timezone.utc),
    }

    async def _run_clustering() -> None:
        try:
            start = time.perf_counter()
            result = _jailbreak_detector.refresh_clusters(
                min_cluster_size=request.min_cluster_size,
                save_to_disk=request.save_to_disk,
            )
            elapsed = time.perf_counter() - start
            _background_jobs[job_id] = {
                "status": "completed",
                "progress_percent": 100,
                "started_at": _background_jobs[job_id]["started_at"],
                "completed_at": datetime.now(timezone.utc),
                "result": {
                    "clusters_found": result.num_clusters,
                    "noise_points": result.noise_count,
                    "duration_seconds": round(elapsed, 2),
                },
            }
            metrics.set_gauge("shield_active_clusters", result.num_clusters)
        except Exception as e:
            logger.exception("Clustering failed: %s", e)
            _background_jobs[job_id]["status"] = "failed"
            _background_jobs[job_id]["error"] = str(e)

    asyncio.create_task(_run_clustering())

    return RefreshClusterResponse(
        job_id=job_id,
        status="started",
        estimated_duration_seconds=max(1, corpus_size // 100),
        corpus_size=corpus_size,
    )


# ── GET /train/status/{job_id} ────────────────────────────────────────────────


@app.get("/train/status/{job_id}", response_model=JobStatusResponse)
async def cluster_job_status(job_id: str) -> JobStatusResponse:
    """Get status of a background clustering job."""
    if job_id not in _background_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = _background_jobs[job_id]
    result = None
    if job.get("result"):
        result = ClusteringJobResult(**job["result"])

    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress_percent=job.get("progress_percent", 0),
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        result=result,
        error=job.get("error"),
    )


# ── GET /metrics ──────────────────────────────────────────────────────────────


@app.get("/metrics")
async def get_metrics() -> PlainTextResponse:
    """Prometheus-compatible metrics export."""
    if not settings.ENABLE_METRICS:
        raise HTTPException(status_code=404, detail="Metrics endpoint disabled")

    # Update gauges
    if _jailbreak_detector:
        metrics.set_gauge("shield_active_clusters", _jailbreak_detector.num_clusters)

    return PlainTextResponse(content=metrics.to_prometheus(), media_type="text/plain")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _get_models_used() -> list:
    """Return list of model names in use."""
    models = []
    if _embedding_provider:
        eh = _embedding_provider.health_check()
        models.append(eh.get("model_name", "unknown"))
    if _privacy_predictor:
        ph = _privacy_predictor.health_check()
        for comp in ph.values():
            models.append(comp.get("model_name", "unknown"))
    return models
