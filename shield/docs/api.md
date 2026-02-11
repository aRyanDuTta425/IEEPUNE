# SHIELD API Reference

Base URL: `http://localhost:8000`

---

## POST /analyze

Analyze a conversation or image edit request for safety violations.

### Conversation Request

```json
{
  "type": "conversation",
  "conversation": [
    {
      "sender": "user",
      "text": "Hello, how are you?",
      "timestamp": "2026-02-10T10:00:00Z"
    },
    {
      "sender": "assistant",
      "text": "I'm doing well!",
      "timestamp": "2026-02-10T10:00:15Z"
    }
  ]
}
```

### Image Edit Request

```json
{
  "type": "image_edit",
  "prompt": "Change the background to a sunset",
  "image_base64": "<base64-encoded-image>",
  "image_url": "https://example.com/photo.jpg"
}
```

> Note: Provide either `image_base64` or `image_url`, not both.

### Response (200)

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2026-02-10T10:00:01Z",
  "final_action": "allow",
  "final_score": 0.15,
  "processing_time_ms": 42,
  "violations": [],
  "scores": {
    "jailbreak_score": 0.15,
    "intent_score": 0.08,
    "privacy_score": 0.0
  },
  "metadata": {
    "mode": "mock",
    "models_used": ["mock_384d"],
    "conversation_length": 2,
    "graph_nodes": 2,
    "graph_edges": 1,
    "jailbreak_corpus_size": 103,
    "active_clusters": 5
  }
}
```

### Violation Object

When a violation is detected:

```json
{
  "module": "jailbreak_detector",
  "score": 0.92,
  "reason": "Prompt closely matches known jailbreak patterns",
  "details": {
    "category": "instruction_injection",
    "top_cluster": 3
  }
}
```

### Error Responses

| Status | Description |
|--------|-------------|
| 413 | Conversation exceeds maximum length |
| 422 | Invalid request body or image format |
| 500 | Internal server error |

---

## GET /health

System health check and component status.

### Response (200)

```json
{
  "status": "healthy",
  "timestamp": "2026-02-10T10:00:00Z",
  "version": "0.1.0",
  "mode": "mock",
  "uptime_seconds": 3600,
  "components": {
    "embedding_provider": {
      "status": "mock",
      "details": {"model_name": "mock_384d", "dimension": 384}
    },
    "jailbreak_detector": {
      "status": "healthy",
      "details": {"corpus_size": 103, "num_clusters": 5}
    }
  }
}
```

Status values: `healthy`, `degraded`, `unhealthy`

---

## POST /ingest/jailbreaks

Add new jailbreak prompts to the detection corpus.

### Request

```json
{
  "prompts": [
    {
      "text": "New jailbreak prompt text",
      "category": "roleplay",
      "severity": 2
    }
  ]
}
```

### Response (200)

```json
{
  "added_count": 1,
  "total_corpus_size": 104,
  "message": "Successfully added 1 prompts to corpus",
  "next_action": "Call POST /train/refresh_clusters to update cluster index"
}
```

---

## POST /train/refresh_clusters

Trigger background re-clustering of the jailbreak corpus.

### Request

```json
{
  "min_cluster_size": 5,
  "save_to_disk": true
}
```

### Response (200)

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "started",
  "estimated_duration_seconds": 2,
  "corpus_size": 104
}
```

---

## GET /train/status/{job_id}

Check status of a background clustering job.

### Response (200)

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "progress_percent": 100,
  "started_at": "2026-02-10T10:00:00Z",
  "completed_at": "2026-02-10T10:00:02Z",
  "result": {
    "clusters_found": 6,
    "noise_points": 3,
    "duration_seconds": 1.85
  }
}
```

Status values: `running`, `completed`, `failed`

---

## GET /metrics

Prometheus-compatible metrics export.

### Response (200, text/plain)

```
# HELP shield_requests_total Total requests processed
# TYPE shield_requests_total counter
shield_requests_total{action="allow"} 42
shield_requests_total{action="block"} 5
shield_requests_total{action="review"} 8

# HELP shield_processing_duration_seconds Request processing time
# TYPE shield_processing_duration_seconds histogram
shield_processing_duration_seconds_count 55
shield_processing_duration_seconds_sum 2.31

# HELP shield_active_clusters Number of active jailbreak clusters
# TYPE shield_active_clusters gauge
shield_active_clusters 6
```

> Returns 404 if `ENABLE_METRICS=false`
