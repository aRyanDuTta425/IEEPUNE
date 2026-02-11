# SHIELD Architecture

## System Overview

SHIELD (Secure Hierarchical Intent Evaluation and Leak Defense) is a three-layer defense system that intercepts, analyzes, and arbitrates every user request to a generative AI system. The system is designed to prevent jailbreaking, detect adversarial intent escalation, and block privacy-violating image edits.

## Data Flow

```
User Request
     │
     ▼
┌─────────────────────────┐
│   FastAPI /analyze       │
│   (api.py)               │
└─────────┬───────────────┘
          │
    ┌─────┴───────┐
    │ Request Type │
    └─────┬───────┘
          │
    ┌─────┼────────────────────┐
    ▼     ▼                    ▼
 Conv   Conv+Image          Image
    │     │                    │
    ▼     ▼                    ▼
┌────────────┐          ┌────────────┐
│ Jailbreak  │          │ Jailbreak  │
│ Detector   │          │ Detector   │
│ (Layer 1)  │          │ (Layer 1)  │
└─────┬──────┘          └─────┬──────┘
      │                       │
      ▼                       ▼
┌────────────┐          ┌────────────┐
│ Intent     │          │ Privacy    │
│ Graph      │          │ Predictor  │
│ (Layer 2)  │          │ (Layer 3)  │
└─────┬──────┘          └─────┬──────┘
      │                       │
      └───────────┬───────────┘
                  ▼
          ┌────────────┐
          │  Decision   │
          │  Fusion     │
          └──────┬─────┘
                 │
          ┌──────┼──────┐
          ▼      ▼      ▼
        ALLOW  REVIEW  BLOCK
```

## Layer 1: Meta-Jailbreak Detector

### Algorithm
1. **Corpus Loading**: Load known jailbreak prompts from JSON
2. **Embedding**: Convert all prompts to dense vectors via embedding provider
3. **Clustering**: Run HDBSCAN to find natural clusters in the embedding space
4. **Centroid Index**: Compute cluster centroids and build a FAISS index for fast lookup
5. **Detection**: For each incoming prompt:
   - Embed the prompt
   - Query the centroid index for top-K nearest centroids
   - Compute similarity score as the maximum cosine similarity
   - Flag as jailbreak if score ≥ threshold (default 0.75)

### Key Formula
```
jailbreak_score = max(cosine_similarity(prompt_embedding, centroid_i)) for i in top-K centroids
```

## Layer 2: Adversarial Intent Graph (AIG)

### Algorithm
1. **Graph Construction**: Each conversation turn becomes a node in a NetworkX digraph
2. **Edge Weights**: Between consecutive turns:
   ```
   w(i,j) = sim(emb_i, emb_j) × exp(−λ × Δt)
   ```
   where `sim` is cosine similarity, `λ` is the decay constant, and `Δt` is time delta in seconds
3. **Base Risk**: Each node gets a base risk from keyword analysis + jailbreak similarity
4. **Risk Propagation**: Iterative update until convergence:
   ```
   r_j^(t+1) = (1-α) × base_j + α × Σ(w(i,j) × r_i^(t)) / Σ(w(i,j))
   ```
5. **Scoring**: Final intent score = max(node_risks) or weighted average

### Convergence
- Default damping factor α = 0.3
- Convergence threshold ε = 0.001
- Maximum iterations = 10

## Layer 3: Privacy Consent Violation Predictor (PCVP)

### Pipeline
1. **Face Detection**: Detect faces in the input image (RetinaFace or MTCNN)
2. **Age Estimation**: Estimate age for each detected face
3. **Identity Classification**: Determine identity type:
   - `minor` (age < 18)
   - `public_figure` (recognized face)
   - `private_person` (default)
4. **Transform Classification**: Classify the editing intent from the text prompt:
   - clothing_removal, sexualization, pose_change, face_swap
   - age_regression, age_progression, background_change, facial_modification
5. **Risk Lookup**: Map (identity_type, transform_type) through the 2D risk matrix
6. **Score**: Maximum risk across all detected faces × transform combinations

### Risk Matrix (excerpt)
| Identity \ Transform | clothing_removal | sexualization | face_swap | background_change |
|----------------------|:---------------:|:------------:|:---------:|:-----------------:|
| **public_figure**    | 0.80            | 0.80         | 0.80      | 0.00              |
| **private_person**   | 0.95            | 0.95         | 0.95      | 0.50              |
| **minor**            | 0.95            | 0.95         | 0.95      | 0.50              |

## Decision Fusion

The final decision aggregates all three layer scores:

```
final_score = max(jailbreak_score, intent_score, privacy_score)

Action:
  BLOCK  if final_score ≥ 0.85
  REVIEW if final_score ≥ 0.65
  ALLOW  otherwise
```

## Deployment Modes

| Mode | Embeddings | Face Detection | Age Estimation | Transform Classification |
|------|-----------|---------------|----------------|-------------------------|
| `mock` | Hash-based (384d) | Mock (configurable) | Mock (default 25) | Keyword-based |
| `lightweight` | sentence-transformers | Mock | Mock | Keyword-based |
| `full` | sentence-transformers | RetinaFace/MTCNN | PyTorch model | CLIP zero-shot |
