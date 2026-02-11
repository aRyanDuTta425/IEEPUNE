"""Pydantic request schemas for the SHIELD API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


class ConversationTurn(BaseModel):
    """A single turn in a conversation."""

    sender: str = Field(..., description="'user', 'assistant', or 'system'")
    text: str = Field(..., min_length=1)
    timestamp: datetime


class ConversationRequest(BaseModel):
    """Analyze a multi-turn conversation for safety violations."""

    type: Literal["conversation"] = "conversation"
    conversation: List[ConversationTurn] = Field(..., min_length=1)
    metadata: Optional[Dict[str, Any]] = None


class ImageEditRequest(BaseModel):
    """Analyze an image editing request for privacy violations."""

    type: Literal["image_edit"] = "image_edit"
    prompt: str = Field(..., min_length=1)
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def _check_image_source(self) -> "ImageEditRequest":
        if self.image_url and self.image_base64:
            raise ValueError("Provide exactly one of image_url or image_base64, not both")
        return self


# Union discriminated by the 'type' field
AnalyzeRequest = Union[ConversationRequest, ImageEditRequest]


class JailbreakPrompt(BaseModel):
    """A single jailbreak prompt for ingestion."""

    text: str = Field(..., min_length=1)
    category: Optional[str] = None
    severity: Optional[int] = Field(None, ge=1, le=3)


class IngestRequest(BaseModel):
    """Ingest new jailbreak prompts into the corpus."""

    prompts: List[JailbreakPrompt] = Field(..., min_length=1)


class RefreshClusterRequest(BaseModel):
    """Trigger re-clustering of jailbreak corpus."""

    min_cluster_size: Optional[int] = None
    save_to_disk: bool = True
