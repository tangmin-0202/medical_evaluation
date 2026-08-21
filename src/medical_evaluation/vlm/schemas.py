from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class VlmReviewRequest(BaseModel):
    checkpoint_id: str = Field(pattern=r"^cp_\d{2}$")
    checkpoint_name: str = Field(min_length=1)
    criteria: list[str] = Field(min_length=1)
    deterministic_status: Literal["correct", "incorrect", "incomplete", "needs_review"]
    reason_code: str = Field(min_length=1)
    features: dict[str, float | bool | None]
    evidence_images: list[Path] = Field(default_factory=list)


class VlmReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_supported: bool
    semantic_status: Literal["supports", "contradicts", "uncertain"]
    reason_zh: str = Field(min_length=1)
    suggestion_zh: str = Field(min_length=1)
    cited_evidence_indices: list[int]
    source: Literal["qwen", "template_fallback"] = "qwen"
    score_override: None = None
