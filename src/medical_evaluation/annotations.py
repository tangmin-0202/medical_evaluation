from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from medical_evaluation.domain import CheckpointStatus, TimeRange
from medical_evaluation.storage import atomic_write_json, safe_child


class SegmentAnnotation(BaseModel):
    checkpoint_id: str = Field(pattern=r"^cp_\d{2}$")
    time_range: TimeRange
    label: CheckpointStatus
    reason: str = Field(min_length=1)


class PointPrompt(BaseModel):
    kind: Literal["point"] = "point"
    video_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    frame_time_sec: float = Field(ge=0)
    object_id: str = Field(min_length=1)
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    positive: bool = True


class BoxPrompt(BaseModel):
    kind: Literal["box"] = "box"
    video_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    frame_time_sec: float = Field(ge=0)
    object_id: str = Field(min_length=1)
    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)
    x2: float = Field(ge=0, le=1)
    y2: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_corners(self) -> BoxPrompt:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("box bottom-right must follow top-left")
        return self


class MaskPrompt(BaseModel):
    kind: Literal["mask"] = "mask"
    video_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    frame_time_sec: float = Field(ge=0)
    object_id: str = Field(min_length=1)
    mask_path: str = Field(min_length=1)
    mask_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


PromptAnnotation = Annotated[PointPrompt | BoxPrompt | MaskPrompt, Field(discriminator="kind")]


class AuditEntry(BaseModel):
    timestamp: datetime
    actor: str = Field(min_length=1)
    prior_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    new_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class VideoAnnotations(BaseModel):
    video_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    steps: list[SegmentAnnotation] = Field(default_factory=list)
    prompts: list[PromptAnnotation] = Field(default_factory=list)
    audit_history: list[AuditEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_step_order(self) -> VideoAnnotations:
        checkpoint_ids = [item.checkpoint_id for item in self.steps]
        if len(checkpoint_ids) != len(set(checkpoint_ids)):
            raise ValueError("duplicate checkpoint IDs are not allowed")
        numeric_ids = [int(item.removeprefix("cp_")) for item in checkpoint_ids]
        if numeric_ids != sorted(numeric_ids):
            raise ValueError("checkpoint IDs must follow fixed order")
        pairs = zip(self.steps[:-1], self.steps[1:], strict=True)
        for previous, current in pairs:
            if current.time_range.start_sec < previous.time_range.end_sec:
                raise ValueError("step ranges must not overlap")
        return self


class AnnotationStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def load_segments(self, video_id: str) -> VideoAnnotations:
        path = self._video_path(video_id)
        return VideoAnnotations.model_validate_json(path.read_text(encoding="utf-8"))

    def save_segments(
        self,
        video_id: str,
        steps: list[SegmentAnnotation],
        *,
        actor: str,
    ) -> VideoAnnotations:
        path = self._video_path(video_id)
        current = self.load_segments(video_id) if path.exists() else None
        prompts = current.prompts if current else []
        history = list(current.audit_history) if current else []
        candidate = VideoAnnotations(video_id=video_id, steps=steps, prompts=prompts)
        prior_digest = _content_digest(current) if current else None
        new_digest = _content_digest(candidate)
        history.append(
            AuditEntry(
                timestamp=datetime.now(UTC),
                actor=actor,
                prior_digest=prior_digest,
                new_digest=new_digest,
            )
        )
        saved = candidate.model_copy(update={"audit_history": history})
        atomic_write_json(path, saved.model_dump(mode="json"))
        return saved

    def _video_path(self, video_id: str) -> Path:
        if re.fullmatch(r"[A-Za-z0-9_-]+", video_id) is None:
            raise ValueError("video_id contains unsupported characters")
        return safe_child(self.root, f"{video_id}.json")


def _content_digest(annotations: VideoAnnotations) -> str:
    payload = {
        "video_id": annotations.video_id,
        "steps": [item.model_dump(mode="json") for item in annotations.steps],
        "prompts": [item.model_dump(mode="json") for item in annotations.prompts],
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
