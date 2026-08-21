from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, computed_field, model_validator

from medical_evaluation.domain import CheckpointStatus, TimeRange
from medical_evaluation.scoring import ScoreSummary, aggregate_equal_weight_score


class EvidenceItem(BaseModel):
    time_sec: float = Field(ge=0)
    overlay_path: str = Field(min_length=1)
    rule: str = Field(min_length=1)


class CheckpointResult(BaseModel):
    checkpoint_id: str = Field(pattern=r"^cp_\d{2}$")
    status: CheckpointStatus
    confidence: float = Field(ge=0, le=1)
    time_range: TimeRange | None = None
    evidence: list[EvidenceItem] = Field(default_factory=list)
    features: dict[str, float | bool | None] = Field(default_factory=dict)
    matched_rules: list[str] = Field(default_factory=list)
    reason_code: str = Field(min_length=1)
    reason: str = ""
    suggestion: str = ""

    @computed_field
    @property
    def score(self) -> float | None:
        if self.status is CheckpointStatus.NEEDS_REVIEW:
            return None
        return 100.0 / 11.0 if self.status is CheckpointStatus.CORRECT else 0.0


class RunAudit(BaseModel):
    rubric_version: str = Field(min_length=1)
    model_versions: dict[str, str]
    started_at: datetime
    completed_at: datetime | None = None
    runtime_sec: float | None = Field(default=None, ge=0)
    peak_gpu_memory_mb: float | None = Field(default=None, ge=0)
    degradations: list[str] = Field(default_factory=list)


class EvaluationReport(BaseModel):
    job_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    checkpoints: list[CheckpointResult]
    audit: RunAudit
    overall_feedback: str = ""

    @model_validator(mode="after")
    def validate_checkpoints(self) -> EvaluationReport:
        checkpoint_ids = [item.checkpoint_id for item in self.checkpoints]
        expected = [f"cp_{number:02d}" for number in range(1, 12)]
        if checkpoint_ids != expected or len(set(checkpoint_ids)) != 11:
            raise ValueError("report must contain 11 unique checkpoint results in order")
        return self

    @computed_field
    @property
    def summary(self) -> ScoreSummary:
        return aggregate_equal_weight_score([item.status for item in self.checkpoints])
