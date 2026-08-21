from __future__ import annotations

from pydantic import BaseModel, Field

from medical_evaluation.domain import CheckpointStatus


class JudgeDecision(BaseModel):
    checkpoint_id: str = Field(pattern=r"^cp_\d{2}$")
    status: CheckpointStatus
    confidence: float = Field(ge=0, le=1)
    matched_rules: list[str] = Field(default_factory=list)
    features: dict[str, float | bool | None]
    reason_code: str = Field(min_length=1)
    reason: str
    suggestion: str


def needs_review(
    checkpoint_id: str,
    features: dict[str, float | bool | None],
    *,
    reason_code: str,
    reason: str,
) -> JudgeDecision:
    return JudgeDecision(
        checkpoint_id=checkpoint_id,
        status=CheckpointStatus.NEEDS_REVIEW,
        confidence=0,
        features=features,
        reason_code=reason_code,
        reason=reason,
        suggestion="请人工复核该考核点的视频证据。",
    )


def incorrect(
    checkpoint_id: str,
    features: dict[str, float | bool | None],
    *,
    reason_code: str,
    reason: str,
    matched_rules: list[str] | None = None,
    suggestion: str,
) -> JudgeDecision:
    return JudgeDecision(
        checkpoint_id=checkpoint_id,
        status=CheckpointStatus.INCORRECT,
        confidence=1,
        matched_rules=matched_rules or [],
        features=features,
        reason_code=reason_code,
        reason=reason,
        suggestion=suggestion,
    )


def correct(
    checkpoint_id: str,
    features: dict[str, float | bool | None],
    *,
    matched_rules: list[str],
    reason: str,
) -> JudgeDecision:
    return JudgeDecision(
        checkpoint_id=checkpoint_id,
        status=CheckpointStatus.CORRECT,
        confidence=1,
        matched_rules=matched_rules,
        features=features,
        reason_code="criteria_satisfied",
        reason=reason,
        suggestion="继续保持规范操作。",
    )
