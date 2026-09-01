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


def incomplete(
    checkpoint_id: str,
    features: dict[str, float | bool | None],
    *,
    reason_code: str,
    reason: str,
    suggestion: str,
) -> JudgeDecision:
    return JudgeDecision(
        checkpoint_id=checkpoint_id,
        status=CheckpointStatus.INCOMPLETE,
        confidence=1,
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


def decide_boolean_rules(
    checkpoint_id: str,
    features: dict[str, float | bool | None],
    rules: dict[str, str],
) -> JudgeDecision:
    required = {name: features.get(name) for name in rules}
    if any(value is None for value in required.values()):
        return needs_review(
            checkpoint_id,
            features,
            reason_code="missing_required_evidence",
            reason="判定所需的视觉证据不完整。",
        )
    failed = [rule for name, rule in rules.items() if required[name] is not True]
    if failed:
        return incorrect(
            checkpoint_id,
            features,
            reason_code="criterion_failed",
            reason="操作未满足该考核点的全部要求。",
            matched_rules=failed,
            suggestion="请对照考核标准重新练习未满足的操作要求。",
        )
    return correct(
        checkpoint_id,
        features,
        matched_rules=list(rules.values()),
        reason="操作满足该考核点的全部要求。",
    )
