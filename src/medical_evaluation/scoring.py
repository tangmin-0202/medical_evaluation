from __future__ import annotations

from pydantic import BaseModel

from medical_evaluation.domain import CheckpointStatus


class ScoreSummary(BaseModel):
    final_score: float | None
    minimum_score: float
    maximum_score: float


def aggregate_equal_weight_score(statuses: list[CheckpointStatus]) -> ScoreSummary:
    if len(statuses) != 11:
        raise ValueError("exactly 11 statuses are required")
    unit = 100.0 / 11.0
    correct_count = sum(value is CheckpointStatus.CORRECT for value in statuses)
    review_count = sum(value is CheckpointStatus.NEEDS_REVIEW for value in statuses)
    minimum = correct_count * unit
    maximum = (correct_count + review_count) * unit
    return ScoreSummary(
        final_score=None if review_count else minimum,
        minimum_score=minimum,
        maximum_score=maximum,
    )
