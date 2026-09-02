from __future__ import annotations

from pydantic import BaseModel

from medical_evaluation.domain import CheckpointStatus


class ScoreSummary(BaseModel):
    final_score: float | None
    minimum_score: float
    maximum_score: float
    provisional_score: float | None
    provisional_minimum_score: float
    provisional_maximum_score: float
    evaluated_count: int
    total_count: int


def aggregate_equal_weight_score(statuses: list[CheckpointStatus]) -> ScoreSummary:
    return aggregate_evaluation_score(statuses, [True] * len(statuses))


def aggregate_evaluation_score(
    statuses: list[CheckpointStatus],
    included: list[bool],
) -> ScoreSummary:
    if len(statuses) != 11:
        raise ValueError("exactly 11 statuses are required")
    if len(included) != len(statuses):
        raise ValueError("included flags must match statuses")

    unit = 100.0 / 11.0
    correct_count = sum(
        is_included and value is CheckpointStatus.CORRECT
        for value, is_included in zip(statuses, included, strict=True)
    )
    review_count = sum(
        is_included and value is CheckpointStatus.NEEDS_REVIEW
        for value, is_included in zip(statuses, included, strict=True)
    )
    evaluated_count = sum(included)
    excluded_count = len(statuses) - evaluated_count
    minimum = correct_count * unit
    maximum = (correct_count + review_count + excluded_count) * unit

    if evaluated_count:
        provisional_unit = 100.0 / evaluated_count
        provisional_minimum = correct_count * provisional_unit
        provisional_maximum = (correct_count + review_count) * provisional_unit
        provisional_score = None if review_count else provisional_minimum
    else:
        provisional_minimum = 0.0
        provisional_maximum = 100.0
        provisional_score = None

    return ScoreSummary(
        final_score=(
            minimum
            if evaluated_count == len(statuses) and review_count == 0
            else None
        ),
        minimum_score=minimum,
        maximum_score=maximum,
        provisional_score=provisional_score,
        provisional_minimum_score=provisional_minimum,
        provisional_maximum_score=provisional_maximum,
        evaluated_count=evaluated_count,
        total_count=len(statuses),
    )
