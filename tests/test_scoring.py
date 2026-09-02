from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from medical_evaluation.domain import CheckpointStatus


def test_known_results_produce_final_score() -> None:
    from medical_evaluation.scoring import aggregate_equal_weight_score

    statuses = [CheckpointStatus.CORRECT] * 8 + [CheckpointStatus.INCORRECT] * 3
    result = aggregate_equal_weight_score(statuses)

    assert result.final_score == pytest.approx(72.72727272727273)
    assert result.minimum_score == result.maximum_score == result.final_score


def test_review_item_produces_range_and_no_final_score() -> None:
    from medical_evaluation.scoring import aggregate_equal_weight_score

    statuses = (
        [CheckpointStatus.CORRECT] * 8
        + [CheckpointStatus.INCORRECT] * 2
        + [CheckpointStatus.NEEDS_REVIEW]
    )
    result = aggregate_equal_weight_score(statuses)

    assert result.final_score is None
    assert result.minimum_score == pytest.approx(72.72727272727273)
    assert result.maximum_score == pytest.approx(81.81818181818181)


def test_score_requires_exactly_11_statuses() -> None:
    from medical_evaluation.scoring import aggregate_equal_weight_score

    with pytest.raises(ValueError, match="exactly 11 statuses"):
        aggregate_equal_weight_score([CheckpointStatus.CORRECT] * 10)


def test_two_evaluated_items_produce_provisional_score_only() -> None:
    from medical_evaluation.scoring import aggregate_evaluation_score

    statuses = [CheckpointStatus.NEEDS_REVIEW] * 11
    statuses[8] = CheckpointStatus.CORRECT
    statuses[10] = CheckpointStatus.INCORRECT
    included = [False] * 11
    included[8] = included[10] = True

    result = aggregate_evaluation_score(statuses, included)

    assert result.final_score is None
    assert result.evaluated_count == 2
    assert result.total_count == 11
    assert result.provisional_score == pytest.approx(50.0)
    assert result.provisional_minimum_score == pytest.approx(50.0)
    assert result.provisional_maximum_score == pytest.approx(50.0)


def test_one_evaluated_correct_item_is_provisional_100() -> None:
    from medical_evaluation.scoring import aggregate_evaluation_score

    statuses = [CheckpointStatus.NEEDS_REVIEW] * 11
    statuses[10] = CheckpointStatus.CORRECT
    included = [False] * 11
    included[10] = True

    result = aggregate_evaluation_score(statuses, included)

    assert result.evaluated_count == 1
    assert result.provisional_score == pytest.approx(100.0)


def test_evaluated_review_item_produces_provisional_range() -> None:
    from medical_evaluation.scoring import aggregate_evaluation_score

    statuses = [CheckpointStatus.NEEDS_REVIEW] * 11
    statuses[8] = CheckpointStatus.CORRECT
    statuses[10] = CheckpointStatus.NEEDS_REVIEW
    included = [False] * 11
    included[8] = included[10] = True

    result = aggregate_evaluation_score(statuses, included)

    assert result.provisional_score is None
    assert result.provisional_minimum_score == pytest.approx(50.0)
    assert result.provisional_maximum_score == pytest.approx(100.0)


def test_report_derives_summary_from_unique_checkpoint_results() -> None:
    from medical_evaluation.reporting import CheckpointResult, EvaluationReport, RunAudit

    checkpoints = [
        CheckpointResult(
            checkpoint_id=f"cp_{number:02d}",
            status=CheckpointStatus.CORRECT,
            confidence=0.9,
            reason_code="criterion_passed",
        )
        for number in range(1, 12)
    ]
    audit = RunAudit(
        rubric_version="2026-08-21",
        model_versions={"judge": "rules-v1"},
        started_at=datetime.now(UTC),
    )
    report = EvaluationReport(job_id="job-1", video_id="success", checkpoints=checkpoints, audit=audit)

    assert report.summary.final_score == pytest.approx(100.0)
    assert report.summary.evaluated_count == 11
    assert report.summary.provisional_score == pytest.approx(100.0)
    assert report.checkpoints[0].score == pytest.approx(100.0 / 11.0)

    with pytest.raises(ValidationError, match="11 unique checkpoint results"):
        EvaluationReport(
            job_id="job-2",
            video_id="success",
            checkpoints=checkpoints[:-1] + [checkpoints[0]],
            audit=audit,
        )
