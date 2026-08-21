from pydantic import ValidationError


def test_time_range_rejects_reverse_bounds() -> None:
    from medical_evaluation.domain import TimeRange

    try:
        TimeRange(start_sec=4.0, end_sec=3.0)
    except ValidationError:
        return
    raise AssertionError("reverse time range must fail")


def test_checkpoint_status_has_four_explicit_states() -> None:
    from medical_evaluation.domain import CheckpointStatus

    assert {item.value for item in CheckpointStatus} == {
        "correct",
        "incorrect",
        "incomplete",
        "needs_review",
    }
