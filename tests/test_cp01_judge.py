from medical_evaluation.judges.cp01_cp03 import judge_cp01

THRESHOLDS = {"max_mark_distance": 0.05}


def test_insufficient_dam_frames_need_review_before_contact_state() -> None:
    result = judge_cp01(
        {"dam_valid_frame_count": 2.0, "pen_contact_detected": False},
        THRESHOLDS,
    )

    assert result.status.value == "needs_review"
    assert result.reason_code == "missing_required_evidence"


def test_no_stable_pen_contact_is_incomplete() -> None:
    result = judge_cp01(
        {"dam_valid_frame_count": 10.0, "pen_contact_detected": False},
        THRESHOLDS,
    )

    assert result.status.value == "incomplete"
    assert result.reason_code == "marking_pen_contact_not_observed"


def test_contact_without_new_mark_is_incorrect() -> None:
    result = judge_cp01(
        {
            "dam_valid_frame_count": 10.0,
            "pen_contact_detected": True,
            "new_mark_candidate_count": 0.0,
            "mark_reference_distance": None,
        },
        THRESHOLDS,
    )

    assert result.status.value == "incorrect"
    assert result.reason_code == "mark_not_left_after_contact"


def test_contact_mark_near_reference_is_correct() -> None:
    result = judge_cp01(
        {
            "dam_valid_frame_count": 10.0,
            "pen_contact_detected": True,
            "new_mark_candidate_count": 1.0,
            "mark_reference_distance": 0.02,
        },
        THRESHOLDS,
    )

    assert result.status.value == "correct"


def test_contact_mark_far_from_reference_is_incorrect() -> None:
    result = judge_cp01(
        {
            "dam_valid_frame_count": 10.0,
            "pen_contact_detected": True,
            "new_mark_candidate_count": 1.0,
            "mark_reference_distance": 0.20,
        },
        THRESHOLDS,
    )

    assert result.status.value == "incorrect"
    assert result.reason_code == "mark_position_incorrect"
