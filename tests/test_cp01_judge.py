from medical_evaluation.judges.cp01_cp03 import judge_cp01

THRESHOLDS = {"max_mark_distance": 0.05}


def test_clear_stage_with_no_stable_mark_is_incomplete() -> None:
    result = judge_cp01(
        {
            "mark_reference_distance": None,
            "mark_missing": True,
            "mark_selection_ambiguous": False,
            "dam_valid_frame_count": 8.0,
        },
        THRESHOLDS,
    )

    assert result.status.value == "incomplete"
    assert result.reason_code == "punch_mark_not_observed"


def test_ambiguous_stable_points_require_review() -> None:
    result = judge_cp01(
        {
            "mark_reference_distance": None,
            "mark_missing": False,
            "mark_selection_ambiguous": True,
            "dam_valid_frame_count": 8.0,
        },
        THRESHOLDS,
    )

    assert result.status.value == "needs_review"
    assert result.reason_code == "ambiguous_punch_candidates"
