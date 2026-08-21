from medical_evaluation.judges.cp09_cp11 import judge_cp09, judge_cp11


def test_cp09_marks_centered_frame_correct() -> None:
    result = judge_cp09({"frame_center_offset": 0.03}, {"max_center_offset": 0.08})

    assert result.status.value == "correct"
    assert result.matched_rules == ["frame_centered"]


def test_cp09_marks_large_offset_incorrect() -> None:
    result = judge_cp09({"frame_center_offset": 0.2}, {"max_center_offset": 0.08})

    assert result.status.value == "incorrect"
    assert result.reason_code == "frame_not_centered"


def test_cp11_requests_review_when_face_region_is_missing() -> None:
    result = judge_cp11(
        {"mouth_nose_visible": False, "face_overlap": None, "frame_coverage": 0.9},
        {"max_face_overlap": 0.02, "min_frame_coverage": 0.85},
    )

    assert result.status.value == "needs_review"
    assert result.reason_code == "face_region_not_visible"


def test_cp11_passes_only_when_both_criteria_pass() -> None:
    thresholds = {"max_face_overlap": 0.02, "min_frame_coverage": 0.85}

    passing = judge_cp11(
        {"mouth_nose_visible": True, "face_overlap": 0.01, "frame_coverage": 0.9},
        thresholds,
    )
    failing = judge_cp11(
        {"mouth_nose_visible": True, "face_overlap": 0.08, "frame_coverage": 0.9},
        thresholds,
    )

    assert passing.status.value == "correct"
    assert passing.matched_rules == ["face_clear", "dam_spread_on_frame"]
    assert failing.status.value == "incorrect"
