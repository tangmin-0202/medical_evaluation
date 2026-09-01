from medical_evaluation.judges.cp09_cp11 import judge_cp09, judge_cp11


def test_cp09_marks_frame_centered_on_oral_region_correct() -> None:
    result = judge_cp09(
        {"frame_oral_center_offset": 0.03},
        {"max_oral_center_offset": 0.08},
    )

    assert result.status.value == "correct"
    assert result.matched_rules == ["frame_centered_on_oral_region"]


def test_cp09_marks_large_oral_relative_offset_incorrect() -> None:
    result = judge_cp09(
        {"frame_oral_center_offset": 0.2},
        {"max_oral_center_offset": 0.08},
    )

    assert result.status.value == "incorrect"
    assert result.reason_code == "frame_not_centered_on_oral_region"


def test_cp09_requests_review_without_paired_evidence() -> None:
    result = judge_cp09(
        {"frame_oral_center_offset": None, "relative_offset_valid_count": 2.0},
        {"max_oral_center_offset": 0.08},
    )

    assert result.status.value == "needs_review"
    assert result.reason_code == "missing_frame_oral_evidence"
    assert "口腔参考区域" in result.reason


def test_cp11_requests_review_when_final_nose_evidence_is_missing() -> None:
    result = judge_cp11(
        {
            "dam_stage_presence_ratio": 0.5,
            "dam_area_ratio": 0.4,
            "nose_overlap": None,
            "visible_frame_area_ratio": 0.0,
        },
        {
            "min_stage_dam_presence_ratio": 0.05,
            "min_dam_area_ratio": 0.20,
            "max_nose_overlap": 0.02,
            "max_visible_frame_area_ratio": 0.005,
        },
    )

    assert result.status.value == "needs_review"
    assert result.reason_code == "missing_required_evidence"


def test_cp11_passes_only_when_all_final_criteria_pass() -> None:
    thresholds = {
        "min_stage_dam_presence_ratio": 0.05,
        "min_dam_area_ratio": 0.20,
        "max_nose_overlap": 0.02,
        "max_visible_frame_area_ratio": 0.005,
    }

    passing = judge_cp11(
        {
            "dam_stage_presence_ratio": 0.5,
            "dam_area_ratio": 0.4,
            "nose_overlap": 0.01,
            "visible_frame_area_ratio": 0.0,
        },
        thresholds,
    )
    failing = judge_cp11(
        {
            "dam_stage_presence_ratio": 0.5,
            "dam_area_ratio": 0.4,
            "nose_overlap": 0.08,
            "visible_frame_area_ratio": 0.0,
        },
        thresholds,
    )

    assert passing.status.value == "correct"
    assert passing.matched_rules == [
        "dam_area_sufficient",
        "nose_clear",
        "frame_covered",
    ]
    assert failing.status.value == "incorrect"
