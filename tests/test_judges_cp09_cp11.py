from __future__ import annotations

from medical_evaluation.judges.cp09_cp11 import judge_cp09, judge_cp10, judge_cp11

HEAD_THRESHOLDS = {
    "min_frame_stable_duration_sec": 1.0,
    "min_frame_center_x_ratio": 0.0,
    "max_frame_center_x_ratio": 1.0,
    "min_frame_center_y_ratio": 0.0,
    "max_frame_center_y_ratio": 1.0,
    "min_frame_scale_ratio": 0.01,
    "max_frame_scale_ratio": 1.0,
    "max_abs_frame_angle_deg": 90.0,
}


def _head_cp09(**updates: float | bool | None) -> dict[str, float | bool | None]:
    features: dict[str, float | bool | None] = {
        "head_registration_reliable": True,
        "head_valid_count": 8.0,
        "frame_valid_count": 8.0,
        "frame_presence_ratio": 0.8,
        "frame_present_at_end": True,
        "frame_stable_duration_sec": 3.0,
        "frame_center_x_ratio": 0.5,
        "frame_center_y_ratio": 0.5,
        "frame_scale_ratio": 0.2,
        "frame_angle_deg": 5.0,
        "head_template_compatible": True,
    }
    features.update(updates)
    return features


def test_cp09_head_relative_distinguishes_not_attempted_and_failed_installation() -> None:
    absent = judge_cp09(
        _head_cp09(
            head_registration_reliable=False,
            head_valid_count=2.0,
            frame_valid_count=0.0,
        ),
        HEAD_THRESHOLDS,
    )
    lost = judge_cp09(_head_cp09(frame_present_at_end=False), HEAD_THRESHOLDS)

    assert (absent.status.value, absent.reason_code) == ("incomplete", "frame_not_observed")
    assert (lost.status.value, lost.reason_code) == ("incorrect", "frame_missing_at_stage_end")


def test_cp09_head_relative_accepts_stable_template_geometry() -> None:
    result = judge_cp09(_head_cp09(), HEAD_THRESHOLDS)

    assert result.status.value == "correct"
    assert result.matched_rules == ["frame_present_at_end", "frame_stable", "frame_geometry_valid"]


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
            "dam_final_presence_ratio": 0.8,
            "dam_area_ratio": 0.4,
            "nose_overlap": None,
            "visible_frame_area_ratio": 0.0,
        },
        {
            "min_stage_dam_presence_ratio": 0.05,
            "min_final_dam_presence_ratio": 0.5,
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
        "min_final_dam_presence_ratio": 0.5,
        "min_dam_area_ratio": 0.20,
        "max_nose_overlap": 0.02,
        "max_visible_frame_area_ratio": 0.005,
    }

    passing = judge_cp11(
        {
            "dam_stage_presence_ratio": 0.5,
            "dam_final_presence_ratio": 0.8,
            "dam_area_ratio": 0.4,
            "nose_overlap": 0.01,
            "visible_frame_area_ratio": 0.0,
        },
        thresholds,
    )
    failing = judge_cp11(
        {
            "dam_stage_presence_ratio": 0.5,
            "dam_final_presence_ratio": 0.8,
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


def test_cp11_text_mode_requires_no_visible_white_frame_and_clear_nose() -> None:
    thresholds = {
        "min_stage_dam_presence_ratio": 0.05,
        "min_final_dam_presence_ratio": 0.5,
        "max_visible_frame_ratio": 0.01,
        "max_nose_overlap": 0.02,
    }
    features = {
        "dam_stage_presence_ratio": 1.0,
        "dam_final_presence_ratio": 1.0,
        "head_registration_reliable": True,
        "direct_white_frame_search": True,
        "visible_frame_ratio": 0.05,
        "nose_overlap": 0.0,
        "visible_frame_area_ratio": 0.05,
    }

    passing = judge_cp11(features | {"visible_frame_ratio": 0.0}, thresholds)
    failing = judge_cp11(features, thresholds)

    assert passing.status.value == "correct"
    assert passing.matched_rules == ["frame_covered", "nose_clear"]
    assert (failing.status.value, failing.reason_code) == (
        "incorrect",
        "final_position_incorrect",
    )


def test_cp11_text_mode_does_not_require_cp09_frame_coverage() -> None:
    thresholds = {
        "min_stage_dam_presence_ratio": 0.05,
        "min_final_dam_presence_ratio": 0.5,
        "max_visible_frame_ratio": 0.01,
        "max_nose_overlap": 0.02,
    }
    features = {
        "dam_stage_presence_ratio": 1.0,
        "dam_final_presence_ratio": 1.0,
        "direct_white_frame_search": True,
        "head_registration_reliable": True,
        "visible_frame_ratio": 0.0,
        "nose_overlap": 0.0,
    }

    decision = judge_cp11(features, thresholds)

    assert decision.status.value == "correct"
    assert decision.reason_code == "criteria_satisfied"


def test_cp10_without_floss_is_incomplete() -> None:
    decision = judge_cp10(
        {
            "tooth_anchor_reliable": True,
            "floss_observed_frame_count": 0.0,
            "upper_contact_frame_count": 0.0,
            "lower_contact_frame_count": 0.0,
        },
        {"min_contact_frames_per_side": 2.0},
    )

    assert (decision.status.value, decision.reason_code) == (
        "incomplete",
        "dental_floss_not_observed",
    )


def test_cp10_one_sided_contact_is_incorrect() -> None:
    decision = judge_cp10(
        {
            "tooth_anchor_reliable": True,
            "floss_observed_frame_count": 4.0,
            "upper_contact_frame_count": 3.0,
            "lower_contact_frame_count": 0.0,
        },
        {"min_contact_frames_per_side": 2.0},
    )

    assert (decision.status.value, decision.reason_code) == (
        "incorrect",
        "floss_contact_incomplete",
    )


def test_cp10_two_sided_contact_is_correct() -> None:
    decision = judge_cp10(
        {
            "tooth_anchor_reliable": True,
            "floss_observed_frame_count": 8.0,
            "upper_contact_frame_count": 2.0,
            "lower_contact_frame_count": 3.0,
        },
        {"min_contact_frames_per_side": 2.0},
    )

    assert (decision.status.value, decision.reason_code) == (
        "correct",
        "criteria_satisfied",
    )
