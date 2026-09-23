from __future__ import annotations

from pathlib import Path

import pytest

from medical_evaluation.judges.cp01_cp03 import judge_cp01, judge_cp03
from medical_evaluation.judges.cp02_punch import judge_cp02_punch
from medical_evaluation.judges.cp04_cp06 import judge_cp04, judge_cp05, judge_cp06
from medical_evaluation.judges.cp07_cp08 import judge_cp07, judge_cp08
from medical_evaluation.judges.cp09_cp11 import judge_cp09, judge_cp10, judge_cp11
from medical_evaluation.pipeline import JUDGES
from medical_evaluation.rubric import load_rubric


@pytest.fixture(scope="module")
def thresholds() -> dict[str, dict[str, float]]:
    rubric = load_rubric(Path(__file__).parents[1] / "config" / "rubric.yaml")
    return {checkpoint.id: checkpoint.thresholds for checkpoint in rubric.checkpoints}


CASES = [
    (
        judge_cp01,
        {
            "dam_valid_frame_count": 10.0,
            "pen_presence_detected": True,
            "mark_candidate_count": 1.0,
            "mark_reference_distance": 0.02,
        },
        {
            "dam_valid_frame_count": 10.0,
            "pen_presence_detected": True,
            "mark_candidate_count": 1.0,
            "mark_reference_distance": 0.2,
        },
    ),
    (
        judge_cp02_punch,
        {
            "stage_scan_reliable": True,
            "punch_action_observed": True,
            "selected_second_largest": True,
            "residue_before": False,
            "cleanup_contact_observed": None,
            "residue_after": False,
        },
        {
            "stage_scan_reliable": True,
            "punch_action_observed": True,
            "selected_second_largest": False,
            "residue_before": False,
            "cleanup_contact_observed": None,
            "residue_after": False,
        },
    ),
    (
        judge_cp03,
        {"punch_contact": True, "released_from_tip": True, "reverse_motion": True},
        {"punch_contact": True, "released_from_tip": False, "reverse_motion": True},
    ),
    (
        judge_cp04,
        {
            "clamp_observed": True,
            "shape_evidence_reliable": True,
            "clear_frame_count": 3.0,
            "matching_frame_count": 3.0,
            "clamp_reference_similarity": 0.9,
            "evidence_consistent": True,
        },
        {
            "clamp_observed": True,
            "shape_evidence_reliable": True,
            "clear_frame_count": 3.0,
            "matching_frame_count": 0.0,
            "clamp_reference_similarity": 0.5,
            "evidence_consistent": True,
        },
    ),
    (
        judge_cp05,
        {"wing_below_dam": True, "jaw_arm_bow_above": True},
        {"wing_below_dam": False, "jaw_arm_bow_above": True},
    ),
    (
        judge_cp06,
        {"opening_increase": True, "spring_backward": True},
        {"opening_increase": True, "spring_backward": False},
    ),
    (
        judge_cp07,
        {"target_tooth_correct": True, "contact_stable": True, "distal_bow_visible": True},
        {"target_tooth_correct": False, "contact_stable": True, "distal_bow_visible": True},
    ),
    (
        judge_cp08,
        {
            "instrument_observed": True,
            "instrument_shape_reliable": True,
            "instrument_shape_match": True,
            "final_state_observable": True,
            "rubber_dam_positioned": True,
            "left_wing_complete": True,
            "right_wing_complete": True,
            "left_wing_hole_detected": True,
            "right_wing_hole_detected": True,
            "left_wing_hole_dam_color_ratio": 0.9,
            "right_wing_hole_dam_color_ratio": 0.9,
            "left_wing_hole_non_dam_color_ratio": 0.05,
            "right_wing_hole_non_dam_color_ratio": 0.05,
            "left_wing_hole_valid_frame_count": 3.0,
            "right_wing_hole_valid_frame_count": 3.0,
        },
        {
            "instrument_observed": True,
            "instrument_shape_reliable": True,
            "instrument_shape_match": False,
        },
    ),
    (judge_cp09, {"frame_oral_center_offset": 0.03}, {"frame_oral_center_offset": 0.6}),
    (
        judge_cp10,
        {
            "tooth_anchor_reliable": True,
            "floss_observed_frame_count": 4.0,
            "upper_contact_frame_count": 2.0,
            "lower_contact_frame_count": 2.0,
        },
        {
            "tooth_anchor_reliable": True,
            "floss_observed_frame_count": 4.0,
            "upper_contact_frame_count": 2.0,
            "lower_contact_frame_count": 0.0,
        },
    ),
    (
        judge_cp11,
        {
            "dam_stage_presence_ratio": 0.5,
            "dam_final_presence_ratio": 0.8,
            "dam_area_ratio": 0.4,
            "nose_overlap": 0.0,
            "visible_frame_area_ratio": 0.0,
        },
        {
            "dam_stage_presence_ratio": 0.5,
            "dam_final_presence_ratio": 0.8,
            "dam_area_ratio": 0.1,
            "nose_overlap": 0.0,
            "visible_frame_area_ratio": 0.0,
        },
    ),
]


@pytest.mark.parametrize(("judge", "passing", "failing"), CASES)
def test_each_judge_has_positive_and_negative_cases(
    judge,
    passing: dict[str, float | bool | None],
    failing: dict[str, float | bool | None],
    thresholds: dict[str, dict[str, float]],
) -> None:
    checkpoint_id = judge.checkpoint_id

    assert judge(passing, thresholds[checkpoint_id]).status.value == "correct"
    assert judge(failing, thresholds[checkpoint_id]).status.value == "incorrect"


@pytest.mark.parametrize(("judge", "passing", "_failing"), CASES)
def test_each_judge_requests_review_for_missing_evidence(
    judge,
    passing: dict[str, float | bool | None],
    _failing: dict[str, float | bool | None],
    thresholds: dict[str, dict[str, float]],
) -> None:
    missing = dict(passing)
    first_key = next(iter(missing))
    missing[first_key] = None

    assert judge(missing, thresholds[judge.checkpoint_id]).status.value == "needs_review"


def test_cp02_requires_cleaning_only_when_residue_is_present(
    thresholds: dict[str, dict[str, float]],
) -> None:
    result = judge_cp02_punch(
        {
            "stage_scan_reliable": True,
            "punch_action_observed": True,
            "selected_second_largest": True,
            "residue_before": True,
            "cleanup_contact_observed": False,
            "residue_after": True,
        },
        thresholds["cp_02"],
    )

    assert result.status.value == "incorrect"
    assert result.reason_code == "residue_not_cleaned"


CP04_PASSING_FEATURES: dict[str, float | bool | None] = {
    "clamp_observed": True,
    "shape_evidence_reliable": True,
    "clear_frame_count": 3.0,
    "matching_frame_count": 3.0,
    "clamp_reference_similarity": 0.9,
    "evidence_consistent": True,
}


@pytest.mark.parametrize(
    ("overrides", "status", "reason_code"),
    [
        ({"clamp_observed": False}, "incomplete", "clamp_not_observed"),
        (
            {"shape_evidence_reliable": False, "clear_frame_count": 0.0},
            "needs_review",
            "unreliable_clamp_shape",
        ),
        (
            {"evidence_consistent": False},
            "needs_review",
            "inconsistent_clamp_evidence",
        ),
        (
            {"matching_frame_count": 0.0, "clamp_reference_similarity": 0.5},
            "incorrect",
            "wrong_clamp_type",
        ),
        (
            {
                "clear_frame_count": 1.0,
                "matching_frame_count": 0.0,
                "clamp_reference_similarity": 0.5,
            },
            "incorrect",
            "wrong_clamp_type",
        ),
        (
            {
                "clear_frame_count": 1.0,
                "matching_frame_count": 1.0,
                "clamp_reference_similarity": 0.9,
            },
            "correct",
            "criteria_satisfied",
        ),
        ({}, "correct", "criteria_satisfied"),
    ],
)
def test_cp04_uses_displayed_reference_shape_state_machine(
    overrides: dict[str, float | bool | None],
    status: str,
    reason_code: str,
    thresholds: dict[str, dict[str, float]],
) -> None:
    result = judge_cp04(
        {**CP04_PASSING_FEATURES, **overrides},
        thresholds["cp_04"],
    )

    assert result.status.value == status
    assert result.reason_code == reason_code


CP08_PASSING_FEATURES: dict[str, float | bool | None] = {
    "instrument_observed": True,
    "instrument_shape_reliable": True,
    "instrument_shape_match": True,
    "final_state_observable": True,
    "rubber_dam_segmentation_conflict": False,
    "rubber_dam_positioned": True,
    "left_wing_complete": True,
    "right_wing_complete": True,
    "left_wing_hole_detected": True,
    "right_wing_hole_detected": True,
    "left_wing_hole_dam_color_ratio": 0.9,
    "right_wing_hole_dam_color_ratio": 0.9,
    "left_wing_hole_non_dam_color_ratio": 0.05,
    "right_wing_hole_non_dam_color_ratio": 0.05,
    "left_wing_hole_valid_frame_count": 3.0,
    "right_wing_hole_valid_frame_count": 3.0,
}


@pytest.mark.parametrize(
    ("overrides", "status", "reason_code"),
    [
        ({"instrument_observed": False}, "incomplete", "cp08_not_performed"),
        (
            {"instrument_shape_reliable": False},
            "needs_review",
            "unreliable_instrument_shape",
        ),
        (
            {"instrument_shape_match": False},
            "incorrect",
            "wrong_instrument_shape",
        ),
        (
            {"final_state_observable": False},
            "needs_review",
            "final_state_unobservable",
        ),
        (
            {
                "rubber_dam_segmentation_conflict": True,
                "rubber_dam_positioned": None,
            },
            "needs_review",
            "rubber_dam_segmentation_unreliable",
        ),
        (
            {"rubber_dam_positioned": False},
            "incorrect",
            "rubber_dam_not_positioned",
        ),
        (
            {"left_wing_complete": False},
            "incorrect",
            "clamp_wing_not_fully_visible",
        ),
        (
            {"left_wing_hole_dam_color_ratio": 0.2},
            "incorrect",
            "non_dam_color_under_wing_hole",
        ),
        (
            {"right_wing_hole_detected": False},
            "needs_review",
            "unreliable_wing_hole_color",
        ),
        ({}, "correct", "criteria_satisfied"),
    ],
)
def test_cp08_uses_ordered_instrument_and_two_hole_state_machine(
    overrides: dict[str, float | bool | None],
    status: str,
    reason_code: str,
    thresholds: dict[str, dict[str, float]],
) -> None:
    features = {**CP08_PASSING_FEATURES, **overrides}

    result = judge_cp08(features, thresholds["cp_08"])

    assert result.status.value == status
    assert result.reason_code == reason_code


def test_cp08_final_unobservable_precedes_segmentation_conflict(
    thresholds: dict[str, dict[str, float]],
) -> None:
    features = {
        **CP08_PASSING_FEATURES,
        "final_state_observable": False,
        "rubber_dam_segmentation_conflict": True,
        "rubber_dam_positioned": None,
    }

    result = judge_cp08(features, thresholds["cp_08"])

    assert result.status.value == "needs_review"
    assert result.reason_code == "final_state_unobservable"


def test_cp08_missing_conflict_feature_keeps_backward_compatible_result(
    thresholds: dict[str, dict[str, float]],
) -> None:
    features = dict(CP08_PASSING_FEATURES)
    del features["rubber_dam_segmentation_conflict"]

    result = judge_cp08(features, thresholds["cp_08"])

    assert result.status.value == "correct"


def test_cp08_checks_each_wing_hole_non_dam_ratio_independently(
    thresholds: dict[str, dict[str, float]],
) -> None:
    features = {
        **CP08_PASSING_FEATURES,
        "right_wing_hole_non_dam_color_ratio": 0.8,
    }

    result = judge_cp08(features, thresholds["cp_08"])

    assert result.status.value == "incorrect"
    assert result.reason_code == "non_dam_color_under_wing_hole"


def test_cp08_keeps_explicit_one_hole_failure_when_other_hole_is_unreliable(
    thresholds: dict[str, dict[str, float]],
) -> None:
    features = {
        **CP08_PASSING_FEATURES,
        "left_wing_hole_dam_color_ratio": 0.2,
        "right_wing_hole_detected": False,
        "right_wing_hole_dam_color_ratio": None,
        "right_wing_hole_non_dam_color_ratio": None,
        "right_wing_hole_valid_frame_count": 0.0,
    }

    result = judge_cp08(features, thresholds["cp_08"])

    assert result.status.value == "incorrect"
    assert result.reason_code == "non_dam_color_under_wing_hole"


@pytest.mark.parametrize(
    "field",
    [
        "left_wing_hole_dam_color_ratio",
        "right_wing_hole_dam_color_ratio",
        "left_wing_hole_non_dam_color_ratio",
        "right_wing_hole_non_dam_color_ratio",
    ],
)
@pytest.mark.parametrize(
    "invalid_ratio",
    [float("nan"), float("inf"), float("-inf"), -0.1, 1.1],
)
def test_cp08_requests_review_for_invalid_per_side_color_ratios(
    field: str,
    invalid_ratio: float,
    thresholds: dict[str, dict[str, float]],
) -> None:
    features = {**CP08_PASSING_FEATURES, field: invalid_ratio}

    result = judge_cp08(features, thresholds["cp_08"])

    assert result.status.value == "needs_review"
    assert result.reason_code == "unreliable_wing_hole_color"


@pytest.mark.parametrize(
    "field",
    [
        "left_wing_hole_valid_frame_count",
        "right_wing_hole_valid_frame_count",
    ],
)
@pytest.mark.parametrize(
    "invalid_count",
    [float("nan"), float("inf"), float("-inf"), -1.0],
)
def test_cp08_requests_review_for_invalid_wing_hole_frame_counts(
    field: str,
    invalid_count: float,
    thresholds: dict[str, dict[str, float]],
) -> None:
    features = {**CP08_PASSING_FEATURES, field: invalid_count}

    result = judge_cp08(features, thresholds["cp_08"])

    assert result.status.value == "needs_review"
    assert result.reason_code == "unreliable_wing_hole_color"


@pytest.mark.parametrize(
    "field",
    [
        "left_wing_hole_dam_color_ratio",
        "right_wing_hole_dam_color_ratio",
        "left_wing_hole_non_dam_color_ratio",
        "right_wing_hole_non_dam_color_ratio",
    ],
)
@pytest.mark.parametrize("invalid_ratio", [False, True])
def test_cp08_rejects_boolean_wing_hole_color_ratios(
    field: str,
    invalid_ratio: bool,
    thresholds: dict[str, dict[str, float]],
) -> None:
    features = {**CP08_PASSING_FEATURES, field: invalid_ratio}

    result = judge_cp08(features, thresholds["cp_08"])

    assert result.status.value == "needs_review"
    assert result.reason_code == "unreliable_wing_hole_color"


@pytest.mark.parametrize(
    "field",
    [
        "left_wing_hole_valid_frame_count",
        "right_wing_hole_valid_frame_count",
    ],
)
@pytest.mark.parametrize("invalid_count", [False, True, 3.5])
def test_cp08_requires_integer_non_boolean_wing_hole_frame_counts(
    field: str,
    invalid_count: float | bool,
    thresholds: dict[str, dict[str, float]],
) -> None:
    features = {**CP08_PASSING_FEATURES, field: invalid_count}
    permissive_thresholds = {
        **thresholds["cp_08"],
        "min_wing_hole_valid_frames": 0.0,
    }

    result = judge_cp08(features, permissive_thresholds)

    assert result.status.value == "needs_review"
    assert result.reason_code == "unreliable_wing_hole_color"


def test_pipeline_registers_all_11_judges() -> None:
    assert list(JUDGES) == [f"cp_{number:02d}" for number in range(1, 12)]
