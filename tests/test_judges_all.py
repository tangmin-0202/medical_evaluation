from __future__ import annotations

from pathlib import Path

import pytest

from medical_evaluation.judges.cp01_cp03 import judge_cp01, judge_cp02, judge_cp03
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
            "pen_contact_detected": True,
            "new_mark_candidate_count": 1.0,
            "mark_reference_distance": 0.02,
        },
        {
            "dam_valid_frame_count": 10.0,
            "pen_contact_detected": True,
            "new_mark_candidate_count": 1.0,
            "mark_reference_distance": 0.2,
        },
    ),
    (
        judge_cp02,
        {"selected_hole_index": 2, "residue_present": False, "residue_cleared": None},
        {"selected_hole_index": 1, "residue_present": False, "residue_cleared": None},
    ),
    (
        judge_cp03,
        {"punch_contact": True, "released_from_tip": True, "reverse_motion": True},
        {"punch_contact": True, "released_from_tip": False, "reverse_motion": True},
    ),
    (
        judge_cp04,
        {"clamp_reference_similarity": 0.9, "semantic_confirmed": True},
        {"clamp_reference_similarity": 0.5, "semantic_confirmed": True},
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
            "blunt_tool_used": True,
            "wings_exposed": True,
            "neck_enclosed": True,
            "green_wing_hole_ratio": 0.01,
        },
        {
            "blunt_tool_used": False,
            "wings_exposed": True,
            "neck_enclosed": True,
            "green_wing_hole_ratio": 0.01,
        },
    ),
    (judge_cp09, {"frame_oral_center_offset": 0.03}, {"frame_oral_center_offset": 0.6}),
    (
        judge_cp10,
        {"mesial_crossing": True, "distal_crossing": True},
        {"mesial_crossing": True, "distal_crossing": False},
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
    result = judge_cp02(
        {"selected_hole_index": 2, "residue_present": True, "residue_cleared": False},
        thresholds["cp_02"],
    )

    assert result.status.value == "incorrect"
    assert result.reason_code == "residue_not_cleared"


def test_pipeline_registers_all_11_judges() -> None:
    assert list(JUDGES) == [f"cp_{number:02d}" for number in range(1, 12)]
