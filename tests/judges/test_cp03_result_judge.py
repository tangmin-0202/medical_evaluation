import pytest

from medical_evaluation.judges import cp03_hole as judge


@pytest.mark.parametrize("changes,status,reason", [
    ({}, "correct", "criteria_satisfied"),
    ({"hole_observed": False}, "incomplete", "hole_not_formed"),
    ({"final_scan_reliable": False}, "needs_review", "unreliable_final_scan"),
    ({"hole_observed": None}, "needs_review", "unreliable_hole_observation"),
    ({"hole_clear_consecutive_frames": 1}, "needs_review", "insufficient_clear_hole_frames"),
    ({"hole_complete": False}, "incorrect", "hole_result_incorrect"),
    ({"hole_round": False}, "incorrect", "hole_result_incorrect"),
    ({"hole_adhesion_free": False}, "incorrect", "hole_result_incorrect"),
    ({"hole_round": None}, "needs_review", "unreliable_hole_quality"),
    ({"cp01_correct": False, "cp02_correct": False, "reverse_motion": False}, "correct", "criteria_satisfied"),
])
def test_result_only_cp03(changes, status, reason):
    features = {"final_scan_reliable": True, "hole_observed": True,
                "hole_clear_consecutive_frames": 3, "hole_complete": True,
                "hole_round": True, "hole_adhesion_free": True}
    features.update(changes)
    result = judge.judge_cp03_hole(features, {"min_clear_hole_frames": 3})
    assert result.status == status
    assert result.reason_code == reason
