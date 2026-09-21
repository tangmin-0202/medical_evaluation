import pytest

from medical_evaluation.judges import cp03_hole as judge


@pytest.mark.parametrize("changes,status,reason", [
    ({}, "correct", "criteria_satisfied"),
    ({"hole_observed": False}, "incomplete", "hole_not_formed"),
    ({"final_scan_reliable": False}, "needs_review", "unreliable_final_scan"),
    ({"hole_observed": None}, "needs_review", "unreliable_hole_observation"),
    ({"hole_clear_consecutive_frames": 0}, "needs_review", "insufficient_clear_hole_frames"),
    ({"hole_complete": False}, "correct", "criteria_satisfied"),
    ({"hole_round": False}, "correct", "criteria_satisfied"),
    ({"hole_adhesion_free": False}, "incorrect", "hole_adhesion_detected"),
    ({"hole_adhesion_free": None}, "needs_review", "unreliable_adhesion_observation"),
    ({"cp01_correct": False, "cp02_correct": False, "reverse_motion": False}, "correct", "criteria_satisfied"),
])
def test_result_only_cp03(changes, status, reason):
    features = {"final_scan_reliable": True, "hole_observed": True,
                "hole_clear_consecutive_frames": 3, "hole_complete": True,
                "hole_round": True, "hole_adhesion_free": True}
    features.update(changes)
    result = judge.judge_cp03_hole(features, {"min_clear_hole_frames": 1})
    assert result.status == status
    assert result.reason_code == reason


def test_cp03_features_need_only_adhesion_quality() -> None:
    result = judge.judge_cp03_hole(
        {
            "final_scan_reliable": True,
            "hole_observed": True,
            "hole_clear_consecutive_frames": 1,
            "hole_adhesion_free": True,
        },
        {"min_clear_hole_frames": 1},
    )

    assert result.status == "correct"
    assert result.matched_rules == ["observed_hole_without_adhesion"]
