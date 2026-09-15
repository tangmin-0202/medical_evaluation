import pytest

from medical_evaluation.judges import cp01_template as judge


@pytest.mark.parametrize("changes,status,reason", [
    ({}, "correct", "criteria_satisfied"),
    ({"stage_scan_reliable": False}, "needs_review", "unreliable_stage_scan"),
    ({"pen_dam_contact_observed": False}, "incomplete", "marking_contact_not_observed"),
    ({"template_reference_reliable": False}, "needs_review", "unreliable_template_reference"),
    ({"new_ink_evidence_reliable": False}, "needs_review", "unreliable_new_ink_evidence"),
    ({"new_mark_count": 0}, "incorrect", "new_mark_not_observed"),
    ({"mark_distance_in_dot_spacings": 0.8}, "incorrect", "mark_position_incorrect"),
    ({"mark_distance_in_dot_spacings": None}, "needs_review", "unreliable_mark_distance"),
    ({"mark_distance_in_dot_spacings": float("nan")}, "needs_review", "unreliable_mark_distance"),
    ({"new_mark_count": 2}, "correct", "criteria_satisfied"),
])
def test_template_judge_requires_contact_and_reliable_new_ink(changes, status, reason):
    features = {
        "stage_scan_reliable": True,
        "pen_dam_contact_observed": True,
        "template_reference_reliable": True,
        "new_ink_evidence_reliable": True,
        "new_mark_count": 1,
        "mark_distance_in_dot_spacings": 0.1,
    }
    features.update(changes)
    decision = judge.judge_cp01_template(features, {"max_mark_distance_in_dot_spacings": 0.3})
    assert decision.status == status
    assert decision.reason_code == reason
