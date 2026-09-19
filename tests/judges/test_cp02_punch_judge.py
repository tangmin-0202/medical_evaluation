import pytest

from medical_evaluation.judges import cp02_punch as judge


@pytest.mark.parametrize("changes,status,reason", [
    ({}, "correct", "criteria_satisfied"),
    ({"selected_second_largest": False, "cleanup_contact_observed": True}, "incorrect", "wrong_punch_hole_selected"),
    ({"residue_before": True, "cleanup_contact_observed": True, "residue_after": False}, "incorrect", "residue_not_cleaned"),
    ({"residue_before": True, "cleanup_contact_observed": False}, "incorrect", "residue_not_cleaned"),
    ({"residue_before": True, "cleanup_contact_observed": True, "residue_after": True}, "incorrect", "residue_not_cleaned"),
    ({"residue_before": True, "cleanup_contact_observed": None, "residue_after": None}, "incorrect", "residue_not_cleaned"),
    ({"selected_second_largest": None}, "needs_review", "unreliable_hole_selection"),
    ({"residue_before": None}, "needs_review", "unreliable_residue_evidence"),
    ({"stage_scan_reliable": False}, "needs_review", "unreliable_stage_scan"),
    ({"punch_action_observed": False}, "correct", "criteria_satisfied"),
])
def test_cp02_uses_only_final_prepunch_selection_and_green_state(changes, status, reason):
    features = {"stage_scan_reliable": True,
                "selected_second_largest": True, "residue_before": False,
                "cleanup_contact_observed": False, "residue_after": None}
    features.update(changes)
    decision = judge.judge_cp02_punch(features, {})
    assert decision.status == status
    assert decision.reason_code == reason
