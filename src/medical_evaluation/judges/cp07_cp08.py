from __future__ import annotations

from medical_evaluation.judges.base import (
    JudgeDecision,
    correct,
    decide_boolean_rules,
    incorrect,
    needs_review,
)


def judge_cp07(
    features: dict[str, float | bool | None],
    _thresholds: dict[str, float],
) -> JudgeDecision:
    return decide_boolean_rules(
        "cp_07",
        features,
        {
            "target_tooth_correct": "correct_target_tooth",
            "contact_stable": "four_point_contact_stable",
            "distal_bow_visible": "bow_faces_distal",
        },
    )


def judge_cp08(
    features: dict[str, float | bool | None],
    thresholds: dict[str, float],
) -> JudgeDecision:
    boolean_rules = {
        "blunt_tool_used": "blunt_tool_used",
        "wings_exposed": "wings_exposed",
        "neck_enclosed": "tooth_neck_enclosed",
    }
    missing = [name for name in boolean_rules if features.get(name) is None]
    ratio = features.get("green_wing_hole_ratio")
    if missing or ratio is None:
        return needs_review(
            "cp_08",
            features,
            reason_code="missing_required_evidence",
            reason="橡皮布就位所需的工具、翼部或孔隙证据不足。",
        )
    boolean_decision = decide_boolean_rules("cp_08", features, boolean_rules)
    if boolean_decision.status.value == "incorrect":
        return boolean_decision
    if float(ratio) > thresholds["max_green_wing_hole_ratio"]:
        return incorrect(
            "cp_08",
            features,
            reason_code="green_gap_around_wing",
            reason="障夹翼部周围仍可见绿色橡皮布孔隙。",
            suggestion="用钝头器械继续翻转橡皮布，使障夹翼完全暴露。",
        )
    return correct(
        "cp_08",
        features,
        matched_rules=[*boolean_rules.values(), "no_green_gap_around_wing"],
        reason="橡皮布就位和翼部暴露符合要求。",
    )


judge_cp07.checkpoint_id = "cp_07"  # type: ignore[attr-defined]
judge_cp08.checkpoint_id = "cp_08"  # type: ignore[attr-defined]
