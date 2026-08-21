from __future__ import annotations

from medical_evaluation.judges.base import (
    JudgeDecision,
    correct,
    decide_boolean_rules,
    incorrect,
    needs_review,
)


def judge_cp01(
    features: dict[str, float | bool | None],
    thresholds: dict[str, float],
) -> JudgeDecision:
    distance = features.get("mark_reference_distance")
    if distance is None:
        return needs_review(
            "cp_01",
            features,
            reason_code="missing_required_evidence",
            reason="未能同时识别打孔标记和参考牙位。",
        )
    if float(distance) > thresholds["max_mark_distance"]:
        return incorrect(
            "cp_01",
            features,
            reason_code="mark_position_incorrect",
            reason="打孔标记偏离目标牙位。",
            suggestion="重新确认 36 牙对应的橡皮布打孔点位。",
        )
    return correct(
        "cp_01",
        features,
        matched_rules=["mark_at_reference_position"],
        reason="打孔标记与目标牙位一致。",
    )


def judge_cp02(
    features: dict[str, float | bool | None],
    thresholds: dict[str, float],
) -> JudgeDecision:
    selected = features.get("selected_hole_index")
    residue_present = features.get("residue_present")
    if selected is None or residue_present is None:
        return needs_review(
            "cp_02",
            features,
            reason_code="missing_required_evidence",
            reason="孔盘位置或残留物证据缺失。",
        )
    if int(selected) != int(thresholds["expected_hole_index"]):
        return incorrect(
            "cp_02",
            features,
            reason_code="wrong_punch_hole",
            reason="未选择孔盘从左向右第二个孔洞。",
            suggestion="打孔前确认并使用孔盘从左向右第二个孔洞。",
        )
    if residue_present is True:
        residue_cleared = features.get("residue_cleared")
        if residue_cleared is None:
            return needs_review(
                "cp_02",
                features,
                reason_code="missing_residue_cleaning_evidence",
                reason="发现残留物，但无法确认是否已用探针清理。",
            )
        if residue_cleared is not True:
            return incorrect(
                "cp_02",
                features,
                reason_code="residue_not_cleared",
                reason="孔洞内有绿色橡皮布残留但未完成清理。",
                suggestion="发现残留时先用探针清理孔洞，再进行打孔。",
            )
    rules = ["second_punch_hole_selected"]
    if residue_present is True:
        rules.append("residue_cleared")
    return correct(
        "cp_02",
        features,
        matched_rules=rules,
        reason="孔盘位置选择正确，残留处理符合要求。",
    )


def judge_cp03(
    features: dict[str, float | bool | None],
    _thresholds: dict[str, float],
) -> JudgeDecision:
    return decide_boolean_rules(
        "cp_03",
        features,
        {
            "punch_contact": "punch_contact",
            "released_from_tip": "released_from_tip",
            "reverse_motion": "up_down_release_motion",
        },
    )


judge_cp01.checkpoint_id = "cp_01"  # type: ignore[attr-defined]
judge_cp02.checkpoint_id = "cp_02"  # type: ignore[attr-defined]
judge_cp03.checkpoint_id = "cp_03"  # type: ignore[attr-defined]
