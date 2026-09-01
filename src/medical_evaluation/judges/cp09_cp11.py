from __future__ import annotations

from medical_evaluation.judges.base import (
    JudgeDecision,
    correct,
    decide_boolean_rules,
    incomplete,
    incorrect,
    needs_review,
)


def judge_cp09(
    features: dict[str, float | bool | None],
    thresholds: dict[str, float],
) -> JudgeDecision:
    offset = features.get("frame_oral_center_offset")
    if offset is None:
        return needs_review(
            "cp_09",
            features,
            reason_code="missing_frame_oral_evidence",
            reason="未能稳定获得支架相对口腔参考区域的位置。",
        )
    if float(offset) > thresholds["max_oral_center_offset"]:
        return incorrect(
            "cp_09",
            features,
            reason_code="frame_not_centered_on_oral_region",
            reason="支架中心相对口腔参考区域偏离允许范围。",
            suggestion="安装支架后，以整个口腔参考区域为参照调整支架位置。",
        )
    return correct(
        "cp_09",
        features,
        matched_rules=["frame_centered_on_oral_region"],
        reason="支架中心相对口腔参考区域居中。",
    )


def judge_cp11(
    features: dict[str, float | bool | None],
    thresholds: dict[str, float],
) -> JudgeDecision:
    presence = features.get("dam_stage_presence_ratio")
    if presence is None:
        return needs_review(
            "cp_11",
            features,
            reason_code="unreliable_stage_presence",
            reason="无法可靠判断本阶段是否出现过橡皮布。",
        )
    if float(presence) < thresholds["min_stage_dam_presence_ratio"]:
        return incomplete(
            "cp_11",
            features,
            reason_code="rubber_dam_not_observed",
            reason="本阶段未观察到橡皮布，操作未完成。",
            suggestion="请完成橡皮布调整并将其充分撑开至支架。",
        )
    dam_area = features.get("dam_area_ratio")
    nose_overlap = features.get("nose_overlap")
    visible_frame = features.get("visible_frame_area_ratio")
    if dam_area is None or nose_overlap is None or visible_frame is None:
        return needs_review(
            "cp_11",
            features,
            reason_code="missing_required_evidence",
            reason="末尾橡皮布、鼻部或支架可见性证据不足。",
        )
    failed_rules: list[str] = []
    if float(dam_area) < thresholds["min_dam_area_ratio"]:
        failed_rules.append("dam_area_sufficient")
    if float(nose_overlap) > thresholds["max_nose_overlap"]:
        failed_rules.append("nose_clear")
    if float(visible_frame) > thresholds["max_visible_frame_area_ratio"]:
        failed_rules.append("frame_covered")
    if failed_rules:
        return incorrect(
            "cp_11",
            features,
            reason_code="final_position_incorrect",
            reason="橡皮布最终位置未同时满足口鼻无遮挡和充分撑开。",
            matched_rules=failed_rules,
            suggestion="重新调整橡皮布游离缘，露出口鼻并均匀撑至支架。",
        )
    return correct(
        "cp_11",
        features,
        matched_rules=["dam_area_sufficient", "nose_clear", "frame_covered"],
        reason="橡皮布未遮挡口鼻，且已充分撑开至支架。",
    )


def judge_cp10(
    features: dict[str, float | bool | None],
    _thresholds: dict[str, float],
) -> JudgeDecision:
    return decide_boolean_rules(
        "cp_10",
        features,
        {
            "mesial_crossing": "mesial_contact_crossed",
            "distal_crossing": "distal_contact_crossed",
        },
    )


judge_cp09.checkpoint_id = "cp_09"  # type: ignore[attr-defined]
judge_cp10.checkpoint_id = "cp_10"  # type: ignore[attr-defined]
judge_cp11.checkpoint_id = "cp_11"  # type: ignore[attr-defined]
