"""Independent CP02 punch Judge, pending real segmentation gate."""

from medical_evaluation.judges.base import (
    JudgeDecision,
    correct,
    incomplete,
    incorrect,
    needs_review,
)


def judge_cp02_punch(
    features: dict[str, float | bool | None], thresholds: dict[str, float],
) -> JudgeDecision:
    if features.get("stage_scan_reliable") is not True:
        return needs_review("cp_02", features, reason_code="unreliable_stage_scan",
                            reason="阶段覆盖或分割不可靠。")
    action = features.get("punch_action_observed")
    if action is False:
        return incomplete("cp_02", features, reason_code="punch_action_not_observed",
                          reason="可靠扫描中未观察到打孔操作。", suggestion="请完成橡皮布打孔。")
    if action is not True:
        return needs_review("cp_02", features, reason_code="unreliable_stage_scan",
                            reason="打孔动作证据缺失。")
    selected = features.get("selected_second_largest")
    if selected is False:
        return incorrect("cp_02", features, reason_code="wrong_punch_hole_selected",
                         reason="冲针实际对准的不是孔盘第二大孔，清理不能弥补选孔错误。",
                         suggestion="打孔前确认冲针对准第二大孔。")
    if selected is not True:
        return needs_review("cp_02", features, reason_code="unreliable_hole_selection",
                            reason="孔径排序或最终冲针对孔关系不可靠。")
    residue = features.get("residue_before")
    if residue is not True and residue is not False:
        return needs_review("cp_02", features, reason_code="unreliable_residue_evidence",
                            reason="无法确认选中孔内是否有绿色橡皮布残留。")
    if residue:
        contact = features.get("cleanup_contact_observed")
        if contact is False:
            return incorrect("cp_02", features, reason_code="residue_not_cleaned",
                             reason="选中孔有绿色残留，但未观察到对该孔的清理。",
                             suggestion="清理选中孔内残留并检查孔内状态。")
        if contact is not True or features.get("residue_after") not in (True, False):
            return needs_review("cp_02", features, reason_code="unreliable_cleanup_result",
                                reason="清理接触或清理后孔内状态不可靠。")
        if features.get("residue_after") is True:
            return incorrect("cp_02", features, reason_code="residue_remaining",
                             reason="观察到清理动作，但绿色残留仍在。",
                             suggestion="继续清理直至孔内无橡皮布残留。")
    return correct("cp_02", features,
                   matched_rules=["second_largest_hole_selected", "residue_absent_or_removed"],
                   reason="实际选中第二大孔，孔内无残留或已完成清理。")


judge_cp02_punch.checkpoint_id = "cp_02"  # type: ignore[attr-defined]
