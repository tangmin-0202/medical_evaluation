"""Independent CP02 punch Judge, pending real segmentation gate."""

from medical_evaluation.judges.base import (
    JudgeDecision,
    correct,
    incorrect,
    needs_review,
)


def judge_cp02_punch(
    features: dict[str, float | bool | None], thresholds: dict[str, float],
) -> JudgeDecision:
    if features.get("stage_scan_reliable") is not True:
        return needs_review("cp_02", features, reason_code="unreliable_stage_scan",
                            reason="阶段覆盖或分割不可靠。")
    selected = features.get("selected_second_largest")
    if selected is False:
        return incorrect("cp_02", features, reason_code="wrong_punch_hole_selected",
                         reason="冲针实际对准的不是孔盘第二大孔，清理不能弥补选孔错误。",
                         suggestion="CP02结束前确认最终对准的是第二大孔。")
    if selected is not True:
        return needs_review("cp_02", features, reason_code="unreliable_hole_selection",
                            reason="孔径排序或最终冲针对孔关系不可靠。")
    residue = features.get("residue_before")
    if residue is not True and residue is not False:
        return needs_review("cp_02", features, reason_code="unreliable_residue_evidence",
                            reason="无法确认选中孔内是否有绿色橡皮布残留。")
    if residue:
        return incorrect("cp_02", features, reason_code="residue_not_cleaned",
                         reason="CP02最后可靠可见状态中，目标孔内仍有绿色橡皮布残留。",
                         suggestion="CP02结束前确认第二大孔内没有绿色残留。")
    return correct("cp_02", features,
                   matched_rules=["second_largest_hole_selected", "prepunch_residue_absent"],
                   reason="CP02最后可靠可见状态中选中第二大孔，且孔内无绿色残留。")


judge_cp02_punch.checkpoint_id = "cp_02"  # type: ignore[attr-defined]
