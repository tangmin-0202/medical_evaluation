"""Template-based CP01 Judge (not activated until real visual gate passes)."""

import math

from medical_evaluation.judges.base import (
    JudgeDecision,
    correct,
    incomplete,
    incorrect,
    needs_review,
)


def judge_cp01_template(
    features: dict[str, float | bool | None], thresholds: dict[str, float],
) -> JudgeDecision:
    if features.get("stage_scan_reliable") is not True:
        return needs_review("cp_01", features, reason_code="unreliable_stage_scan",
                            reason="阶段覆盖或分割不可靠，不能据此认定未标记。")
    if features.get("pen_dam_contact_observed") is not True:
        return incomplete("cp_01", features, reason_code="marking_contact_not_observed",
                          reason="可靠阶段扫描中未观察到笔尖到达橡皮布。",
                          suggestion="请用标记笔在橡皮布上完成目标牙位标记。")
    if features.get("template_reference_reliable") is not True:
        return needs_review("cp_01", features, reason_code="unreliable_template_reference",
                            reason="模板方向、黑点布局或当前帧配准不可靠。")
    if features.get("new_ink_evidence_reliable") is not True:
        return needs_review("cp_01", features, reason_code="unreliable_new_ink_evidence",
                            reason="无法可靠区分新增笔迹与透出的模板点。")
    count = features.get("new_mark_count")
    if count is None or not math.isfinite(float(count)) or float(count) < 0:
        return needs_review("cp_01", features, reason_code="unreliable_new_ink_evidence",
                            reason="新增笔迹数量证据缺失或无效。")
    if count == 0:
        return incorrect("cp_01", features, reason_code="new_mark_not_observed",
                         reason="观察到笔尖接触，但可靠后续画面未发现新增标记。",
                         suggestion="在目标牙位留下清晰且可确认的笔迹。")
    distance = features.get("mark_distance_in_dot_spacings")
    if distance is None or not math.isfinite(float(distance)) or float(distance) < 0:
        return needs_review("cp_01", features, reason_code="unreliable_mark_distance",
                            reason="无法可靠计算标记与目标模板点的距离。")
    limit = thresholds.get("max_mark_distance_in_dot_spacings")
    if limit is None or not math.isfinite(limit) or limit <= 0:
        raise ValueError("a calibrated positive mark-distance threshold is required")
    if float(distance) > limit:
        return incorrect("cp_01", features, reason_code="mark_position_incorrect",
                         reason="新增标记偏离模板右下牙弓从上往下第三点。",
                         suggestion="按模板右下牙弓从上往下第三点重新定位标记。")
    return correct("cp_01", features, matched_rules=["new_mark_at_template_target"],
                   reason="观察到标记操作，新增笔迹位于目标模板点允许范围内。")


judge_cp01_template.checkpoint_id = "cp_01"  # type: ignore[attr-defined]
