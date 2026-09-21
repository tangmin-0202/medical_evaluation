"""Result-only CP03 Judge; no release-motion or prior CP score dependencies."""

import math

from medical_evaluation.judges.base import (
    JudgeDecision,
    correct,
    incomplete,
    incorrect,
    needs_review,
)


def judge_cp03_hole(
    features: dict[str, float | bool | None], thresholds: dict[str, float],
) -> JudgeDecision:
    if features.get("final_scan_reliable") is not True:
        return needs_review("cp_03", features, reason_code="unreliable_final_scan",
                            reason="打孔后橡皮布未获得可靠可见的扫描证据。")
    observed = features.get("hole_observed")
    if observed is False:
        return incomplete("cp_03", features, reason_code="hole_not_formed",
                          reason="可靠清晰画面确认橡皮布未形成贯通孔。",
                          suggestion="完成橡皮布打孔并检查结果。")
    if observed is not True:
        return needs_review("cp_03", features, reason_code="unreliable_hole_observation",
                            reason="无法确认是否形成贯通孔，不能把小白点当作孔。")
    required = thresholds.get("min_clear_hole_frames")
    if required is None or not math.isfinite(required) or required < 2 or required != int(required):
        raise ValueError("at least two consecutive clear frames are required")
    count = features.get("hole_clear_consecutive_frames")
    if count is None or not math.isfinite(float(count)) or float(count) < required:
        return needs_review("cp_03", features, reason_code="insufficient_clear_hole_frames",
                            reason="相邻清晰孔画面不足，无法稳定判断边缘与粘连。")
    adhesion_free = features.get("hole_adhesion_free")
    if adhesion_free is not True and adhesion_free is not False:
        return needs_review(
            "cp_03",
            features,
            reason_code="unreliable_adhesion_observation",
            reason="孔边缘粘连证据不可靠。",
        )
    if adhesion_free is False:
        return incorrect(
            "cp_03",
            features,
            reason_code="hole_adhesion_detected",
            reason="目标孔边缘仍有橡皮布薄片、翻边或残留粘连。",
            suggestion="清除孔边粘连后再次检查打孔结果。",
        )
    return correct("cp_03", features,
                   matched_rules=["observed_hole_without_adhesion"],
                   reason="相邻清晰画面显示目标孔边缘无橡皮布粘连。")


judge_cp03_hole.checkpoint_id = "cp_03"  # type: ignore[attr-defined]
