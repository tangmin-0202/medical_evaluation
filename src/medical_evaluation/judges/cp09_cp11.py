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
    if "head_registration_reliable" in features:
        return _judge_cp09_head_relative(features, thresholds)
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


def _judge_cp09_head_relative(
    features: dict[str, float | bool | None],
    thresholds: dict[str, float],
) -> JudgeDecision:
    head_count = float(features.get("head_valid_count") or 0.0)
    if head_count <= 0.0:
        return needs_review(
            "cp_09",
            features,
            reason_code="unreliable_head_registration",
            reason="无法可靠建立头模参照坐标。",
        )
    if float(features.get("frame_valid_count") or 0.0) == 0.0:
        return incomplete(
            "cp_09",
            features,
            reason_code="frame_not_observed",
            reason="本阶段未观察到支架，安装未完成。",
            suggestion="完成支架安装，并在阶段结束前保持支架在位。",
        )
    if features.get("head_registration_reliable") is not True:
        return needs_review(
            "cp_09",
            features,
            reason_code="unreliable_head_registration",
            reason="无法可靠建立头模参照坐标。",
        )
    if features.get("frame_present_at_end") is not True:
        return incorrect(
            "cp_09",
            features,
            reason_code="frame_missing_at_stage_end",
            reason="支架曾出现，但阶段结束时未保持在位。",
            suggestion="重新安装并固定支架，确认阶段结束时仍稳定在位。",
        )
    duration = features.get("frame_stable_duration_sec")
    if duration is None or float(duration) < thresholds["min_frame_stable_duration_sec"]:
        return incorrect(
            "cp_09",
            features,
            reason_code="frame_not_stabilized",
            reason="支架在阶段结束前未形成足够长的稳定状态。",
            suggestion="调整支架后短暂停留，确认位置不再持续变化。",
        )
    if features.get("head_template_compatible") is not True:
        return needs_review(
            "cp_09",
            features,
            reason_code="head_template_incompatible",
            reason="当前头模视角无法可靠套用已校准的位置模板。",
        )
    names = (
        "frame_center_x_ratio",
        "frame_center_y_ratio",
        "frame_scale_ratio",
        "frame_angle_deg",
    )
    if any(features.get(name) is None for name in names):
        return needs_review(
            "cp_09",
            features,
            reason_code="missing_frame_geometry",
            reason="支架位置、尺度或方向证据不足。",
        )
    x = float(features["frame_center_x_ratio"])  # type: ignore[arg-type]
    y = float(features["frame_center_y_ratio"])  # type: ignore[arg-type]
    scale = float(features["frame_scale_ratio"])  # type: ignore[arg-type]
    angle = float(features["frame_angle_deg"])  # type: ignore[arg-type]
    valid = (
        thresholds["min_frame_center_x_ratio"] <= x <= thresholds["max_frame_center_x_ratio"]
        and thresholds["min_frame_center_y_ratio"] <= y <= thresholds["max_frame_center_y_ratio"]
        and thresholds["min_frame_scale_ratio"] <= scale <= thresholds["max_frame_scale_ratio"]
        and abs(angle) <= thresholds["max_abs_frame_angle_deg"]
    )
    if not valid:
        return incorrect(
            "cp_09",
            features,
            reason_code="frame_geometry_out_of_range",
            reason="支架相对头模的位置、尺度或方向超出允许范围。",
            suggestion="以头模为参照重新调整支架的居中位置和方向。",
        )
    return correct(
        "cp_09",
        features,
        matched_rules=["frame_present_at_end", "frame_stable", "frame_geometry_valid"],
        reason="支架在阶段结束时稳定在位，且相对头模的位置符合模板。",
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
    final_presence = features.get("dam_final_presence_ratio")
    if final_presence is None:
        return needs_review(
            "cp_11",
            features,
            reason_code="unreliable_final_presence",
            reason="无法可靠判断末尾是否仍有橡皮布。",
        )
    if float(final_presence) < thresholds["min_final_dam_presence_ratio"]:
        return incorrect(
            "cp_11",
            features,
            reason_code="rubber_dam_missing_at_end",
            reason="阶段中出现过橡皮布，但末尾已不在位。",
            matched_rules=["dam_present_at_end"],
            suggestion="重新完成末尾调整，确保橡皮布最终保持撑开并覆盖支架。",
        )
    if "expected_frame_dam_coverage_ratio" in features:
        return _judge_cp11_head_relative(features, thresholds)
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


def _judge_cp11_head_relative(
    features: dict[str, float | bool | None],
    thresholds: dict[str, float],
) -> JudgeDecision:
    if features.get("frame_reference_available") is not True:
        return needs_review(
            "cp_11",
            features,
            reason_code="missing_cp09_frame_reference",
            reason="缺少 CP09 保存的实际支架参考。",
        )
    if features.get("head_registration_reliable") is not True:
        return needs_review(
            "cp_11",
            features,
            reason_code="unreliable_head_registration",
            reason="无法可靠地将 CP09 支架和鼻部模板映射到 CP11。",
        )
    coverage = features.get("expected_frame_dam_coverage_ratio")
    visible_frame = features.get("visible_frame_ratio")
    nose_overlap = features.get("nose_overlap")
    if coverage is None or visible_frame is None or nose_overlap is None:
        return needs_review(
            "cp_11",
            features,
            reason_code="missing_required_evidence",
            reason="支架覆盖或鼻部无遮挡证据不足。",
        )
    failed: list[str] = []
    if float(visible_frame) > thresholds["max_visible_frame_ratio"]:
        failed.append("frame_covered")
    if float(nose_overlap) > thresholds["max_nose_overlap"]:
        failed.append("nose_clear")
    if failed:
        return incorrect(
            "cp_11",
            features,
            reason_code="final_position_incorrect",
            reason="橡皮布最终未同时满足覆盖支架和鼻部无遮挡。",
            matched_rules=failed,
            suggestion="调整橡皮布游离缘，使其覆盖支架，同时保持鼻部完全暴露。",
        )
    return correct(
        "cp_11",
        features,
        matched_rules=["frame_covered", "nose_clear"],
        reason="橡皮布已覆盖支架，且鼻部无遮挡。",
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
