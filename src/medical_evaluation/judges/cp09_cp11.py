from __future__ import annotations

from medical_evaluation.judges.base import JudgeDecision, correct, incorrect, needs_review


def judge_cp09(
    features: dict[str, float | bool | None],
    thresholds: dict[str, float],
) -> JudgeDecision:
    offset = features.get("frame_center_offset")
    if offset is None:
        return needs_review(
            "cp_09",
            features,
            reason_code="missing_frame_center_evidence",
            reason="未能稳定识别支架与画面中心。",
        )
    if float(offset) > thresholds["max_center_offset"]:
        return incorrect(
            "cp_09",
            features,
            reason_code="frame_not_centered",
            reason="支架中心偏离允许范围。",
            suggestion="安装支架后检查四周张力，并将支架调整至画面中央。",
        )
    return correct(
        "cp_09",
        features,
        matched_rules=["frame_centered"],
        reason="支架位置居中。",
    )


def judge_cp11(
    features: dict[str, float | bool | None],
    thresholds: dict[str, float],
) -> JudgeDecision:
    if features.get("mouth_nose_visible") is not True:
        return needs_review(
            "cp_11",
            features,
            reason_code="face_region_not_visible",
            reason="口鼻区域不可见，无法确认橡皮布是否遮挡。",
        )
    face_overlap = features.get("face_overlap")
    frame_coverage = features.get("frame_coverage")
    if face_overlap is None or frame_coverage is None:
        return needs_review(
            "cp_11",
            features,
            reason_code="missing_required_evidence",
            reason="橡皮布覆盖或撑开证据缺失。",
        )
    failed_rules: list[str] = []
    if float(face_overlap) > thresholds["max_face_overlap"]:
        failed_rules.append("face_clear")
    if float(frame_coverage) < thresholds["min_frame_coverage"]:
        failed_rules.append("dam_spread_on_frame")
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
        matched_rules=["face_clear", "dam_spread_on_frame"],
        reason="橡皮布未遮挡口鼻，且已充分撑开至支架。",
    )


judge_cp09.checkpoint_id = "cp_09"  # type: ignore[attr-defined]
judge_cp11.checkpoint_id = "cp_11"  # type: ignore[attr-defined]
