from __future__ import annotations

from medical_evaluation.judges.base import (
    JudgeDecision,
    correct,
    decide_boolean_rules,
    incomplete,
    incorrect,
    needs_review,
)


def judge_cp04(
    features: dict[str, float | bool | None],
    thresholds: dict[str, float],
) -> JudgeDecision:
    observed = features.get("clamp_observed")
    reliable = features.get("shape_evidence_reliable")
    clear_count = features.get("clear_frame_count")
    matching_count = features.get("matching_frame_count")
    similarity = features.get("clamp_reference_similarity")
    consistent = features.get("evidence_consistent")
    if observed is None:
        return needs_review(
            "cp_04",
            features,
            reason_code="missing_required_evidence",
            reason="无法确定学员是否展示了所选障夹。",
        )
    if observed is not True:
        return incomplete(
            "cp_04",
            features,
            reason_code="clamp_not_observed",
            reason="本阶段未观察到学员清晰展示所选障夹。",
            suggestion="选择障夹后，请在白色手套掌心上清晰展示其完整外形。",
        )
    if reliable is not True or clear_count is None or float(clear_count) < thresholds[
        "min_clear_clamp_frames"
    ]:
        return needs_review(
            "cp_04",
            features,
            reason_code="unreliable_clamp_shape",
            reason="观察到障夹，但清晰完整的外形证据不足。",
        )
    if consistent is not True:
        return needs_review(
            "cp_04",
            features,
            reason_code=(
                "missing_required_evidence"
                if consistent is None
                else "inconsistent_clamp_evidence"
            ),
            reason="不同清晰帧中的障夹外形证据不一致。",
        )
    if similarity is None or matching_count is None:
        return needs_review(
            "cp_04",
            features,
            reason_code="missing_required_evidence",
            reason="缺少与 success 参考障夹进行外形比较的证据。",
        )
    if (
        float(similarity) < thresholds["min_clamp_similarity"]
        or float(matching_count) < thresholds["min_matching_clamp_frames"]
    ):
        return incorrect(
            "cp_04",
            features,
            reason_code="wrong_clamp_type",
            reason="清晰展示的障夹外形与 success 参考障夹不匹配。",
            suggestion="请重新选择与 success 参考外形一致的橡皮障夹。",
        )
    return correct(
        "cp_04",
        features,
        matched_rules=["success_reference_clamp_shape"],
        reason="所选障夹在旋转、尺度和翻面归一化后与 success 参考外形一致。",
    )


def judge_cp05(
    features: dict[str, float | bool | None],
    _thresholds: dict[str, float],
) -> JudgeDecision:
    return decide_boolean_rules(
        "cp_05",
        features,
        {
            "wing_below_dam": "wing_below_dam",
            "jaw_arm_bow_above": "jaw_arm_bow_above_dam",
        },
    )


def judge_cp06(
    features: dict[str, float | bool | None],
    _thresholds: dict[str, float],
) -> JudgeDecision:
    return decide_boolean_rules(
        "cp_06",
        features,
        {
            "opening_increase": "clamp_opened",
            "spring_backward": "spring_locked_backward",
        },
    )


judge_cp04.checkpoint_id = "cp_04"  # type: ignore[attr-defined]
judge_cp05.checkpoint_id = "cp_05"  # type: ignore[attr-defined]
judge_cp06.checkpoint_id = "cp_06"  # type: ignore[attr-defined]
