from __future__ import annotations

from medical_evaluation.judges.base import (
    JudgeDecision,
    correct,
    decide_boolean_rules,
    incorrect,
    needs_review,
)


def judge_cp04(
    features: dict[str, float | bool | None],
    thresholds: dict[str, float],
) -> JudgeDecision:
    similarity = features.get("clamp_reference_similarity")
    semantic = features.get("semantic_confirmed")
    if similarity is None or semantic is None:
        return needs_review(
            "cp_04",
            features,
            reason_code="missing_required_evidence",
            reason="障夹外观或语义证据不足。",
        )
    if float(similarity) < thresholds["min_clamp_similarity"] or semantic is not True:
        return incorrect(
            "cp_04",
            features,
            reason_code="wrong_clamp_type",
            reason="所选障夹与下颌磨牙障夹不匹配。",
            suggestion="选择适用于下颌磨牙的橡皮障夹。",
        )
    return correct(
        "cp_04",
        features,
        matched_rules=["mandibular_molar_clamp"],
        reason="所选障夹符合下颌磨牙障夹特征。",
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
