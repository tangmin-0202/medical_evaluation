from __future__ import annotations

from math import isfinite

from medical_evaluation.judges.base import (
    JudgeDecision,
    correct,
    decide_boolean_rules,
    incomplete,
    incorrect,
    needs_review,
)


def _valid_ratio(value: float | bool | None) -> bool:
    if value is None or isinstance(value, bool):
        return False
    numeric = float(value)
    return isfinite(numeric) and 0.0 <= numeric <= 1.0


def _valid_frame_count(value: float | bool | None, minimum: float) -> bool:
    if value is None or isinstance(value, bool):
        return False
    numeric = float(value)
    return (
        isfinite(numeric)
        and numeric >= 0.0
        and numeric.is_integer()
        and numeric >= minimum
    )


def judge_cp07(
    features: dict[str, float | bool | None],
    _thresholds: dict[str, float],
) -> JudgeDecision:
    return decide_boolean_rules(
        "cp_07",
        features,
        {
            "target_tooth_correct": "correct_target_tooth",
            "contact_stable": "four_point_contact_stable",
            "distal_bow_visible": "bow_faces_distal",
        },
    )


def judge_cp08(
    features: dict[str, float | bool | None],
    thresholds: dict[str, float],
) -> JudgeDecision:
    instrument_observed = features.get("instrument_observed")
    if instrument_observed is False:
        return incomplete(
            "cp_08",
            features,
            reason_code="cp08_not_performed",
            reason="CP08 阶段未观察到约定的器具操作。",
            suggestion="请使用符合约定外形的器具完成橡皮布就位操作。",
        )
    if instrument_observed is not True or features.get("instrument_shape_reliable") is not True:
        return needs_review(
            "cp_08",
            features,
            reason_code="unreliable_instrument_shape",
            reason="器具候选存在，但长柄、细杆和弯曲工作端的外形证据不可靠。",
        )
    instrument_shape_match = features.get("instrument_shape_match")
    if instrument_shape_match is None:
        return needs_review(
            "cp_08",
            features,
            reason_code="unreliable_instrument_shape",
            reason="器具外形无法可靠判定。",
        )
    if instrument_shape_match is not True:
        return incorrect(
            "cp_08",
            features,
            reason_code="wrong_instrument_shape",
            reason="观察到的器具不符合长金属柄、细工作杆和弯曲工作端的约定外形。",
            suggestion="请使用符合约定外形的器具完成橡皮布就位操作。",
        )

    if features.get("final_state_observable") is not True:
        return needs_review(
            "cp_08",
            features,
            reason_code="final_state_unobservable",
            reason="CP09 末尾目标区域不可可靠观察。",
        )

    rubber_dam_positioned = features.get("rubber_dam_positioned")
    if rubber_dam_positioned is None:
        return needs_review(
            "cp_08",
            features,
            reason_code="final_state_unobservable",
            reason="CP09 末尾无法可靠判断橡皮布是否在位。",
        )
    if rubber_dam_positioned is not True:
        return incorrect(
            "cp_08",
            features,
            reason_code="rubber_dam_not_positioned",
            reason="CP09 末尾确认橡皮布未在目标区域就位。",
            suggestion="请完成橡皮布就位后再进入后续操作。",
        )

    wing_values = (
        features.get("left_wing_complete"),
        features.get("right_wing_complete"),
    )
    if any(value is False for value in wing_values):
        return incorrect(
            "cp_08",
            features,
            reason_code="clamp_wing_not_fully_visible",
            reason="橡皮障夹至少一侧翼部未完整露出。",
            suggestion="请调整橡皮布，使障夹左右翼部都完整露出。",
        )
    if any(value is not True for value in wing_values):
        return needs_review(
            "cp_08",
            features,
            reason_code="final_state_unobservable",
            reason="无法可靠判断橡皮障夹左右翼部是否完整露出。",
        )

    min_valid_frames = thresholds["min_wing_hole_valid_frames"]
    sides = ("left", "right")
    hole_reliable: dict[str, bool] = {}
    for side in sides:
        valid_frame_count = features.get(f"{side}_wing_hole_valid_frame_count")
        dam_color_ratio = features.get(f"{side}_wing_hole_dam_color_ratio")
        non_dam_color_ratio = features.get(f"{side}_wing_hole_non_dam_color_ratio")
        hole_reliable[side] = (
            features.get(f"{side}_wing_hole_detected") is True
            and _valid_frame_count(valid_frame_count, min_valid_frames)
            and _valid_ratio(dam_color_ratio)
            and _valid_ratio(non_dam_color_ratio)
        )
    non_dam_color_present = any(
        hole_reliable[side]
        and (
            float(features[f"{side}_wing_hole_dam_color_ratio"])
            < thresholds["min_wing_hole_dam_color_ratio"]
            or float(features[f"{side}_wing_hole_non_dam_color_ratio"])
            > thresholds["max_wing_hole_non_dam_color_ratio"]
        )
        for side in sides
    )
    if non_dam_color_present:
        return incorrect(
            "cp_08",
            features,
            reason_code="non_dam_color_under_wing_hole",
            reason="至少一个障夹翼孔内不是与同帧橡皮布一致的颜色。",
            suggestion="请调整橡皮布，使左右翼孔内都能看到橡皮布。",
        )
    if not all(hole_reliable.values()):
        return needs_review(
            "cp_08",
            features,
            reason_code="unreliable_wing_hole_color",
            reason="无法同时可靠定位左右翼孔并判断孔内颜色。",
        )

    return correct(
        "cp_08",
        features,
        matched_rules=[
            "instrument_shape_match",
            "rubber_dam_positioned",
            "left_wing_complete",
            "right_wing_complete",
            "left_wing_hole_dam_color_match",
            "right_wing_hole_dam_color_match",
        ],
        reason="观察到符合约定外形的器具，且两侧障夹翼部完整、翼孔内均为橡皮布同源颜色。",
    )


judge_cp07.checkpoint_id = "cp_07"  # type: ignore[attr-defined]
judge_cp08.checkpoint_id = "cp_08"  # type: ignore[attr-defined]
