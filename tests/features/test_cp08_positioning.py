from pathlib import Path

import cv2
import numpy as np
import pytest

from medical_evaluation.features.cp08_positioning import (
    measure_instrument_shape,
    measure_wing_hole_color,
    split_clamp_wings,
)


def _valid_instrument(*, extra_clamp: bool = False, merge_clamp: bool = False) -> np.ndarray:
    mask = np.zeros((240, 420), np.uint8)
    cv2.rectangle(mask, (35, 102), (225, 137), 1, -1)
    points = np.array([[225, 119], [300, 119], [340, 105], [370, 72]], np.int32)
    cv2.polylines(mask, [points], False, 1, 9)
    cv2.circle(mask, (370, 72), 5, 1, -1)
    clamp_x = 290 if merge_clamp else 80
    if extra_clamp or merge_clamp:
        cv2.ellipse(mask, (clamp_x, 185), (25, 18), 0, 15, 345, 1, 7)
        cv2.rectangle(mask, (clamp_x - 27, 163), (clamp_x - 17, 183), 1, -1)
        cv2.rectangle(mask, (clamp_x + 17, 163), (clamp_x + 27, 183), 1, -1)
    if merge_clamp:
        cv2.line(mask, (300, 122), (310, 169), 1, 7)
    return mask


def test_valid_curved_long_handle_and_thin_shaft_matches():
    result = measure_instrument_shape(_valid_instrument())

    assert result.observed is True
    assert result.reliable is True
    assert result.matches_proxy is True
    assert result.elongation > 4.0
    assert result.handle_to_shaft_width_ratio > 2.0
    assert result.curvature_deg > 15.0


def test_instrument_proxy_is_stable_across_practical_mask_scales():
    for scale in (0.5, 1.4):
        mask = cv2.resize(_valid_instrument(), None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_NEAREST)
        result = measure_instrument_shape(mask)
        assert result.reliable is True
        assert result.matches_proxy is True


def test_too_small_instrument_mask_fails_safely():
    mask = cv2.resize(_valid_instrument(), None, fx=0.04, fy=0.04,
                      interpolation=cv2.INTER_NEAREST)

    result = measure_instrument_shape(mask)

    assert result.reliable is False
    assert result.matches_proxy is None


def test_disconnected_compact_u_clamp_is_discarded_from_instrument_measurement():
    result = measure_instrument_shape(_valid_instrument(extra_clamp=True))

    assert result.reliable is True
    assert result.matches_proxy is True
    assert result.input_component_count == 2
    assert result.selected_component_area_px < int(_valid_instrument(extra_clamp=True).sum())


def test_real_gate_mask_excludes_u_clamp_and_matches_agreed_proxy():
    fixture = Path(__file__).parents[1] / "fixtures/cp08/real_gate_instrument_with_clamp.png"
    mask = cv2.imread(str(fixture), cv2.IMREAD_GRAYSCALE)

    result = measure_instrument_shape(mask)

    assert result.input_component_count == 2
    assert result.reliable is True
    assert result.matches_proxy is True
    assert result.component_mask is not None
    assert not result.component_mask[:, : mask.shape[1] // 2].any()
    assert result.component_mask[:, mask.shape[1] // 2 :].any()


def test_real_success_mask_matches_user_confirmed_instrument_shape():
    fixture = Path(__file__).parents[1] / "fixtures/cp08/real_success_instrument_148.png"
    mask = cv2.imread(str(fixture), cv2.IMREAD_GRAYSCALE)

    result = measure_instrument_shape(mask)

    assert result.reliable is True
    assert result.matches_proxy is True
    assert result.elongation is not None and result.elongation > 8.0
    assert result.handle_to_shaft_width_ratio is not None
    assert result.curvature_deg is not None


@pytest.mark.parametrize("side", ["left", "right"])
def test_real_success_filled_sam_wing_finds_green_hole_from_original(side: str):
    fixtures = Path(__file__).parents[1] / "fixtures/cp08"
    frame = cv2.imread(str(fixtures / "real_success_final_frame.jpg"))
    wing = cv2.imread(
        str(fixtures / f"real_success_{side}_wing.png"), cv2.IMREAD_GRAYSCALE
    )
    dam = cv2.imread(
        str(fixtures / "real_success_dam.png"), cv2.IMREAD_GRAYSCALE
    )

    result = measure_wing_hole_color(frame, wing, dam)

    assert result.hole_detected is True
    assert result.reliable is True
    assert result.hole_area_px >= 20
    assert result.sample_area_px >= 9
    assert result.dam_color_ratio is not None
    assert result.dam_color_ratio >= 0.7
    assert result.non_dam_color_ratio is not None
    assert result.non_dam_color_ratio <= 0.3


def test_straight_rod_is_reliable_but_does_not_match():
    mask = np.zeros((160, 420), np.uint8)
    cv2.rectangle(mask, (30, 68), (380, 91), 1, -1)

    result = measure_instrument_shape(mask)

    assert result.reliable is True
    assert result.matches_proxy is False
    assert result.curvature_deg < 10.0


def test_compact_blob_is_reliable_non_match():
    mask = np.zeros((160, 220), np.uint8)
    cv2.circle(mask, (110, 80), 35, 1, -1)

    result = measure_instrument_shape(mask)

    assert result.observed is True
    assert result.reliable is True
    assert result.matches_proxy is False


def test_two_similar_elongated_components_are_ambiguous():
    mask = np.zeros((220, 420), np.uint8)
    cv2.line(mask, (25, 60), (390, 60), 1, 13)
    cv2.line(mask, (25, 155), (390, 155), 1, 13)

    result = measure_instrument_shape(mask)

    assert result.observed is True
    assert result.reliable is False
    assert result.matches_proxy is None
    assert result.reason == "ambiguous_instrument_components"


def test_disconnected_handle_and_working_shaft_are_unreliable_fragmentation():
    mask = _valid_instrument()
    mask[:, 218:232] = 0

    result = measure_instrument_shape(mask)

    assert result.observed is True
    assert result.reliable is False
    assert result.matches_proxy is None
    assert result.reason == "fragmented_instrument_component"


def test_merged_instrument_and_clamp_is_unreliable():
    result = measure_instrument_shape(_valid_instrument(merge_clamp=True))

    assert result.observed is True
    assert result.reliable is False
    assert result.matches_proxy is None
    assert result.reason == "branched_or_merged_component"


def _tooth_and_clamp(*, left: bool = True, right: bool = True):
    tooth = np.zeros((180, 240), np.uint8)
    clamp = np.zeros_like(tooth)
    cv2.ellipse(tooth, (120, 90), (25, 35), 0, 0, 360, 1, -1)
    cv2.rectangle(clamp, (97, 70), (143, 110), 1, -1)
    if left:
        cv2.rectangle(clamp, (35, 58), (100, 122), 1, -1)
    if right:
        cv2.rectangle(clamp, (140, 58), (205, 122), 1, -1)
    return tooth, clamp


def test_split_clamp_wings_returns_both_sides_and_excludes_central_body():
    tooth, clamp = _tooth_and_clamp()

    result = split_clamp_wings(tooth, clamp)

    assert result.reliable is True
    assert result.left_wing is not None
    assert result.right_wing is not None
    assert result.left_complete is True
    assert result.right_complete is True
    assert result.left_completeness_score > 0.8
    assert result.right_completeness_score > 0.8
    assert not result.left_wing[:, 105:136].any()
    assert not result.right_wing[:, 105:136].any()


def test_split_clamp_wings_preserves_missing_side_as_none():
    tooth, clamp = _tooth_and_clamp(right=False)

    result = split_clamp_wings(tooth, clamp)

    assert result.left_wing is not None
    assert result.right_wing is None
    assert result.left_complete is True
    assert result.right_complete is False
    assert result.right_completeness_score == 0.0


def test_bilateral_truncated_stubs_are_present_but_not_complete():
    tooth = np.zeros((180, 240), np.uint8)
    clamp = np.zeros_like(tooth)
    cv2.ellipse(tooth, (120, 90), (25, 35), 0, 0, 360, 1, -1)
    cv2.rectangle(clamp, (98, 72), (142, 108), 1, -1)
    cv2.rectangle(clamp, (82, 79), (99, 101), 1, -1)
    cv2.rectangle(clamp, (141, 79), (158, 101), 1, -1)

    result = split_clamp_wings(tooth, clamp)

    assert result.left_wing is not None
    assert result.right_wing is not None
    assert result.left_complete is False
    assert result.right_complete is False
    assert result.left_completeness_score < 0.5
    assert result.right_completeness_score < 0.5


def test_split_clamp_wings_rejects_different_mask_shapes():
    tooth, clamp = _tooth_and_clamp()

    with pytest.raises(ValueError, match="matching spatial dimensions"):
        split_clamp_wings(tooth, clamp[:-1])


def _wing_scene(*, hole_bgr=(45, 155, 55), rim_bgr=(210, 210, 210), with_hole=True):
    frame = np.full((120, 120, 3), (35, 150, 45), np.uint8)
    wing = np.zeros((120, 120), np.uint8)
    dam = np.zeros((120, 120), np.uint8)
    cv2.circle(wing, (60, 60), 34, 1, -1)
    frame[wing.astype(bool)] = rim_bgr
    dam[10:30, 10:110] = 1
    if with_hole:
        cv2.circle(wing, (60, 60), 14, 0, -1)
        cv2.circle(frame, (60, 60), 14, rim_bgr, -1)
        cv2.circle(frame, (60, 60), 10, hole_bgr, -1)
    return frame, wing, dam


def test_wing_hole_color_reports_dam_matching_interior():
    frame, wing, dam = _wing_scene()

    result = measure_wing_hole_color(frame, wing, dam)

    assert result.hole_detected is True
    assert result.reliable is True
    assert result.dam_color_ratio > 0.9
    assert result.non_dam_color_ratio < 0.1


def test_each_wing_hole_color_is_measured_independently():
    green_frame, wing, dam = _wing_scene(hole_bgr=(45, 155, 55))
    red_frame, _, _ = _wing_scene(hole_bgr=(40, 40, 190))

    left = measure_wing_hole_color(green_frame, wing, dam)
    right = measure_wing_hole_color(red_frame, wing, dam)

    assert left.dam_color_ratio > 0.9
    assert right.dam_color_ratio < 0.1
    assert right.non_dam_color_ratio > 0.9


def test_red_reference_contamination_cannot_make_red_hole_match_dam_green():
    frame, wing, dam = _wing_scene(hole_bgr=(40, 40, 190))
    reference_y, reference_x = np.nonzero(dam)
    contaminated = round(len(reference_y) * 0.12)
    frame[reference_y[:contaminated], reference_x[:contaminated]] = (40, 40, 190)

    result = measure_wing_hole_color(frame, wing, dam)

    assert result.reliable is True
    assert result.dam_color_ratio < 0.1
    assert result.non_dam_color_ratio > 0.9


def test_green_shadow_variation_still_matches_same_frame_dam():
    frame, wing, dam = _wing_scene(hole_bgr=(18, 78, 24))
    reference_y, reference_x = np.nonzero(dam)
    shadowed = len(reference_y) // 3
    frame[reference_y[:shadowed], reference_x[:shadowed]] = (18, 78, 24)

    result = measure_wing_hole_color(frame, wing, dam)

    assert result.reliable is True
    assert result.dam_color_ratio > 0.9


def test_missing_wing_hole_is_unreliable():
    frame, wing, dam = _wing_scene(with_hole=False)

    result = measure_wing_hole_color(frame, wing, dam)

    assert result.hole_detected is False
    assert result.reliable is False
    assert result.dam_color_ratio is None


def test_elongated_rectangular_enclosed_gap_is_not_a_wing_hole():
    frame = np.full((120, 120, 3), (35, 150, 45), np.uint8)
    wing = np.zeros((120, 120), np.uint8)
    dam = np.zeros((120, 120), np.uint8)
    cv2.rectangle(wing, (25, 25), (95, 95), 1, -1)
    cv2.rectangle(wing, (43, 55), (77, 65), 0, -1)
    dam[5:20, 5:115] = 1

    result = measure_wing_hole_color(frame, wing, dam)

    assert result.hole_detected is False
    assert result.reliable is False
    assert result.reason == "hole_geometry_invalid"


def test_compact_square_enclosed_gaps_are_not_circular_wing_holes():
    for side in (20, 14):
        frame = np.full((120, 120, 3), (35, 150, 45), np.uint8)
        wing = np.zeros((120, 120), np.uint8)
        dam = np.zeros((120, 120), np.uint8)
        cv2.circle(wing, (60, 60), 34, 1, -1)
        half = side // 2
        cv2.rectangle(wing, (60 - half, 60 - half), (60 + half, 60 + half), 0, -1)
        dam[5:20, 5:115] = 1

        result = measure_wing_hole_color(frame, wing, dam)

        assert result.hole_detected is False
        assert result.reliable is False
        assert result.reason == "hole_geometry_invalid"


def test_inward_sampling_excludes_bright_metal_hole_rim():
    frame, wing, dam = _wing_scene(rim_bgr=(255, 255, 255))

    result = measure_wing_hole_color(frame, wing, dam)

    assert result.sample_area_px < result.hole_area_px
    assert result.dam_color_ratio > 0.9
    assert not result.sample_mask[60, 48]
    assert result.sample_mask[60, 60]
