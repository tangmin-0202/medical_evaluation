import cv2
import numpy as np
import pytest

from medical_evaluation.features import cp03_hole as hole


def test_complete_circle_metrics():
    mask = np.zeros((120, 120), np.uint8)
    cv2.circle(mask, (60, 60), 25, 1, -1)
    result = hole.measure_hole(mask)
    assert result["component_count"] == 1
    assert result["circularity"] > 0.85
    assert result["solidity"] > 0.95
    assert result["axis_ratio"] == pytest.approx(1, abs=0.05)
    assert result["touches_roi_edge"] is False


def test_elongated_hole_is_not_round():
    mask = np.zeros((120, 120), np.uint8)
    cv2.ellipse(mask, (60, 60), (30, 10), 0, 0, 360, 1, -1)
    assert hole.measure_hole(mask)["axis_ratio"] < 0.5


def test_attached_flap_reduces_solidity():
    mask = np.zeros((120, 120), np.uint8)
    cv2.circle(mask, (60, 60), 25, 1, -1)
    mask[35:61, 57:64] = 0
    assert hole.measure_hole(mask)["solidity"] < 0.95


def test_empty_and_tiny_masks_are_not_clear_holes():
    mask = np.zeros((20, 20), np.uint8)
    assert hole.measure_hole(mask) is None
    mask[5, 5] = 1
    assert hole.measure_hole(mask) is None


def test_cropped_hole_is_flagged():
    mask = np.zeros((100, 100), np.uint8)
    cv2.circle(mask, (0, 50), 20, 1, -1)
    assert hole.measure_hole(mask)["touches_roi_edge"] is True


def _dam_with_hole(*, flap: bool = False, isolated_speck: bool = False):
    image = np.zeros((160, 200, 3), np.uint8)
    image[:] = (40, 160, 40)
    dam_mask = np.ones((160, 200), np.uint8)
    cv2.circle(image, (100, 80), 24, (90, 70, 55), -1)
    cv2.circle(dam_mask, (100, 80), 24, 0, -1)
    if flap:
        cv2.rectangle(image, (97, 56), (103, 79), (40, 160, 40), -1)
        cv2.rectangle(dam_mask, (97, 56), (103, 79), 1, -1)
    if isolated_speck:
        cv2.circle(image, (100, 80), 3, (40, 160, 40), -1)
        cv2.circle(dam_mask, (100, 80), 3, 1, -1)
    return image, dam_mask


def test_clear_hole_has_no_connected_adhesion() -> None:
    image, dam_mask = _dam_with_hole()

    result = hole.analyze_hole_adhesion(image, dam_mask)

    assert result.hole_observed is True
    assert result.adhesion_detected is False
    assert result.center_xy == pytest.approx((100, 80), abs=1)


def test_connected_green_flap_is_adhesion() -> None:
    image, dam_mask = _dam_with_hole(flap=True)

    result = hole.analyze_hole_adhesion(image, dam_mask)

    assert result.hole_observed is True
    assert result.adhesion_detected is True
    assert result.adhesion_area_ratio > 0.03


def test_isolated_green_speck_is_not_connected_adhesion() -> None:
    image, dam_mask = _dam_with_hole(isolated_speck=True)

    result = hole.analyze_hole_adhesion(image, dam_mask)

    assert result.hole_observed is True
    assert result.adhesion_detected is False


def test_no_internal_hole_is_reported() -> None:
    image = np.full((120, 160, 3), (40, 160, 40), np.uint8)
    dam_mask = np.ones((120, 160), np.uint8)

    result = hole.analyze_hole_adhesion(image, dam_mask)

    assert result.hole_observed is False
    assert result.reliable is True


def test_opening_clipped_by_frame_edge_is_unreliable() -> None:
    image = np.full((120, 160, 3), (40, 160, 40), np.uint8)
    dam_mask = np.ones((120, 160), np.uint8)
    cv2.circle(image, (0, 60), 24, (90, 70, 55), -1)
    cv2.circle(dam_mask, (0, 60), 24, 0, -1)

    result = hole.analyze_hole_adhesion(image, dam_mask)

    assert result.hole_observed is False
