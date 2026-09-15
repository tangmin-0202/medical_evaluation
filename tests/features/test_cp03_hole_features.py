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
