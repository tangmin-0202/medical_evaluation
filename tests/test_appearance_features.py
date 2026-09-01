from __future__ import annotations

import cv2
import numpy as np
import pytest

from medical_evaluation.features.appearance import (
    box_overlap_ratio,
    green_dam_area_ratio,
    visible_reference_area_ratio,
)


def test_green_dam_area_ratio_reports_confident_absence_and_half_frame() -> None:
    black = np.zeros((20, 20, 3), dtype=np.uint8)
    half_green = black.copy()
    half_green[:, :10] = (30, 170, 90)

    assert green_dam_area_ratio(black) == 0.0
    assert green_dam_area_ratio(half_green) == pytest.approx(0.5)


def test_box_overlap_is_normalized_by_box_area() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:10] = True

    assert box_overlap_ratio(mask, (0.25, 0.25, 0.75, 0.75)) == pytest.approx(0.5)


def test_green_covered_frame_candidate_is_not_visible_white_frame() -> None:
    white = np.full((20, 20, 3), 240, dtype=np.uint8)
    reference_lab = cv2.cvtColor(white, cv2.COLOR_BGR2LAB)[0, 0].astype(float)
    candidate = np.zeros((20, 20), dtype=bool)
    candidate[5:15, 5:15] = True
    green = np.full((20, 20, 3), (30, 170, 90), dtype=np.uint8)

    assert visible_reference_area_ratio(
        white,
        candidate,
        reference_lab,
        max_lab_distance=10,
    ) == pytest.approx(0.25)
    assert visible_reference_area_ratio(
        green,
        candidate,
        reference_lab,
        max_lab_distance=10,
    ) == 0.0
