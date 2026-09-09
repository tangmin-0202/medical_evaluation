from __future__ import annotations

import cv2
import numpy as np
import pytest

from medical_evaluation.features.appearance import (
    box_overlap_ratio,
    green_dam_area_ratio,
    green_dam_mask,
    visible_reference_area_ratio,
    visible_reference_mask,
    visible_white_frame_near_dam_edge,
)


def test_green_dam_area_ratio_reports_confident_absence_and_half_frame() -> None:
    black = np.zeros((20, 20, 3), dtype=np.uint8)
    half_green = black.copy()
    half_green[:, :10] = (30, 170, 90)

    assert green_dam_area_ratio(black) == 0.0
    assert green_dam_area_ratio(half_green) == pytest.approx(0.5)


def test_green_dam_mask_excludes_skin_colored_pixels() -> None:
    frame = np.full((20, 20, 3), (120, 170, 220), dtype=np.uint8)
    frame[:, :10] = (30, 170, 90)

    mask = green_dam_mask(frame)

    assert mask[:, :10].all()
    assert not mask[:, 10:].any()


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
    assert not visible_reference_mask(
        green,
        candidate,
        reference_lab,
        max_lab_distance=10,
    ).any()


def test_white_frame_search_uses_outer_dam_edge_and_excludes_teeth_and_gloves() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    dam = np.zeros((100, 100), dtype=bool)
    dam[20:80, 20:80] = True
    image[dam] = (30, 170, 90)

    # Central white teeth are inside the dam and must not count as exposed frame.
    image[45:55, 45:55] = 240
    # A white glove entering from the image edge must also be ignored.
    image[20:35, :25] = 240
    # A thin white strip beside the outer dam edge is the exposed-frame candidate.
    image[17:20, 35:70] = 240

    visible, search_band = visible_white_frame_near_dam_edge(
        image,
        dam,
        boundary_width_ratio=0.05,
        minimum_component_area_ratio=0.0005,
    )

    assert search_band[18, 50]
    assert visible[18, 50]
    assert not visible[50, 50]
    assert not visible[25, 0]


def test_white_frame_search_returns_empty_when_only_teeth_are_white() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    dam = np.zeros((100, 100), dtype=bool)
    dam[15:85, 15:85] = True
    image[dam] = (30, 170, 90)
    image[45:55, 45:55] = 240

    visible, _search_band = visible_white_frame_near_dam_edge(image, dam)

    assert not visible.any()


def test_white_frame_search_ignores_white_tooth_hole_inside_dam_mask() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    dam = np.zeros((100, 100), dtype=bool)
    dam[15:85, 15:85] = True
    dam[45:55, 45:55] = False
    image[dam] = (30, 170, 90)
    image[45:55, 45:55] = 240

    visible, search_band = visible_white_frame_near_dam_edge(image, dam)

    assert not search_band[50, 50]
    assert not visible.any()
