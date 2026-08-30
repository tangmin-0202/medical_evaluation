from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from medical_evaluation.features.appearance import (
    blur_score,
    color_ratio_hsv,
    reference_cosine_similarity,
)
from medical_evaluation.features.geometry import (
    area_ratio,
    boundary_contact_ratio,
    bounding_box_center,
    centroid_normalized,
    centroid_xy,
    frame_center_offset,
    intersection_ratio,
    mask_iou,
    relative_bbox_center_offset,
    write_overlay,
)
from medical_evaluation.features.motion import (
    direction_change_count,
    displacement_sequence,
    velocity_sequence,
)


def test_centroid_and_intersection_ratio() -> None:
    first = np.zeros((10, 10), bool)
    first[2:6, 2:6] = True
    second = np.zeros((10, 10), bool)
    second[4:8, 4:8] = True

    assert centroid_xy(first) == (3.5, 3.5)
    assert centroid_normalized(first) == pytest.approx((3.5 / 9, 3.5 / 9))
    assert intersection_ratio(first, second) == 4 / 16
    assert mask_iou(first, second) == 4 / 28
    assert area_ratio(first) == 0.16


def test_bounding_box_center_and_frame_offset() -> None:
    centered = np.zeros((100, 200), bool)
    centered[25:75, 50:150] = True
    shifted = np.zeros((100, 200), bool)
    shifted[25:75, 100:200] = True
    empty = np.zeros((100, 200), bool)

    assert bounding_box_center(centered) == (99.5, 49.5)
    assert frame_center_offset(centered) == pytest.approx(0.0, abs=0.004)
    assert frame_center_offset(shifted) == pytest.approx(0.25, abs=0.004)
    assert bounding_box_center(empty) is None
    assert frame_center_offset(empty) is None


def test_relative_bbox_center_offset_uses_reference_scale() -> None:
    oral = np.zeros((100, 200), bool)
    oral[20:80, 50:150] = True
    centered_frame = np.zeros((100, 200), bool)
    centered_frame[40:60, 90:110] = True
    shifted_frame = np.zeros((100, 200), bool)
    shifted_frame[40:60, 140:160] = True

    assert relative_bbox_center_offset(centered_frame, oral) == pytest.approx(0.0)
    assert relative_bbox_center_offset(shifted_frame, oral) == pytest.approx(0.5)


def test_relative_bbox_center_offset_handles_missing_and_mismatched_masks() -> None:
    empty = np.zeros((10, 10), bool)
    present = np.ones((10, 10), bool)

    assert relative_bbox_center_offset(empty, present) is None
    assert relative_bbox_center_offset(present, empty) is None
    with pytest.raises(ValueError, match="dimensions"):
        relative_bbox_center_offset(present, np.ones((8, 8), bool))


def test_empty_geometry_returns_none_instead_of_false_zero() -> None:
    empty = np.zeros((5, 5), bool)

    assert centroid_xy(empty) is None
    assert centroid_normalized(empty) is None
    assert intersection_ratio(empty, empty) is None
    assert mask_iou(empty, empty) is None


def test_boundary_contact_ratio_detects_nearby_pixels() -> None:
    first = np.zeros((8, 8), bool)
    first[3:5, 3:5] = True
    second = np.zeros((8, 8), bool)
    second[3:5, 5:7] = True

    assert boundary_contact_ratio(first, second, max_distance=1.0) == 0.5


def test_displacement_and_velocity_preserve_time_order() -> None:
    points = [(0.0, 0.0), (1.0, 2.0), (1.0, 5.0)]

    assert displacement_sequence(points) == [(1.0, 2.0), (0.0, 3.0)]
    assert velocity_sequence(points, [0.0, 0.5, 2.0]) == [(2.0, 4.0), (0.0, 2.0)]
    assert direction_change_count([(1, 0), (-1, 0), (-1, 1)]) == 1


def test_appearance_features_are_explicit_for_missing_regions() -> None:
    green = np.zeros((10, 10, 3), dtype=np.uint8)
    green[:] = (0, 255, 0)
    full = np.ones((10, 10), bool)
    empty = np.zeros((10, 10), bool)

    assert color_ratio_hsv(green, full, hue_range=(35, 85)) == 1.0
    assert color_ratio_hsv(green, empty, hue_range=(35, 85)) is None
    assert blur_score(green, empty) is None
    assert reference_cosine_similarity(np.array([1, 0]), np.array([1, 0])) == 1.0
    assert reference_cosine_similarity(np.array([0, 0]), np.array([1, 0])) is None


def test_overlay_is_written_only_beneath_evidence_root(tmp_path: Path) -> None:
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    mask = np.zeros((20, 20), bool)
    mask[5:15, 5:15] = True

    output = write_overlay(frame, {"rubber_dam": mask}, tmp_path, "cp_01/frame.jpg")

    assert output.is_file()
    assert cv2.imread(str(output)).sum() > 0
    with pytest.raises(ValueError, match="outside"):
        write_overlay(frame, {"rubber_dam": mask}, tmp_path, "../escape.jpg")
