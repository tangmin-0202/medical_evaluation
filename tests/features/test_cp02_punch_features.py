import numpy as np
import pytest

from medical_evaluation.features import cp02_punch as punch


def test_second_largest_is_not_second_screen_position():
    holes = [punch.Hole(0.1, 0.3, 0.02), punch.Hole(0.5, 0.3, 0.06),
             punch.Hole(0.8, 0.3, 0.04)]
    assert punch.second_largest_hole(holes, layout_reliable=True, min_radius_gap=0.005) == 2
    assert punch.second_largest_hole(holes, layout_reliable=False, min_radius_gap=0.005) is None


def test_ambiguous_size_ranking_cannot_pick_a_hole():
    holes = [punch.Hole(0.1, 0.3, 0.04), punch.Hole(0.5, 0.3, 0.04)]
    assert punch.second_largest_hole(holes, layout_reliable=True, min_radius_gap=0.005) is None


def test_alignment_requires_tip_near_actual_hole_and_unique_match():
    holes = [punch.Hole(0.1, 0.3, 0.06), punch.Hole(0.5, 0.3, 0.04)]
    assert punch.aligned_hole(holes, (0.5, 0.3), max_distance_in_radii=1) == 1
    assert punch.aligned_hole(holes, (0.9, 0.9), max_distance_in_radii=1) is None


def test_same_color_ratio_samples_only_hole_interior():
    frame = np.zeros((10, 10, 3), np.uint8)
    frame[:] = (0, 180, 0)
    frame[4:6, 4:6] = (255, 255, 255)
    mask = np.zeros((10, 10), bool)
    mask[4:6, 4:6] = True
    reference = np.zeros((10, 10), bool)
    reference[:2] = True
    assert punch.dam_color_ratio(frame, mask, reference, max_lab_distance=20) == 0
    frame[4:6, 4:6] = (0, 180, 0)
    assert punch.dam_color_ratio(frame, mask, reference, max_lab_distance=20) == 1
    assert punch.dam_color_ratio(frame, np.zeros_like(mask), reference, max_lab_distance=20) is None


def test_invalid_hole_radius_is_rejected():
    with pytest.raises(ValueError):
        punch.Hole(0.1, 0.2, -0.1)
