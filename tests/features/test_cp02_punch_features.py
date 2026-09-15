import cv2
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


def _disk_frame(center, *, moving=True):
    frame = np.full((360, 640, 3), (170, 120, 70), np.uint8)
    cv2.circle(frame, (520, 90), 28, (175, 175, 175), -1)
    for angle in np.linspace(0, 2 * np.pi, 5, endpoint=False):
        point = (520 + int(15 * np.cos(angle)), 90 + int(15 * np.sin(angle)))
        cv2.circle(frame, point, 4, (25, 25, 25), -1)
    if moving:
        cv2.circle(frame, center, 30, (185, 185, 185), -1)
        for angle in np.linspace(0, 2 * np.pi, 6, endpoint=False):
            point = (
                center[0] + int(16 * np.cos(angle)),
                center[1] + int(16 * np.sin(angle)),
            )
            cv2.circle(frame, point, 4, (20, 20, 20), -1)
    return frame


def test_locates_moving_multihole_disk_and_rejects_static_distractor():
    frames = [_disk_frame((120 + 35 * index, 250)) for index in range(5)]

    result = punch.locate_moving_multihole_disk(frames)

    assert result is not None
    assert result.hole_count >= 3
    assert result.motion_ratio >= 0.15
    assert result.y > 200


def test_static_multihole_disks_do_not_create_automatic_box():
    frame = _disk_frame((120, 250), moving=False)

    assert punch.locate_moving_multihole_disk([frame.copy() for _ in range(5)]) is None


def test_prefers_true_hole_disk_over_adjacent_round_press_mechanism():
    frames = []
    for index in range(5):
        frame = np.full((360, 640, 3), (170, 120, 70), np.uint8)
        disk_center = (150 + 30 * index, 250)
        press_center = (205 + 30 * index, 245)
        cv2.circle(frame, disk_center, 30, (185, 185, 185), -1)
        for angle in np.linspace(0, 2 * np.pi, 6, endpoint=False):
            point = (
                disk_center[0] + int(17 * np.cos(angle)),
                disk_center[1] + int(17 * np.sin(angle)),
            )
            cv2.circle(frame, point, 4, (20, 20, 20), -1)
        cv2.circle(frame, press_center, 36, (180, 180, 180), -1)
        for offset in range(-24, 25, 6):
            cv2.rectangle(
                frame,
                (press_center[0] + offset - 1, press_center[1] - 16),
                (press_center[0] + offset + 1, press_center[1] + 16),
                (25, 25, 25),
                -1,
            )
        frames.append(frame)

    result = punch.locate_moving_multihole_disk(frames)

    assert result is not None
    expected_centers = [150 + 30 * index for index in range(5)]
    assert min(abs(result.x - expected) for expected in expected_centers) <= 10
