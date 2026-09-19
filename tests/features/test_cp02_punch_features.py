import math

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


def test_finds_hole_aligned_with_external_punch_plunger():
    frame = np.full((260, 360, 3), (170, 120, 70), np.uint8)
    center = (170, 130)
    cv2.circle(frame, center, 62, (190, 190, 190), -1)
    holes = []
    for radius, angle in zip((9, 7, 6, 5, 4), (210, 250, 290, 330, 10), strict=True):
        radians = np.deg2rad(angle)
        point = (
            center[0] + round(35 * np.cos(radians)),
            center[1] + round(35 * np.sin(radians)),
        )
        cv2.circle(frame, point, radius, (20, 20, 20), -1)
        holes.append(punch.Hole(*point, radius))
    # The fixed opposite jaw/plunger continues radially beyond hole 1.
    direction = np.deg2rad(250)
    cv2.line(
        frame,
        (center[0] + round(50 * np.cos(direction)), center[1] + round(50 * np.sin(direction))),
        (center[0] + round(105 * np.cos(direction)), center[1] + round(105 * np.sin(direction))),
        (40, 40, 40),
        10,
    )
    disk = punch.MovingDisk(0, *center, 64, 5, 0.5, 60)

    result = punch.aligned_hole_from_plunger(frame, disk, holes)

    assert result == 1


def test_plunger_alignment_rejects_two_equally_supported_holes():
    frame = np.full((260, 360, 3), (170, 120, 70), np.uint8)
    center = (170, 130)
    cv2.circle(frame, center, 62, (190, 190, 190), -1)
    holes = []
    for angle in (210, 250, 290, 330, 10):
        radians = np.deg2rad(angle)
        point = (
            center[0] + round(35 * np.cos(radians)),
            center[1] + round(35 * np.sin(radians)),
        )
        cv2.circle(frame, point, 5, (20, 20, 20), -1)
        holes.append(punch.Hole(*point, 5))
    for angle in (210, 250):
        direction = np.deg2rad(angle)
        cv2.line(
            frame,
            (center[0] + round(50 * np.cos(direction)), center[1] + round(50 * np.sin(direction))),
            (center[0] + round(105 * np.cos(direction)), center[1] + round(105 * np.sin(direction))),
            (40, 40, 40),
            10,
        )
    disk = punch.MovingDisk(0, *center, 64, 5, 0.5, 60)

    assert punch.aligned_hole_from_plunger(frame, disk, holes) is None


def test_selected_hole_is_opposite_broad_handle_and_jaw_cluster():
    frame = np.full((300, 380, 3), (170, 120, 70), np.uint8)
    center = (180, 145)
    cv2.circle(frame, center, 62, (190, 190, 190), -1)
    holes = []
    for angle in (210, 250, 290, 330, 10):
        radians = np.deg2rad(angle)
        point = (
            center[0] + round(35 * np.cos(radians)),
            center[1] + round(35 * np.sin(radians)),
        )
        cv2.circle(frame, point, 5, (20, 20, 20), -1)
        holes.append(punch.Hole(*point, 5))
    for angle in (55, 85, 110):
        radians = np.deg2rad(angle)
        cv2.line(
            frame,
            (center[0] + round(50 * np.cos(radians)), center[1] + round(50 * np.sin(radians))),
            (center[0] + round(120 * np.cos(radians)), center[1] + round(120 * np.sin(radians))),
            (45, 45, 45),
            8,
        )
    disk = punch.MovingDisk(0, *center, 64, 5, 0.5, 60)

    assert punch.aligned_hole_opposite_handle(frame, disk, holes) == 1


def test_handle_opposite_alignment_rejects_equidistant_holes():
    frame = np.full((300, 380, 3), (170, 120, 70), np.uint8)
    center = (180, 145)
    cv2.circle(frame, center, 62, (190, 190, 190), -1)
    holes = []
    for angle in (210, 250, 290, 330, 10):
        radians = np.deg2rad(angle)
        point = (
            center[0] + round(35 * np.cos(radians)),
            center[1] + round(35 * np.sin(radians)),
        )
        cv2.circle(frame, point, 5, (20, 20, 20), -1)
        holes.append(punch.Hole(*point, 5))
    for angle in (65, 90, 115):
        radians = np.deg2rad(angle)
        cv2.line(
            frame,
            (center[0] + round(50 * np.cos(radians)), center[1] + round(50 * np.sin(radians))),
            (center[0] + round(120 * np.cos(radians)), center[1] + round(120 * np.sin(radians))),
            (45, 45, 45),
            8,
        )
    disk = punch.MovingDisk(0, *center, 64, 5, 0.5, 60)

    assert punch.aligned_hole_opposite_handle(frame, disk, holes) is None


def test_probe_contact_requires_a_line_entering_the_same_hole():
    frame = np.full((160, 160, 3), 180, np.uint8)
    hole = punch.Hole(80, 80, 8)
    cv2.line(frame, (35, 80), (82, 80), (35, 35, 35), 3)

    assert punch.probe_contacts_hole(frame, hole) is True


def test_probe_contact_rejects_a_nearby_line_that_misses_the_hole():
    frame = np.full((160, 160, 3), 180, np.uint8)
    hole = punch.Hole(80, 80, 8)
    cv2.line(frame, (35, 55), (125, 55), (35, 35, 35), 3)

    assert punch.probe_contacts_hole(frame, hole) is False


def test_punch_event_requires_sustained_green_interaction_with_motion():
    observations = [
        punch.PunchInteractionObservation(1.0, 0.00, 0.03),
        punch.PunchInteractionObservation(1.2, 0.02, 0.04),
        punch.PunchInteractionObservation(1.4, 0.54, 0.05),
        punch.PunchInteractionObservation(1.6, 0.58, 0.07),
        punch.PunchInteractionObservation(1.8, 0.61, 0.04),
        punch.PunchInteractionObservation(2.0, 0.04, 0.06),
    ]

    event = punch.detect_punch_event(observations)

    assert event is not None
    assert event.time_sec == pytest.approx(1.4)
    assert event.end_time_sec == pytest.approx(1.8)
    assert event.reason == "sustained_green_punch_interaction"


def test_punch_event_rejects_dry_trial_alignment_and_single_green_flash():
    dry_trial = [
        punch.PunchInteractionObservation(1.0, 0.00, 0.08),
        punch.PunchInteractionObservation(1.2, 0.01, 0.10),
        punch.PunchInteractionObservation(1.4, 0.00, 0.07),
    ]
    green_flash = [
        punch.PunchInteractionObservation(2.0, 0.00, 0.02),
        punch.PunchInteractionObservation(2.2, 0.30, 0.10),
        punch.PunchInteractionObservation(2.4, 0.01, 0.03),
    ]

    assert punch.detect_punch_event(dry_trial) is None
    assert punch.detect_punch_event(green_flash) is None


def test_punch_event_rejects_stationary_green_background():
    observations = [
        punch.PunchInteractionObservation(1.0, 0.30, 0.002),
        punch.PunchInteractionObservation(1.2, 0.32, 0.003),
        punch.PunchInteractionObservation(1.4, 0.31, 0.002),
    ]

    assert punch.detect_punch_event(observations) is None


def test_punch_event_rejects_moderate_green_background_drift_during_alignment():
    observations = [
        punch.PunchInteractionObservation(1.0, 0.06, 0.04),
        punch.PunchInteractionObservation(1.2, 0.08, 0.08),
        punch.PunchInteractionObservation(1.4, 0.19, 0.09),
        punch.PunchInteractionObservation(1.6, 0.22, 0.07),
        punch.PunchInteractionObservation(1.8, 0.25, 0.06),
    ]

    assert punch.detect_punch_event(observations) is None


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


def test_last_moving_disk_uses_late_visible_frames_without_sam3_mask():
    frames = [_disk_frame((120 + 30 * index, 250)) for index in range(4)]
    frames += [_disk_frame((350 + 25 * index, 250)) for index in range(3)]
    frames.append(_disk_frame((0, 0), moving=False))

    result = punch.locate_last_moving_multihole_disk(frames)

    assert result is not None
    assert result.frame_position >= 4
    assert result.x > 300


def test_last_moving_disk_does_not_infer_from_static_background():
    frame = _disk_frame((0, 0), moving=False)
    assert punch.locate_last_moving_multihole_disk([frame.copy() for _ in range(5)]) is None


def test_rejects_unrectified_foreshortened_disk_before_ranking():
    frame = np.full((260, 360, 3), (170, 120, 70), np.uint8)
    center = (170, 130)
    cv2.ellipse(frame, center, (62, 42), 24, 0, 360, (195, 195, 195), -1)
    radii = [9, 7, 5, 4, 3]
    angles = np.deg2rad([20, 92, 164, 236, 308])
    rotation = np.deg2rad(24)
    for radius, angle in zip(radii, angles, strict=True):
        local_x = 35 * np.cos(angle)
        local_y = 24 * np.sin(angle)
        x = center[0] + local_x * np.cos(rotation) - local_y * np.sin(rotation)
        y = center[1] + local_x * np.sin(rotation) + local_y * np.cos(rotation)
        cv2.ellipse(
            frame, (round(x), round(y)), (radius, max(2, round(radius * 0.68))),
            24, 0, 360, (25, 25, 25), -1,
        )
    disk = punch.MovingDisk(0, *center, 66, 5, 0.5, 60)

    result = punch.measure_disk_holes(frame, disk)

    assert result.reliable is False
    assert result.reason == "unrectified_tilted_disk"
    assert result.second_largest_index is None


def test_measures_five_hole_punch_layout_on_partial_arc():
    frame = np.full((240, 320, 3), (170, 120, 70), np.uint8)
    center = (150, 130)
    cv2.circle(frame, center, 65, (190, 190, 190), -1)
    for radius, angle in zip((8, 7, 6, 5, 4), (205, 235, 265, 295, 325), strict=True):
        radians = np.deg2rad(angle)
        point = (
            center[0] + round(40 * np.cos(radians)),
            center[1] + round(40 * np.sin(radians)),
        )
        cv2.circle(frame, point, radius, (20, 20, 20), -1)
    disk = punch.MovingDisk(0, *center, 67, 5, 0.5, 60)

    result = punch.measure_disk_holes(frame, disk)

    assert result.reliable is True
    assert result.reason == "criteria_satisfied"
    assert result.second_largest_index is not None


def test_disk_hole_measurement_selects_consecutive_outer_arc_over_mechanism_decoys():
    frame = np.full((260, 340, 3), (170, 120, 70), np.uint8)
    center = (160, 130)
    cv2.circle(frame, center, 68, (190, 190, 190), -1)
    expected_points = []
    for radius, angle in zip((8, 7, 6, 5, 4), (200, 230, 260, 290, 320), strict=True):
        radians = np.deg2rad(angle)
        point = (
            center[0] + round(42 * np.cos(radians)),
            center[1] + round(42 * np.sin(radians)),
        )
        expected_points.append(point)
        cv2.circle(frame, point, radius, (20, 20, 20), -1)

    # Central mechanism/reflections may be round and dark, but they are not a
    # consecutive row on the selectable-hole annulus.
    for point, radius in (((125, 132), 4), ((160, 101), 4), ((188, 132), 4)):
        cv2.circle(frame, point, radius, (20, 20, 20), -1)
    disk = punch.MovingDisk(0, *center, 70, 8, 0.5, 60)

    result = punch.measure_disk_holes(frame, disk)

    assert result.reliable is True
    assert len(result.holes) == 5
    for hole in result.holes:
        assert min(math.dist((hole.x, hole.y), point) for point in expected_points) < 2.0


def test_hole_arc_selection_prefers_outer_selectable_ring_over_inner_mechanism_arc():
    center = (100.0, 100.0)
    holes = []
    for radial, angles in (
        (28.0, (20, 50, 80, 110, 140)),
        (43.0, (200, 230, 260, 290, 320)),
    ):
        for position, angle in enumerate(angles):
            radians = np.deg2rad(angle)
            holes.append(punch.Hole(
                center[0] + radial * np.cos(radians),
                center[1] + radial * np.sin(radians),
                8.0 - 0.7 * position,
            ))

    selected = punch._select_consecutive_hole_arc(holes, center, 70.0)

    assert selected is not None
    assert set(selected) == set(range(5, 10))


def test_hole_arc_selection_bounds_circle_fits_to_local_angular_windows(monkeypatch):
    center = (100.0, 100.0)
    holes = []
    for position, angle in enumerate(range(0, 360, 30)):
        radians = np.deg2rad(angle)
        radial = 43.0 if 5 <= position <= 9 else 31.0
        holes.append(punch.Hole(
            center[0] + radial * np.cos(radians),
            center[1] + radial * np.sin(radians),
            8.0 - 0.25 * position,
        ))
    original = punch.np.linalg.lstsq
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(punch.np.linalg, "lstsq", counted)

    assert punch._select_consecutive_hole_arc(holes, center, 70.0) is not None
    assert calls <= 300


def test_hole_arc_allows_moderate_perspective_spacing_from_real_tracking():
    center = (1086.0, 825.0)
    holes = (
        punch.Hole(1090.0, 801.0, 4.0),
        punch.Hole(1103.0, 803.0, 3.3),
        punch.Hole(1078.0, 809.0, 2.6),
        punch.Hole(1072.0, 825.0, 1.9),
    )

    selected = punch._select_consecutive_hole_arc(holes, center, 37.0)

    assert selected is not None
    assert set(selected) == {0, 1, 2, 3}


def test_disk_hole_measurement_uses_four_rim_holes_without_central_aperture():
    frame = np.full((240, 320, 3), (170, 120, 70), np.uint8)
    center = (150, 120)
    cv2.circle(frame, center, 65, (190, 190, 190), -1)
    for radius, angle in zip((8, 7, 6, 5), (205, 235, 265, 295), strict=True):
        radians = np.deg2rad(angle)
        point = (
            center[0] + round(40 * np.cos(radians)),
            center[1] + round(40 * np.sin(radians)),
        )
        cv2.circle(frame, point, radius, (20, 20, 20), -1)
    cv2.circle(frame, (center[0] + 18, center[1]), 5, (20, 20, 20), -1)
    disk = punch.MovingDisk(0, *center, 67, 5, 0.5, 60)

    result = punch.measure_disk_holes(frame, disk)

    assert result.reliable is True
    assert result.second_largest_index is not None
    assert len(result.holes) == 4
    assert all(math.dist((hole.x, hole.y), center) > 30 for hole in result.holes)


def test_disk_hole_measurement_rejects_lower_mechanism_dark_highlights():
    frame = np.full((260, 360, 3), (170, 120, 70), np.uint8)
    center = (170, 130)
    cv2.circle(frame, center, 62, (190, 190, 190), -1)
    for point in ((155, 150), (170, 158), (185, 150)):
        cv2.ellipse(frame, point, (3, 10), 0, 0, 360, (25, 25, 25), -1)
    disk = punch.MovingDisk(0, *center, 64, 3, 0.5, 60)

    result = punch.measure_disk_holes(frame, disk)

    assert result.reliable is False
    assert result.second_largest_index is None
    assert result.reason == "insufficient_spatially_distinct_holes"


def test_disk_hole_measurement_accepts_warm_front_facing_disk_from_edge_support():
    frame = np.full((240, 320, 3), (170, 120, 70), np.uint8)
    center = (150, 120)
    cv2.circle(frame, center, 48, (90, 145, 205), -1)
    for radius, angle in zip((8, 7, 5, 4), (10, 100, 190, 280), strict=True):
        radians = np.deg2rad(angle)
        point = (
            center[0] + round(27 * np.cos(radians)),
            center[1] + round(27 * np.sin(radians)),
        )
        cv2.circle(frame, point, radius, (20, 20, 20), -1)
    disk = punch.MovingDisk(0, *center, 50, 4, 0.5, 60)

    result = punch.measure_disk_holes(frame, disk, expected_hole_count=4)

    assert result.reliable is True
    assert result.reason == "criteria_satisfied"


def test_disk_hole_measurement_rejects_warm_foreshortened_face():
    frame = np.full((240, 320, 3), (170, 120, 70), np.uint8)
    center = (150, 120)
    cv2.ellipse(frame, center, (55, 31), 20, 0, 360, (90, 145, 205), -1)
    for radius, angle in zip((8, 7, 5, 4, 3), (10, 82, 154, 226, 298), strict=True):
        radians = np.deg2rad(angle)
        point = (
            center[0] + round(32 * np.cos(radians)),
            center[1] + round(18 * np.sin(radians)),
        )
        cv2.ellipse(frame, point, (radius, max(2, radius // 2)), 20, 0, 360, (20, 20, 20), -1)
    disk = punch.MovingDisk(0, *center, 58, 5, 0.5, 60)

    result = punch.measure_disk_holes(frame, disk)

    assert result.reliable is False
    assert result.reason == "unreliable_disk_plane"


def test_disk_hole_measurement_uses_expected_layout_not_noisy_locator_count():
    frame = np.full((240, 320, 3), (170, 120, 70), np.uint8)
    center = (150, 120)
    cv2.circle(frame, center, 65, (190, 190, 190), -1)
    for radius, angle in zip((8, 7, 6, 5, 4), (205, 235, 265, 295, 325), strict=True):
        radians = np.deg2rad(angle)
        point = (
            center[0] + round(40 * np.cos(radians)),
            center[1] + round(40 * np.sin(radians)),
        )
        cv2.circle(frame, point, radius, (20, 20, 20), -1)
    disk = punch.MovingDisk(0, *center, 67, 6, 0.5, 60)

    result = punch.measure_disk_holes(frame, disk, expected_hole_count=5)

    assert len(result.holes) == 5
    assert result.reliable is True
    assert result.reason == "criteria_satisfied"
    assert result.second_largest_index is not None


def test_disk_hole_measurement_accepts_four_consecutive_visible_holes():
    frame = np.full((240, 320, 3), (170, 120, 70), np.uint8)
    center = (150, 120)
    cv2.circle(frame, center, 48, (190, 190, 190), -1)
    for radius, angle in zip((8, 7, 6, 4), (215, 250, 285, 320), strict=True):
        radians = np.deg2rad(angle)
        point = (
            center[0] + round(27 * np.cos(radians)),
            center[1] + round(27 * np.sin(radians)),
        )
        cv2.circle(frame, point, radius, (20, 20, 20), -1)
    disk = punch.MovingDisk(0, *center, 50, 4, 0.5, 60)

    result = punch.measure_disk_holes(frame, disk, expected_hole_count=5)

    assert len(result.holes) == 4
    assert result.reliable is True
    assert result.reason == "criteria_satisfied"
    assert result.second_largest_index is not None


def test_disk_hole_measurement_uses_monotonic_arc_when_two_large_holes_quantize_equal():
    frame = np.full((260, 340, 3), (170, 120, 70), np.uint8)
    center = (160, 130)
    cv2.circle(frame, center, 68, (190, 190, 190), -1)
    points = []
    for radius, angle in zip((8, 8, 5, 3), (210, 245, 280, 315), strict=True):
        radians = np.deg2rad(angle)
        point = (
            center[0] + round(43 * np.cos(radians)),
            center[1] + round(43 * np.sin(radians)),
        )
        points.append(point)
        cv2.circle(frame, point, radius, (20, 20, 20), -1)
    disk = punch.MovingDisk(0, *center, 70, 4, 0.5, 60)

    result = punch.measure_disk_holes(frame, disk)

    assert result.reliable is True
    assert result.second_largest_index is not None
    selected = result.holes[result.second_largest_index]
    assert math.dist((selected.x, selected.y), points[1]) < 2.0


def test_disk_hole_measurement_accepts_three_consecutive_front_holes():
    frame = np.full((240, 320, 3), (170, 120, 70), np.uint8)
    center = (150, 120)
    cv2.circle(frame, center, 58, (190, 190, 190), -1)
    for radius, angle in zip((8, 6, 4), (220, 255, 290), strict=True):
        radians = np.deg2rad(angle)
        point = (
            center[0] + round(36 * np.cos(radians)),
            center[1] + round(36 * np.sin(radians)),
        )
        cv2.circle(frame, point, radius, (20, 20, 20), -1)
    disk = punch.MovingDisk(0, *center, 60, 3, 0.5, 60)

    result = punch.measure_disk_holes(frame, disk, expected_hole_count=5)

    assert len(result.holes) == 3
    assert result.reliable is True
    assert result.reason == "criteria_satisfied"
    assert result.second_largest_index is not None


def test_locates_five_hole_layout_near_previous_automatic_disk():
    frame = np.full((300, 500, 3), (170, 120, 70), np.uint8)
    center = (180, 155)
    cv2.circle(frame, center, 58, (190, 190, 190), -1)
    for radius, angle in zip((9, 7, 6, 5, 4), (205, 235, 265, 295, 325), strict=True):
        radians = np.deg2rad(angle)
        point = (
            center[0] + round(36 * np.cos(radians)),
            center[1] + round(36 * np.sin(radians)),
        )
        cv2.circle(frame, point, radius, (20, 20, 20), -1)
    cv2.circle(frame, (410, 80), 45, (190, 190, 190), -1)
    reference = punch.MovingDisk(0, 165, 150, 55, 5, 0.5, 60)

    result = punch.locate_disk_layout_near(frame, reference)

    assert result is not None
    disk, layout = result
    assert disk.x == pytest.approx(center[0], abs=8)
    assert disk.y == pytest.approx(center[1], abs=8)
    assert layout.reliable is True
    assert layout.second_largest_index is not None


def test_disk_layout_candidates_measure_only_local_roi(monkeypatch):
    frame = np.full((1080, 1920, 3), (170, 120, 70), np.uint8)
    center = (960, 540)
    cv2.circle(frame, center, 58, (190, 190, 190), -1)
    for radius, angle in zip((9, 7, 6, 5, 4), (205, 235, 265, 295, 325), strict=True):
        radians = np.deg2rad(angle)
        point = (
            center[0] + round(36 * np.cos(radians)),
            center[1] + round(36 * np.sin(radians)),
        )
        cv2.circle(frame, point, radius, (20, 20, 20), -1)
    measured_shapes = []
    original = punch.measure_disk_holes

    def record_shape(local_frame, disk, **kwargs):
        measured_shapes.append(local_frame.shape)
        return original(local_frame, disk, **kwargs)

    monkeypatch.setattr(punch, "measure_disk_holes", record_shape)
    reference = punch.MovingDisk(0, 945, 535, 55, 5, 0.5, 60)

    result = punch.locate_disk_layout_near(frame, reference)

    assert result is not None
    assert measured_shapes
    assert all(height <= 160 and width <= 160 for height, width, _ in measured_shapes)
    assert all(hole.x > 800 and hole.y > 400 for hole in result[1].holes)


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


def test_prefers_solid_hole_disk_over_moving_open_press_with_more_dark_spots():
    frames = []
    for index in range(5):
        frame = np.full((360, 640, 3), (170, 120, 70), np.uint8)
        disk_center = (145 + 28 * index, 240)
        press_center = (225 + 28 * index, 240)
        cv2.circle(frame, disk_center, 30, (190, 190, 190), -1)
        for angle in np.linspace(0, 2 * np.pi, 6, endpoint=False):
            point = (
                disk_center[0] + int(16 * np.cos(angle)),
                disk_center[1] + int(16 * np.sin(angle)),
            )
            cv2.circle(frame, point, 4, (20, 20, 20), -1)
        cv2.circle(frame, press_center, 36, (190, 190, 190), 3)
        for angle in np.linspace(0, 2 * np.pi, 10, endpoint=False):
            point = (
                press_center[0] + int(20 * np.cos(angle)),
                press_center[1] + int(20 * np.sin(angle)),
            )
            cv2.circle(frame, point, 3, (20, 20, 20), -1)
        frames.append(frame)

    result = punch.locate_moving_multihole_disk(frames)

    assert result is not None
    assert min(abs(result.x - (145 + 28 * index)) for index in range(5)) <= 10


def test_disk_surface_contrast_rejects_hollow_press_opening():
    frame = np.full((240, 400, 3), (170, 120, 70), np.uint8)
    cv2.circle(frame, (100, 120), 30, (190, 190, 190), -1)
    cv2.circle(frame, (290, 120), 30, (190, 190, 190), 2)

    solid = punch._disk_surface_contrast(frame, 100, 120, 30)
    hollow = punch._disk_surface_contrast(frame, 290, 120, 30)

    assert solid > 70
    assert hollow < 20


def test_disk_candidate_must_be_fully_inside_image():
    assert punch._disk_fully_visible(120, 120, 30, 400, 240)
    assert not punch._disk_fully_visible(120, 220, 30, 400, 240)


def _held_punch_frames(*, include_handle=True, handle_color=(195, 195, 195)):
    frames = []
    for index in range(5):
        frame = np.full((360, 640, 3), (170, 120, 70), np.uint8)
        cv2.rectangle(frame, (500, 45), (560, 125), (190, 190, 190), -1)
        center = (130 + 35 * index, 235)
        if include_handle:
            cv2.line(
                frame,
                (center[0] + 16, center[1] + 8),
                (center[0] + 110, center[1] + 70),
                handle_color,
                18,
            )
        cv2.circle(frame, center, 30, (185, 185, 185), -1)
        for angle in np.linspace(0, 2 * np.pi, 6, endpoint=False):
            point = (
                center[0] + int(16 * np.cos(angle)),
                center[1] + int(16 * np.sin(angle)),
            )
            cv2.circle(frame, point, 4, (20, 20, 20), -1)
        frames.append(frame)
    return frames


def test_held_punch_box_follows_moving_tool_not_static_distractor():
    frames = _held_punch_frames()
    disk = punch.locate_moving_multihole_disk(frames)

    box = punch.locate_held_punch_box(frames, disk)

    assert disk is not None
    assert box is not None
    expected_handle_end = (130 + 35 * box.frame_position + 105, 300)
    assert box.contains(disk.x, disk.y)
    assert box.contains(*expected_handle_end)
    assert not box.contains(530, 85)


def test_held_punch_box_requires_connected_elongated_motion():
    frames = _held_punch_frames(include_handle=False)
    disk = punch.locate_moving_multihole_disk(frames)

    assert disk is not None
    assert punch.locate_held_punch_box(frames, disk) is None


def test_held_punch_box_uses_edges_when_warm_lighting_raises_saturation():
    frames = _held_punch_frames(handle_color=(75, 135, 195))
    disk = punch.locate_moving_multihole_disk(frames)

    box = punch.locate_held_punch_box(frames, disk)

    assert disk is not None
    assert box is not None
    expected_handle_end = (130 + 35 * box.frame_position + 105, 300)
    assert box.contains(disk.x, disk.y)
    assert box.contains(*expected_handle_end)


def _mask_fixture():
    frame = np.full((240, 320, 3), (30, 170, 30), np.uint8)
    tool_mask = np.zeros((240, 320), bool)
    cv2.line(tool_mask.view(np.uint8), (115, 125), (245, 185), 1, 18)
    cv2.circle(tool_mask.view(np.uint8), (100, 120), 24, 1, -1)
    frame[tool_mask] = (190, 190, 190)
    for angle in np.linspace(0, 2 * np.pi, 6, endpoint=False):
        point = (100 + int(13 * np.cos(angle)), 120 + int(13 * np.sin(angle)))
        cv2.circle(frame, point, 3, (20, 20, 20), -1)
    disk = punch.MovingDisk(
        frame_position=0,
        x=100,
        y=120,
        radius=24,
        hole_count=6,
        motion_ratio=0.6,
    )
    box = punch.HeldPunchBox(
        frame_position=0,
        x1=65,
        y1=85,
        x2=270,
        y2=210,
        elongation=2.0,
        motion_ratio=0.6,
    )
    return frame, tool_mask, disk, box


def test_held_punch_mask_accepts_anchor_connected_elongated_metal():
    frame, tool_mask, disk, box = _mask_fixture()

    result = punch.measure_held_punch_mask(frame, tool_mask, disk, box)

    assert result.accepted is True
    assert result.reason == "criteria_satisfied"
    assert result.green_ratio < 0.1


def test_held_punch_mask_rejects_green_sheet():
    frame, _, disk, box = _mask_fixture()
    green_mask = np.zeros(frame.shape[:2], bool)
    green_mask[70:225, 45:300] = True

    result = punch.measure_held_punch_mask(frame, green_mask, disk, box)

    assert result.accepted is False
    assert result.reason == "green_sheet_mask"


def test_held_punch_mask_rejects_mask_without_visible_disk():
    frame, _, disk, box = _mask_fixture()
    unrelated = np.zeros(frame.shape[:2], bool)
    unrelated[20:60, 220:300] = True
    frame[unrelated] = (190, 190, 190)

    result = punch.measure_held_punch_mask(frame, unrelated, disk, box)

    assert result.accepted is False
    assert result.reason == "disk_holes_not_visible"


def test_held_punch_mask_accepts_moved_multihole_disk_with_glove():
    frame, _, disk, box = _mask_fixture()
    moved = np.zeros(frame.shape[:2], bool)
    cv2.circle(moved.view(np.uint8), (230, 110), 31, 1, -1)
    cv2.rectangle(moved.view(np.uint8), (220, 135), (295, 205), 1, -1)
    frame[moved] = (190, 190, 190)
    frame[140:205, 240:295] = (175, 195, 220)
    for angle in np.linspace(0, 2 * np.pi, 6, endpoint=False):
        point = (230 + int(17 * np.cos(angle)), 110 + int(17 * np.sin(angle)))
        cv2.circle(frame, point, 4, (20, 20, 20), -1)

    result = punch.measure_held_punch_mask(frame, moved, disk, box)

    assert result.accepted is True
    assert result.reason == "criteria_satisfied"
    assert result.visible_hole_count >= 3


def test_held_punch_mask_rejects_moved_blob_without_holes():
    frame, _, disk, box = _mask_fixture()
    unrelated = np.zeros(frame.shape[:2], bool)
    cv2.circle(unrelated.view(np.uint8), (230, 110), 31, 1, -1)
    frame[unrelated] = (190, 190, 190)

    result = punch.measure_held_punch_mask(frame, unrelated, disk, box)

    assert result.accepted is False
    assert result.reason == "disk_holes_not_visible"


def test_visible_disk_prefers_solid_wheel_over_lower_mechanism_dark_spots():
    frame = np.full((300, 420, 3), (170, 120, 70), np.uint8)
    mask = np.zeros(frame.shape[:2], bool)
    cv2.circle(frame, (130, 90), 36, (195, 195, 195), -1)
    cv2.circle(mask.view(np.uint8), (130, 90), 36, 1, -1)
    for angle in np.linspace(0, 2 * np.pi, 5, endpoint=False):
        point = (130 + int(20 * np.cos(angle)), 90 + int(20 * np.sin(angle)))
        cv2.circle(frame, point, 4, (20, 20, 20), -1)
    cv2.rectangle(frame, (110, 135), (180, 255), (170, 170, 170), -1)
    cv2.rectangle(mask.view(np.uint8), (110, 135), (180, 255), 1, -1)
    cv2.circle(frame, (145, 205), 34, (180, 180, 180), -1)
    for angle in np.linspace(0, 2 * np.pi, 10, endpoint=False):
        point = (145 + int(19 * np.cos(angle)), 205 + int(19 * np.sin(angle)))
        cv2.circle(frame, point, 3, (20, 20, 20), -1)

    hole_count, center = punch._visible_disk_in_mask(frame, mask, 36)

    assert hole_count >= 3
    assert center is not None
    assert abs(center[0] - 130) <= 10
    assert abs(center[1] - 90) <= 10


def test_visible_disk_rejects_lower_mechanism_when_wheel_holes_occluded():
    frame = np.full((300, 420, 3), (170, 120, 70), np.uint8)
    mask = np.zeros(frame.shape[:2], bool)
    cv2.circle(frame, (130, 90), 36, (195, 195, 195), -1)
    cv2.circle(mask.view(np.uint8), (130, 90), 36, 1, -1)
    for angle in np.linspace(0, 2 * np.pi, 2, endpoint=False):
        point = (130 + int(20 * np.cos(angle)), 90 + int(20 * np.sin(angle)))
        cv2.circle(frame, point, 4, (20, 20, 20), -1)
    cv2.rectangle(frame, (105, 135), (180, 260), (170, 170, 170), -1)
    cv2.rectangle(mask.view(np.uint8), (105, 135), (180, 260), 1, -1)
    cv2.circle(frame, (145, 205), 34, (180, 180, 180), -1)
    for angle in np.linspace(0, 2 * np.pi, 10, endpoint=False):
        point = (145 + int(19 * np.cos(angle)), 205 + int(19 * np.sin(angle)))
        cv2.circle(frame, point, 3, (20, 20, 20), -1)

    hole_count, center = punch._visible_disk_in_mask(frame, mask, 36)

    assert hole_count == 0
    assert center is None


def test_final_prepunch_selection_uses_last_stable_alignment_not_transient():
    observations = [
        punch.PrePunchObservation(10.0, 1, 12.0, True),
        punch.PrePunchObservation(10.5, 1, 12.5, True),
        punch.PrePunchObservation(11.0, None, 30.0, True),
        punch.PrePunchObservation(11.5, 0, 30.5, True),
        punch.PrePunchObservation(12.0, 0, 30.5, True),
    ]

    assert punch.final_prepunch_hole(observations, contact_time_sec=12.4) == 0


def test_final_prepunch_selection_rejects_unseen_readjustment():
    observations = [
        punch.PrePunchObservation(10.0, 1, 12.0, True),
        punch.PrePunchObservation(10.5, 1, 12.5, True),
        punch.PrePunchObservation(11.0, None, 30.0, True),
        punch.PrePunchObservation(11.5, None, None, False),
    ]

    assert punch.final_prepunch_hole(observations, contact_time_sec=12.0) is None


def test_prepunch_scan_extends_to_next_stage_start_when_gap_contains_adjustment():
    from medical_evaluation.domain import TimeRange

    scan = punch.prepunch_scan_range(
        TimeRange(start_sec=19.0, end_sec=50.0),
        TimeRange(start_sec=55.0, end_sec=71.0),
    )

    assert scan == TimeRange(start_sec=19.0, end_sec=55.0)


def test_automatic_punch_box_padding_keeps_wheel_inside_prompt():
    box = punch.HeldPunchBox(0, 1040, 777, 1147, 946, 1.5, 0.6)

    coordinates = box.normalized(1920, 1080, padding_px=28)

    assert coordinates == [1012 / 1920, 749 / 1080, 1175 / 1920, 974 / 1080]
