import cv2
import numpy as np
import pytest

from medical_evaluation.features import cp01_template as template


def test_target_is_third_right_lower_dot_not_third_screen_dot():
    points = [(0.7, 0.55), (0.68, 0.62), (0.65, 0.69), (0.6, 0.76)]
    result = template.select_target_dot(
        points[::-1] + [(0.3, 0.6), (0.7, 0.3)],
        orientation_reliable=True, layout_complete=True,
    )
    assert result == pytest.approx((0.65, 0.69))


def test_missing_layout_or_unknown_orientation_cannot_select_target():
    points = [(0.7, 0.55), (0.68, 0.62), (0.65, 0.69)]
    assert template.select_target_dot(
        points, orientation_reliable=True, layout_complete=False,
    ) is None
    assert template.select_target_dot(
        points, orientation_reliable=False, layout_complete=True,
    ) is None


def test_detect_dots_ignores_center_lines_and_large_dark_objects():
    image = np.full((300, 300, 3), 255, dtype=np.uint8)
    cv2.line(image, (150, 0), (150, 299), (0, 0, 0), 2)
    cv2.line(image, (0, 150), (299, 150), (0, 0, 0), 2)
    for y in (170, 195, 220):
        cv2.circle(image, (210, y), 3, (0, 0, 0), -1)
    cv2.rectangle(image, (20, 20), (70, 70), (0, 0, 0), -1)
    points = template.detect_template_dots(image)
    assert len(points) == 3


def test_new_ink_does_not_count_existing_template_dots():
    before = np.full((100, 100, 3), 180, dtype=np.uint8)
    cv2.circle(before, (60, 60), 3, (20, 20, 20), -1)
    after = before.copy()
    cv2.circle(after, (70, 70), 3, (10, 10, 10), -1)
    mask = template.new_ink_mask(before, after, np.ones((100, 100), bool))
    assert not mask[60, 60]
    assert mask[70, 70]


def test_mark_distance_uses_neighbor_spacing_and_ignores_extra_mark():
    distance = template.nearest_mark_distance(
        [(0.9, 0.1), (0.66, 0.69)], (0.65, 0.69), neighbor_spacing=0.07,
    )
    assert distance == pytest.approx(1 / 7)
    assert template.nearest_mark_distance([], (0.65, 0.69), neighbor_spacing=0.07) is None


def test_target_mapping_uses_current_registration():
    matrix = np.array([[200, 0, 10], [0, 300, 20], [0, 0, 1]], dtype=float)
    assert template.map_target((0.65, 0.69), matrix) == pytest.approx((140, 227))


def test_invalid_inputs_do_not_generate_fabricated_positions():
    with pytest.raises(ValueError):
        template.map_target((0.65, 0.69), np.zeros((3, 3)))
    with pytest.raises(ValueError):
        template.nearest_mark_distance([], (0.65, 0.69), neighbor_spacing=0)
