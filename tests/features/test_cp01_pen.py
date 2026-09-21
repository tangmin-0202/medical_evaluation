import cv2
import numpy as np

from medical_evaluation.features.cp01_pen import (
    measure_pen_candidate,
    segment_pen_candidate,
)


def _frame_and_mask(*, value: int = 25) -> tuple[np.ndarray, np.ndarray]:
    frame = np.full((120, 180, 3), 220, dtype=np.uint8)
    mask = np.zeros((120, 180), dtype=bool)
    mask[48:68, 25:155] = True
    frame[mask] = value
    return frame, mask


def test_empty_mask_is_not_observed() -> None:
    frame = np.full((80, 120, 3), 220, dtype=np.uint8)
    result = measure_pen_candidate(frame, np.zeros((80, 120), dtype=bool))

    assert result.observed is False
    assert result.accepted is False
    assert result.reason == "pen_not_observed"


def test_bright_elongated_region_is_not_a_black_pen() -> None:
    frame, mask = _frame_and_mask(value=210)
    result = measure_pen_candidate(frame, mask)

    assert result.observed is True
    assert result.accepted is False
    assert result.reason == "pen_appearance_mismatch"


def test_fragmented_dark_mask_is_rejected() -> None:
    frame = np.full((120, 180, 3), 220, dtype=np.uint8)
    mask = np.zeros((120, 180), dtype=bool)
    cv2.rectangle(mask.view(np.uint8), (15, 45), (75, 65), 1, -1)
    cv2.rectangle(mask.view(np.uint8), (105, 45), (165, 65), 1, -1)
    frame[mask] = 20

    result = measure_pen_candidate(frame, mask)

    assert result.observed is True
    assert result.accepted is False
    assert result.reason == "fragmented_pen_mask"


def test_dark_dominant_elongated_mask_is_accepted() -> None:
    frame, mask = _frame_and_mask()
    result = measure_pen_candidate(frame, mask)

    assert result.observed is True
    assert result.accepted is True
    assert result.reason == "accepted"
    assert result.area_px == int(mask.sum())
    assert result.dark_pixel_ratio == 1.0
    assert result.dominant_component_ratio == 1.0
    assert result.elongation > 5.0


def test_opencv_segments_black_pen_over_green_dam() -> None:
    frame = np.full((240, 360, 3), (180, 150, 120), dtype=np.uint8)
    cv2.rectangle(frame, (70, 55), (300, 205), (45, 155, 55), -1)
    cv2.rectangle(frame, (125, 118), (285, 142), (25, 25, 25), -1)

    mask = segment_pen_candidate(frame)
    result = measure_pen_candidate(frame, mask)

    assert result.accepted is True
    assert mask[130, 180]


def test_opencv_ignores_template_cross_and_border_sleeve() -> None:
    frame = np.full((240, 360, 3), (180, 150, 120), dtype=np.uint8)
    cv2.rectangle(frame, (70, 55), (300, 205), (45, 155, 55), -1)
    cv2.line(frame, (180, 60), (180, 200), (20, 20, 20), 2)
    cv2.rectangle(frame, (0, 170), (115, 239), (20, 20, 20), -1)

    mask = segment_pen_candidate(frame)

    assert not mask.any()
