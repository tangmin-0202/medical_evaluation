import cv2
import numpy as np

from medical_evaluation.features.tooth_floss import (
    detect_floss_line,
    locate_tooth_anchor,
    segment_tooth,
)


def _scene(*, floss_y: int | None = None) -> np.ndarray:
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    image[:] = (30, 170, 90)
    cv2.rectangle(image, (155, 105), (245, 205), (35, 35, 35), -1)
    cv2.rectangle(image, (180, 130), (220, 180), (225, 225, 225), -1)
    if floss_y is not None:
        cv2.line(image, (90, floss_y), (310, floss_y + 10), (235, 235, 235), 2)
    return image


def test_locates_white_tooth_surrounded_by_dark_clamp() -> None:
    anchor = locate_tooth_anchor([_scene()])

    assert anchor is not None
    assert anchor.score >= 0.6
    assert 175 <= anchor.center_x <= 225
    assert 145 <= anchor.center_y <= 175


def test_rejects_white_object_without_dark_clamp() -> None:
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    image[:] = (30, 170, 90)
    cv2.rectangle(image, (180, 130), (220, 180), (225, 225, 225), -1)

    assert locate_tooth_anchor([image]) is None


def test_segments_tooth_and_detects_local_thin_floss() -> None:
    image = _scene(floss_y=145)
    anchor = locate_tooth_anchor([_scene()])
    assert anchor is not None

    tooth = segment_tooth(image, anchor)
    floss = detect_floss_line(image, anchor)

    assert tooth[155, 200]
    assert tooth.sum() > 500
    assert floss is not None
    assert floss.length > 80
    assert floss.white_ratio >= 0.42


def test_does_not_detect_green_fold_as_floss() -> None:
    image = _scene()
    cv2.line(image, (90, 145), (310, 155), (20, 120, 60), 3)
    anchor = locate_tooth_anchor([_scene()])
    assert anchor is not None

    assert detect_floss_line(image, anchor) is None
