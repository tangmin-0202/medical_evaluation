from __future__ import annotations

import cv2
import numpy as np

from medical_evaluation.features.cp04_clamp import (
    compare_clamp_mask,
    measure_display_candidate,
    normalize_clamp_mask,
)


def _clamp_mask(size: int = 220) -> np.ndarray:
    mask = np.zeros((size, size), np.uint8)
    cv2.ellipse(mask, (110, 118), (48, 62), 0, 205, 335, 1, 16)
    cv2.rectangle(mask, (49, 76), (88, 103), 1, -1)
    cv2.rectangle(mask, (132, 76), (171, 103), 1, -1)
    cv2.rectangle(mask, (67, 95), (84, 151), 1, -1)
    cv2.rectangle(mask, (136, 95), (153, 151), 1, -1)
    cv2.circle(mask, (69, 89), 7, 0, -1)
    cv2.circle(mask, (151, 89), 7, 0, -1)
    return mask.astype(bool)


def _transform(mask: np.ndarray, *, angle: float, scale: float, mirror: bool) -> np.ndarray:
    source = mask[:, ::-1] if mirror else mask
    matrix = cv2.getRotationMatrix2D((110, 110), angle, scale)
    return cv2.warpAffine(
        source.astype(np.uint8),
        matrix,
        (220, 220),
        flags=cv2.INTER_NEAREST,
    ).astype(bool)


def test_normalized_comparison_accepts_rotation_scale_and_reflection() -> None:
    reference = _clamp_mask()
    candidate = _transform(reference, angle=73, scale=0.72, mirror=True)

    measurement = compare_clamp_mask(candidate, [reference])

    assert measurement.reliable is True
    assert measurement.similarity is not None
    assert measurement.similarity >= 0.80
    assert normalize_clamp_mask(candidate).shape == (160, 160)


def test_comparison_rejects_elongated_plier_shape() -> None:
    plier = np.zeros((220, 220), np.uint8)
    cv2.rectangle(plier, (101, 20), (119, 199), 1, -1)
    cv2.circle(plier, (110, 34), 24, 1, 5)

    measurement = compare_clamp_mask(plier.astype(bool), [_clamp_mask()])

    assert measurement.reliable is True
    assert measurement.similarity is not None
    assert measurement.similarity < 0.80


def test_display_candidate_requires_clamp_to_be_on_or_near_glove() -> None:
    frame = np.full((220, 220, 3), 120, np.uint8)
    hand = np.zeros((220, 220), np.uint8)
    cv2.circle(hand, (65, 110), 50, 1, -1)
    clamp = _clamp_mask()
    clamp = cv2.resize(clamp.astype(np.uint8), (60, 60), interpolation=cv2.INTER_NEAREST)
    placed = np.zeros_like(hand)
    placed[80:140, 145:205] = clamp

    measurement = measure_display_candidate(frame, hand.astype(bool), placed.astype(bool))

    assert measurement.reliable is False
    assert measurement.reason == "clamp_outside_gloved_hand"


def test_display_candidate_rejects_boundary_clipped_mask() -> None:
    frame = np.full((220, 220, 3), 120, np.uint8)
    hand = np.ones((220, 220), bool)
    clamp = np.zeros((220, 220), np.uint8)
    clamp[:50, :50] = 1

    measurement = measure_display_candidate(frame, hand, clamp.astype(bool))

    assert measurement.reliable is False
    assert measurement.boundary_touch is True
    assert measurement.reason == "clamp_touches_frame_boundary"


def test_display_candidate_rejects_long_handled_plier_on_glove() -> None:
    frame = np.full((220, 220, 3), 120, np.uint8)
    hand = np.ones((220, 220), bool)
    plier = np.zeros((220, 220), np.uint8)
    cv2.rectangle(plier, (25, 101), (198, 119), 1, -1)
    cv2.ellipse(plier, (35, 110), (28, 34), 0, 210, 330, 1, 9)

    measurement = measure_display_candidate(frame, hand, plier.astype(bool))

    assert measurement.reliable is False
    assert measurement.reason == "elongated_non_clamp_object"
    assert measurement.elongation_ratio is not None
    assert measurement.elongation_ratio > 2.5
