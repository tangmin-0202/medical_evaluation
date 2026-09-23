from __future__ import annotations

import cv2
import numpy as np

from medical_evaluation.features.cp04_clamp import (
    GloveMetalCandidate,
    compare_clamp_mask,
    extract_metal_candidate_from_glove,
    measure_display_candidate,
    normalize_clamp_mask,
    select_metal_candidate_near_seed,
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


def _glove_scene(*, object_center: tuple[int, int] | None) -> tuple[np.ndarray, np.ndarray]:
    frame = np.full((240, 320, 3), (180, 110, 45), np.uint8)
    hand = np.zeros(frame.shape[:2], np.uint8)
    cv2.ellipse(hand, (145, 145), (72, 82), 0, 0, 360, 1, -1)
    for x in (105, 130, 155, 180):
        cv2.rectangle(hand, (x, 35), (x + 22, 135), 1, -1)
    frame[hand.astype(bool)] = (220, 225, 230)
    if object_center is not None:
        x, y = object_center
        cv2.ellipse(frame, (x, y), (22, 16), 0, 0, 360, (82, 86, 90), -1)
        cv2.circle(frame, (x, y), 7, (220, 225, 230), -1)
    return frame, hand.astype(bool)


def test_extracts_compact_metal_object_from_gloved_palm() -> None:
    frame, hand = _glove_scene(object_center=(145, 145))

    candidate = extract_metal_candidate_from_glove(frame, hand)

    assert candidate is not None
    assert candidate.object_mask[145, 125:166].any()
    assert candidate.hand_mask[145, 145]


def test_extracts_metal_object_displayed_on_gloved_fingers() -> None:
    frame, hand = _glove_scene(object_center=(155, 82))

    candidate = extract_metal_candidate_from_glove(frame, hand)

    assert candidate is not None
    assert candidate.object_mask[82, 135:176].any()


def test_extracts_bright_neutral_metal_against_warm_glove() -> None:
    frame, hand = _glove_scene(object_center=None)
    cv2.ellipse(frame, (145, 145), (24, 18), 0, 0, 360, (185, 185, 185), -1)
    cv2.circle(frame, (145, 145), 8, (220, 225, 230), -1)

    candidate = extract_metal_candidate_from_glove(frame, hand)

    assert candidate is not None
    assert candidate.object_mask[145, 121:170].any()


def test_reconstructs_split_reflective_clamp_instead_of_one_fragment() -> None:
    frame, hand = _glove_scene(object_center=None)
    metal = (190, 190, 190)
    shadowed_metal = (65, 92, 122)
    cv2.rectangle(frame, (125, 105), (165, 123), metal, -1)
    cv2.rectangle(frame, (125, 150), (165, 168), metal, -1)
    cv2.rectangle(frame, (139, 120), (151, 153), shadowed_metal, -1)
    cv2.circle(frame, (136, 114), 4, (220, 225, 230), -1)
    cv2.circle(frame, (155, 159), 4, (220, 225, 230), -1)

    candidate = extract_metal_candidate_from_glove(frame, hand)

    assert candidate is not None
    assert candidate.object_mask[110, 128]
    assert candidate.object_mask[165, 162]
    assert candidate.object_mask[140, 145]


def test_seeded_selection_rejects_larger_distant_glove_shadow() -> None:
    shape = (240, 320)
    hand = np.ones(shape, bool)
    clamp = np.zeros(shape, bool)
    clamp[70:115, 130:180] = True
    shadow = np.zeros(shape, bool)
    shadow[145:215, 35:115] = True
    seed = np.zeros(shape, bool)
    seed[88:96, 148:156] = True
    candidates = [
        GloveMetalCandidate(hand, shadow, 0.95),
        GloveMetalCandidate(hand, clamp, 0.72),
    ]

    selected = select_metal_candidate_near_seed(candidates, seed)

    assert selected is not None
    assert np.array_equal(selected.object_mask, clamp)


def test_extracts_object_excluded_as_hole_from_sam_hand_mask() -> None:
    frame, hand = _glove_scene(object_center=(145, 145))
    cv2.ellipse(hand.view(np.uint8), (145, 145), (25, 19), 0, 0, 360, 0, -1)

    candidate = extract_metal_candidate_from_glove(frame, hand)

    assert candidate is not None
    assert candidate.object_mask[145, 123:168].any()


def test_ignores_metal_plier_outside_glove() -> None:
    frame, hand = _glove_scene(object_center=None)
    cv2.rectangle(frame, (260, 45), (275, 210), (75, 80, 85), -1)

    assert extract_metal_candidate_from_glove(frame, hand) is None
