from __future__ import annotations

import cv2
import numpy as np
import pytest

from medical_evaluation.features.mannequin_registration import (
    RegistrationLimits,
    compose_similarity,
    estimate_similarity_registration,
    invert_similarity,
    registration_is_continuous,
    transform_points,
    warp_mask,
)


def _similarity(
    *, angle_deg: float = 8.0, scale: float = 1.08, tx: float = 13.0, ty: float = -7.0
) -> np.ndarray:
    theta = np.deg2rad(angle_deg)
    return np.asarray(
        [
            [scale * np.cos(theta), -scale * np.sin(theta), tx],
            [scale * np.sin(theta), scale * np.cos(theta), ty],
        ],
        dtype=np.float64,
    )


def _textured_pair() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = 240, 320
    image = np.zeros((height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(mask, (155, 125), (100, 78), 12, 0, 360, 255, -1)
    rng = np.random.default_rng(2408)
    for _ in range(180):
        x = int(rng.integers(65, 245))
        y = int(rng.integers(55, 195))
        if mask[y, x]:
            colour = tuple(int(value) for value in rng.integers(70, 250, size=3))
            cv2.circle(image, (x, y), int(rng.integers(2, 5)), colour, -1)
    cv2.rectangle(image, (92, 92), (122, 119), (255, 255, 255), 3)
    cv2.line(image, (175, 70), (222, 142), (80, 220, 250), 4)
    image[mask == 0] = 15

    matrix = _similarity()
    target = cv2.warpAffine(image, matrix, (width, height))
    target_mask = cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_NEAREST)
    return image, mask.astype(bool), target, target_mask.astype(bool), matrix


def test_estimates_similarity_transform_from_masked_image_structure() -> None:
    reference, reference_mask, target, target_mask, expected = _textured_pair()

    result = estimate_similarity_registration(
        reference,
        reference_mask,
        target,
        target_mask,
        RegistrationLimits(min_matches=8, min_inliers=6, min_mask_iou=0.75),
    )

    assert result.accepted, result.reason
    probe = np.asarray([[100.0, 80.0], [155.0, 125.0], [220.0, 170.0]])
    assert transform_points(probe, result.matrix) == pytest.approx(
        transform_points(probe, expected), abs=1.5
    )
    assert result.inlier_count >= 6
    assert result.mask_iou >= 0.75
    assert result.residual_px <= 2.0


def test_inverse_and_composition_map_cp09_region_into_cp11() -> None:
    cp09 = _similarity(angle_deg=7, scale=1.05, tx=12, ty=-3)
    cp11 = _similarity(angle_deg=9, scale=1.02, tx=18, ty=4)
    cp09_points = np.asarray([[120.0, 70.0], [160.0, 100.0], [190.0, 150.0]])

    mapped = transform_points(
        cp09_points,
        compose_similarity(cp11, invert_similarity(cp09)),
    )
    template_points = transform_points(cp09_points, invert_similarity(cp09))

    assert mapped == pytest.approx(transform_points(template_points, cp11), abs=1e-7)


def test_warp_mask_uses_nearest_neighbour_and_preserves_boolean_type() -> None:
    mask = np.zeros((60, 80), dtype=bool)
    mask[20:40, 30:50] = True
    matrix = _similarity(angle_deg=0, scale=1, tx=5, ty=-4)

    warped = warp_mask(mask, matrix, output_shape=(60, 80))

    assert warped.dtype == np.bool_
    assert warped[16:36, 35:55].all()
    assert int(warped.sum()) == int(mask.sum())


def test_registration_rejects_textureless_evidence() -> None:
    image = np.full((120, 160, 3), 90, dtype=np.uint8)
    mask = np.zeros((120, 160), dtype=bool)
    mask[20:100, 30:130] = True

    result = estimate_similarity_registration(image, mask, image, mask)

    assert not result.accepted
    assert result.reason == "insufficient_feature_matches"


def test_search_region_expands_with_image_size_across_changing_visible_fragments() -> None:
    reference, full_mask, target, _target_mask, expected = _textured_pair()
    reference_mask = np.zeros_like(full_mask)
    reference_mask[45:105, 55:255] = full_mask[45:105, 55:255]
    transformed_full = warp_mask(full_mask, expected, output_shape=full_mask.shape)
    target_mask = np.zeros_like(full_mask)
    target_mask[145:220, 55:285] = transformed_full[145:220, 55:285]

    result = estimate_similarity_registration(
        reference,
        reference_mask,
        target,
        target_mask,
        RegistrationLimits(
            min_matches=8,
            min_inliers=6,
            min_mask_iou=0.0,
            search_dilation_diagonal_ratio=0.18,
        ),
    )

    assert result.accepted, result.reason
    assert result.match_count >= 8


def test_large_images_are_downsampled_without_changing_full_resolution_transform() -> None:
    reference, reference_mask, target, target_mask, expected = _textured_pair()
    reference = cv2.resize(reference, (1280, 960), interpolation=cv2.INTER_LINEAR)
    target = cv2.resize(target, (1280, 960), interpolation=cv2.INTER_LINEAR)
    reference_mask = cv2.resize(
        reference_mask.astype(np.uint8), (1280, 960), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    target_mask = cv2.resize(
        target_mask.astype(np.uint8), (1280, 960), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    expected = expected.copy()
    expected[:, 2] *= 4.0

    result = estimate_similarity_registration(
        reference,
        reference_mask,
        target,
        target_mask,
        RegistrationLimits(
            min_matches=8,
            min_inliers=6,
            min_mask_iou=0.70,
            analysis_max_width=640,
        ),
    )

    probe = np.asarray([[400.0, 320.0], [620.0, 500.0], [880.0, 680.0]])
    assert result.accepted, result.reason
    assert transform_points(probe, result.matrix) == pytest.approx(
        transform_points(probe, expected), abs=4.0
    )


@pytest.mark.parametrize(
    ("current", "expected"),
    [
        (_similarity(angle_deg=3, scale=1.02, tx=3, ty=2), True),
        (_similarity(angle_deg=18, scale=1.02, tx=3, ty=2), False),
        (_similarity(angle_deg=3, scale=1.30, tx=3, ty=2), False),
        (_similarity(angle_deg=3, scale=1.02, tx=80, ty=2), False),
    ],
)
def test_temporal_quality_gate_rejects_transform_jumps(
    current: np.ndarray, expected: bool
) -> None:
    assert registration_is_continuous(
        np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        current,
        image_shape=(240, 320),
        max_angle_delta_deg=6.0,
        max_scale_delta=0.10,
        max_translation_diagonal_ratio=0.12,
    ) is expected
