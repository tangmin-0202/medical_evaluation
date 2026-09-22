"""Deterministic shape measurements for the CP04 displayed clamp."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ClampDisplayMeasurement:
    reliable: bool
    reason: str
    hand_proximity_ratio: float | None
    relative_area: float | None
    boundary_touch: bool
    sharpness: float | None


@dataclass(frozen=True)
class ClampMatchMeasurement:
    reliable: bool
    reason: str
    similarity: float | None
    mask_iou: float | None
    contour_similarity: float | None
    aspect_similarity: float | None


def measure_display_candidate(
    frame_bgr: np.ndarray,
    hand_mask: np.ndarray,
    clamp_mask: np.ndarray,
) -> ClampDisplayMeasurement:
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr must be an HxWx3 image")
    shape = frame_bgr.shape[:2]
    hand = _as_mask(hand_mask, shape)
    clamp = _largest_component(_as_mask(clamp_mask, shape))
    if not hand.any():
        return ClampDisplayMeasurement(False, "gloved_hand_not_observed", None, None, False, None)
    if not clamp.any():
        return ClampDisplayMeasurement(False, "clamp_not_observed", None, None, False, None)

    boundary_touch = bool(
        clamp[0].any() or clamp[-1].any() or clamp[:, 0].any() or clamp[:, -1].any()
    )
    clamp_area = int(clamp.sum())
    hand_area = int(hand.sum())
    relative_area = clamp_area / max(hand_area, 1)
    radius = max(3, round(min(shape) * 0.06))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
    )
    near_hand = cv2.dilate(hand.astype(np.uint8), kernel).astype(bool)
    proximity = float(np.logical_and(clamp, near_hand).sum() / clamp_area)
    sharpness = _masked_sharpness(frame_bgr, clamp)

    if boundary_touch:
        reason = "clamp_touches_frame_boundary"
        reliable = False
    elif proximity < 0.8:
        reason = "clamp_outside_gloved_hand"
        reliable = False
    elif not 0.002 <= relative_area <= 0.35:
        reason = "implausible_clamp_size"
        reliable = False
    else:
        reason = "reliable_display_candidate"
        reliable = True
    return ClampDisplayMeasurement(
        reliable,
        reason,
        proximity,
        relative_area,
        boundary_touch,
        sharpness,
    )


def normalize_clamp_mask(mask: np.ndarray, *, size: int = 160) -> np.ndarray:
    if size < 32:
        raise ValueError("normalization size must be at least 32")
    component = _largest_component(np.asarray(mask, dtype=bool))
    if component.sum() < 8:
        return np.zeros((size, size), dtype=bool)

    points_yx = np.argwhere(component)
    points_xy = points_yx[:, ::-1].astype(np.float32)
    centered = points_xy - points_xy.mean(axis=0, keepdims=True)
    covariance = np.cov(centered, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    major = vectors[:, int(np.argmax(values))]
    angle = np.degrees(np.arctan2(float(major[1]), float(major[0])))
    rotation = cv2.getRotationMatrix2D(
        (component.shape[1] / 2.0, component.shape[0] / 2.0),
        angle - 90.0,
        1.0,
    )
    rotated = cv2.warpAffine(
        component.astype(np.uint8),
        rotation,
        (component.shape[1], component.shape[0]),
        flags=cv2.INTER_NEAREST,
    ).astype(bool)
    cropped = _crop(rotated)
    if not cropped.any():
        return np.zeros((size, size), dtype=bool)
    margin = max(4, size // 16)
    available = size - margin * 2
    scale = min(available / cropped.shape[1], available / cropped.shape[0])
    width = max(1, round(cropped.shape[1] * scale))
    height = max(1, round(cropped.shape[0] * scale))
    resized = cv2.resize(
        cropped.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    canvas = np.zeros((size, size), dtype=bool)
    x = (size - width) // 2
    y = (size - height) // 2
    canvas[y : y + height, x : x + width] = resized
    return canvas


def compare_clamp_mask(
    mask: np.ndarray,
    reference_masks: list[np.ndarray] | tuple[np.ndarray, ...],
) -> ClampMatchMeasurement:
    candidate = normalize_clamp_mask(mask)
    if candidate.sum() < 8 or not reference_masks:
        return ClampMatchMeasurement(False, "missing_comparable_mask", None, None, None, None)

    best: tuple[float, float, float, float] | None = None
    for raw_reference in reference_masks:
        reference = normalize_clamp_mask(raw_reference)
        if reference.sum() < 8:
            continue
        for variant in _orientation_variants(candidate):
            iou = _iou(variant, reference)
            contour = _contour_similarity(variant, reference)
            aspect = _aspect_similarity(variant, reference)
            score = 0.60 * iou + 0.30 * contour + 0.10 * aspect
            metrics = (score, iou, contour, aspect)
            if best is None or metrics[0] > best[0]:
                best = metrics
    if best is None:
        return ClampMatchMeasurement(False, "missing_reference_mask", None, None, None, None)
    return ClampMatchMeasurement(
        True,
        "shape_compared",
        float(best[0]),
        float(best[1]),
        float(best[2]),
        float(best[3]),
    )


def _as_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    value = np.asarray(mask, dtype=bool)
    if value.shape != shape:
        value = cv2.resize(
            value.astype(np.uint8),
            (shape[1], shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    return value


def _largest_component(mask: np.ndarray) -> np.ndarray:
    value = np.asarray(mask, dtype=np.uint8)
    if not value.any():
        return value.astype(bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(value, connectivity=8)
    if count <= 1:
        return np.zeros_like(value, dtype=bool)
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == index


def _crop(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros((0, 0), dtype=bool)
    return mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def _orientation_variants(mask: np.ndarray) -> tuple[np.ndarray, ...]:
    rotated = np.rot90(mask, 2)
    return mask, mask[:, ::-1], mask[::-1], rotated


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    union = np.logical_or(first, second).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(first, second).sum() / union)


def _contour_similarity(first: np.ndarray, second: np.ndarray) -> float:
    first_contours, _ = cv2.findContours(
        first.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    second_contours, _ = cv2.findContours(
        second.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not first_contours or not second_contours:
        return 0.0
    distance = cv2.matchShapes(
        max(first_contours, key=cv2.contourArea),
        max(second_contours, key=cv2.contourArea),
        cv2.CONTOURS_MATCH_I1,
        0,
    )
    return float(np.exp(-3.0 * max(0.0, distance)))


def _aspect_similarity(first: np.ndarray, second: np.ndarray) -> float:
    first_crop = _crop(first)
    second_crop = _crop(second)
    if not first_crop.any() or not second_crop.any():
        return 0.0
    first_ratio = first_crop.shape[1] / first_crop.shape[0]
    second_ratio = second_crop.shape[1] / second_crop.shape[0]
    return float(min(first_ratio, second_ratio) / max(first_ratio, second_ratio))


def _masked_sharpness(frame_bgr: np.ndarray, mask: np.ndarray) -> float:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    values = laplacian[mask]
    return 0.0 if values.size == 0 else float(values.var())
