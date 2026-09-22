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
    elongation_ratio: float | None


@dataclass(frozen=True)
class ClampMatchMeasurement:
    reliable: bool
    reason: str
    similarity: float | None
    mask_iou: float | None
    contour_similarity: float | None
    aspect_similarity: float | None


@dataclass(frozen=True)
class GloveMetalCandidate:
    hand_mask: np.ndarray
    object_mask: np.ndarray
    score: float


def extract_metal_candidate_from_glove(
    frame_bgr: np.ndarray,
    hand_mask: np.ndarray,
) -> GloveMetalCandidate | None:
    """Return the strongest compact non-glove object displayed on a glove."""
    candidates = extract_metal_candidates_from_glove(frame_bgr, hand_mask)
    return None if not candidates else candidates[0]


def extract_metal_candidates_from_glove(
    frame_bgr: np.ndarray,
    hand_mask: np.ndarray,
) -> list[GloveMetalCandidate]:
    """Find all plausible compact non-glove objects on segmented glove components."""
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr must be an HxWx3 image")
    shape = frame_bgr.shape[:2]
    hand = _as_mask(hand_mask, shape)
    if not hand.any():
        return []

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        hand.astype(np.uint8), connectivity=8
    )
    minimum_hand_area = max(200, round(shape[0] * shape[1] * 0.003))
    candidates: list[GloveMetalCandidate] = []
    for index in range(1, count):
        hand_area = int(stats[index, cv2.CC_STAT_AREA])
        if hand_area < minimum_hand_area:
            continue
        component = labels == index
        # A metal object resting on the glove is usually a hole in the SAM hand
        # mask. Fill only enclosed holes; filling the whole external contour
        # would also absorb the tray/background visible between spread fingers.
        filled = _fill_enclosed_holes(component)
        radius = max(2, round(min(shape) * 0.004))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
        )
        support = cv2.dilate(filled, kernel).astype(bool)

        glove_pixels = component & (gray >= np.percentile(gray[component], 35))
        if glove_pixels.sum() < 50:
            glove_pixels = component
        glove_gray = float(np.median(gray[glove_pixels]))
        hand_y = int(stats[index, cv2.CC_STAT_TOP])
        hand_h = int(stats[index, cv2.CC_STAT_HEIGHT])
        component_start = len(candidates)
        # Several darkness levels prevent a mild glove shadow from joining the
        # much darker metal into one giant component. A real object only needs
        # to survive one level; downstream multi-frame filtering removes noise.
        for darkness in (20.0, 35.0, 50.0, 65.0):
            foreground = support & (gray < glove_gray - darkness)
            join_radius = max(2, round(min(shape) * 0.007))
            foreground = cv2.morphologyEx(
                foreground.astype(np.uint8),
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (join_radius * 2 + 1, join_radius * 2 + 1),
                ),
            )
            foreground = cv2.morphologyEx(
                foreground, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
            )
            object_count, object_labels, object_stats, _ = (
                cv2.connectedComponentsWithStats(foreground, connectivity=8)
            )
            for object_index in range(1, object_count):
                object_area = int(object_stats[object_index, cv2.CC_STAT_AREA])
                relative_area = object_area / max(hand_area, 1)
                if not 0.0015 <= relative_area <= 0.12:
                    continue
                object_mask = object_labels == object_index
                overlap = float(
                    np.logical_and(object_mask, filled.astype(bool)).sum() / object_area
                )
                if overlap < 0.65:
                    continue
                object_width = int(object_stats[object_index, cv2.CC_STAT_WIDTH])
                object_height = int(object_stats[object_index, cv2.CC_STAT_HEIGHT])
                extent = object_area / max(object_width * object_height, 1)
                if extent < 0.25:
                    continue
                elongation = _elongation_ratio(object_mask)
                if elongation > 2.5:
                    continue
                ys, _ = np.nonzero(object_mask)
                center_y_ratio = (float(np.mean(ys)) - hand_y) / max(hand_h, 1)
                if center_y_ratio > 0.82:
                    continue
                area_score = max(0.0, 1.0 - abs(relative_area - 0.035) / 0.085)
                elongation_score = max(0.0, 1.0 - (elongation - 1.0) / 1.5)
                position_score = max(0.0, 1.0 - center_y_ratio)
                extent_score = min(1.0, extent / 0.6)
                score = (
                    0.4 * area_score
                    + 0.25 * elongation_score
                    + 0.2 * extent_score
                    + 0.15 * position_score
                )
                candidates.append(
                    GloveMetalCandidate(component, object_mask, float(score))
                )
        # A clamp can split into separate dark upper/lower pieces around its
        # bright center. Preserve the union as another candidate instead of
        # forcing the extractor to choose one fragment.
        local = sorted(
            candidates[component_start:],
            key=lambda item: int(item.object_mask.sum()),
            reverse=True,
        )[:16]
        merge_radius = max(3, round(min(shape) * 0.025))
        merge_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (merge_radius * 2 + 1, merge_radius * 2 + 1)
        )
        for left_index, left in enumerate(local):
            left_area = int(left.object_mask.sum())
            expanded = cv2.dilate(left.object_mask.astype(np.uint8), merge_kernel) > 0
            for right in local[left_index + 1 :]:
                right_area = int(right.object_mask.sum())
                intersection = int(np.logical_and(left.object_mask, right.object_mask).sum())
                if intersection > 0.2 * min(left_area, right_area):
                    continue
                if not np.logical_and(expanded, right.object_mask).any():
                    continue
                merged = np.logical_or(left.object_mask, right.object_mask)
                merged_area = int(merged.sum())
                relative_area = merged_area / max(hand_area, 1)
                _, _, width, height = cv2.boundingRect(merged.astype(np.uint8))
                extent = merged_area / max(width * height, 1)
                elongation = _elongation_ratio(merged)
                if not 0.003 <= relative_area <= 0.12 or extent < 0.20 or elongation > 2.5:
                    continue
                center_y_ratio = (float(np.nonzero(merged)[0].mean()) - hand_y) / max(
                    hand_h, 1
                )
                if center_y_ratio > 0.82:
                    continue
                area_score = max(0.0, 1.0 - abs(relative_area - 0.035) / 0.085)
                elongation_score = max(0.0, 1.0 - (elongation - 1.0) / 1.5)
                extent_score = min(1.0, extent / 0.6)
                position_score = max(0.0, 1.0 - center_y_ratio)
                score = (
                    0.4 * area_score
                    + 0.25 * elongation_score
                    + 0.2 * extent_score
                    + 0.15 * position_score
                    + 0.05
                )
                candidates.append(GloveMetalCandidate(component, merged, float(score)))
    return sorted(candidates, key=lambda item: item.score, reverse=True)


def _fill_enclosed_holes(mask: np.ndarray) -> np.ndarray:
    value = mask.astype(np.uint8)
    padded = cv2.copyMakeBorder(value, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    outside = (padded == 0).astype(np.uint8)
    flood = outside.copy()
    flood_mask = np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 2)
    holes = (flood[1:-1, 1:-1] == 1).astype(np.uint8)
    return cv2.bitwise_or(value, holes)


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
        return ClampDisplayMeasurement(
            False, "gloved_hand_not_observed", None, None, False, None, None
        )
    if not clamp.any():
        return ClampDisplayMeasurement(
            False, "clamp_not_observed", None, None, False, None, None
        )

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
    elongation_ratio = _elongation_ratio(clamp)

    if boundary_touch:
        reason = "clamp_touches_frame_boundary"
        reliable = False
    elif proximity < 0.8:
        reason = "clamp_outside_gloved_hand"
        reliable = False
    elif not 0.002 <= relative_area <= 0.35:
        reason = "implausible_clamp_size"
        reliable = False
    elif elongation_ratio > 2.5:
        reason = "elongated_non_clamp_object"
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
        elongation_ratio,
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


def _elongation_ratio(mask: np.ndarray) -> float:
    points_yx = np.argwhere(mask)
    if len(points_yx) < 3:
        return float("inf")
    rectangle = cv2.minAreaRect(points_yx[:, ::-1].astype(np.float32))
    width, height = rectangle[1]
    shorter = min(width, height)
    return float("inf") if shorter <= 0 else float(max(width, height) / shorter)


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
