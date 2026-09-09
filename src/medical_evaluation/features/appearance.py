from __future__ import annotations

import cv2
import numpy as np


def green_dam_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """Return pixels whose HSV appearance is confidently rubber-dam green."""
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or frame_bgr.size == 0:
        raise ValueError("frame must be a non-empty BGR image")
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    return (
        (hsv[..., 0] >= 35)
        & (hsv[..., 0] <= 95)
        & (hsv[..., 1] >= 50)
        & (hsv[..., 2] >= 40)
    )


def green_dam_area_ratio(frame_bgr: np.ndarray) -> float:
    """Return the fraction of the readable frame confidently green."""

    return float(green_dam_mask(frame_bgr).mean())


def box_overlap_ratio(
    mask: np.ndarray,
    normalized_box: tuple[float, float, float, float],
) -> float:
    """Return mask intersection divided by the normalized box pixel area."""

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    x1, y1, x2, y2 = normalized_box
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError("box must be ordered and normalized")
    height, width = binary.shape
    left, right = round(x1 * width), round(x2 * width)
    top, bottom = round(y1 * height), round(y2 * height)
    box_area = max((right - left) * (bottom - top), 1)
    return float(binary[top:bottom, left:right].sum() / box_area)


def visible_reference_area_ratio(
    frame_bgr: np.ndarray,
    candidate_mask: np.ndarray,
    reference_lab: np.ndarray,
    *,
    max_lab_distance: float,
) -> float:
    """Measure candidate pixels whose Lab appearance still matches a visible frame."""

    return float(
        visible_reference_mask(
            frame_bgr,
            candidate_mask,
            reference_lab,
            max_lab_distance=max_lab_distance,
        ).mean()
    )


def visible_reference_mask(
    frame_bgr: np.ndarray,
    candidate_mask: np.ndarray,
    reference_lab: np.ndarray,
    *,
    max_lab_distance: float,
) -> np.ndarray:
    """Return candidate pixels that still look like the CP09 white frame."""

    candidate = _region(frame_bgr, candidate_mask)
    reference = np.asarray(reference_lab, dtype=float).reshape(-1)
    if reference.shape != (3,):
        raise ValueError("reference_lab must have three channels")
    if max_lab_distance <= 0:
        raise ValueError("max_lab_distance must be positive")
    if not candidate.any():
        return np.zeros(candidate.shape, dtype=bool)
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(float)
    distance = np.linalg.norm(lab - reference, axis=2)
    return candidate & (distance <= max_lab_distance)


def visible_white_frame_near_dam_edge(
    frame_bgr: np.ndarray,
    dam_mask: np.ndarray,
    *,
    boundary_width_ratio: float = 0.015,
    minimum_component_area_ratio: float = 0.00005,
) -> tuple[np.ndarray, np.ndarray]:
    """Find non-border white components in a narrow band around the dam edge."""

    dam = _region(frame_bgr, dam_mask)
    if not 0 < boundary_width_ratio < 0.5:
        raise ValueError("boundary_width_ratio must be between zero and 0.5")
    if minimum_component_area_ratio <= 0:
        raise ValueError("minimum_component_area_ratio must be positive")
    height, width = dam.shape
    contours, _ = cv2.findContours(
        dam.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    exterior = np.zeros_like(dam, dtype=np.uint8)
    cv2.drawContours(exterior, contours, -1, 1, cv2.FILLED)
    dam_exterior = exterior.astype(bool)
    radius = max(2, round(min(height, width) * boundary_width_ratio))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
    )
    dilated = cv2.dilate(dam_exterior.astype(np.uint8), kernel).astype(bool)
    eroded = cv2.erode(dam_exterior.astype(np.uint8), kernel).astype(bool)
    search_band = dilated & ~eroded

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    white = (hsv[..., 1] <= 70) & (hsv[..., 2] >= 160)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        white.astype(np.uint8), 8
    )
    visible = np.zeros_like(dam)
    minimum_area = max(1, round(height * width * minimum_component_area_ratio))
    for label in range(1, count):
        x, y, component_width, component_height, area = map(int, stats[label])
        touches_border = (
            x == 0
            or y == 0
            or x + component_width >= width
            or y + component_height >= height
        )
        if touches_border or area < minimum_area:
            continue
        component = labels == label
        visible |= component & search_band
    return visible, search_band


def color_ratio_hsv(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    hue_range: tuple[int, int],
    minimum_saturation: int = 40,
    minimum_value: int = 30,
) -> float | None:
    binary = _region(frame_bgr, mask)
    if not binary.any():
        return None
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    selected = hsv[binary]
    lower_hue, upper_hue = hue_range
    matches = (
        (selected[:, 0] >= lower_hue)
        & (selected[:, 0] <= upper_hue)
        & (selected[:, 1] >= minimum_saturation)
        & (selected[:, 2] >= minimum_value)
    )
    return float(matches.mean())


def dark_ratio(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    maximum_value: int = 55,
) -> float | None:
    binary = _region(frame_bgr, mask)
    if not binary.any():
        return None
    value = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)[..., 2]
    return float((value[binary] <= maximum_value).mean())


def blur_score(frame_bgr: np.ndarray, mask: np.ndarray) -> float | None:
    binary = _region(frame_bgr, mask)
    if not binary.any():
        return None
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian[binary].var())


def reference_cosine_similarity(
    embedding: np.ndarray,
    reference: np.ndarray,
) -> float | None:
    first = np.asarray(embedding, dtype=np.float32).ravel()
    second = np.asarray(reference, dtype=np.float32).ravel()
    if first.shape != second.shape:
        raise ValueError("embedding dimensions must match")
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator == 0:
        return None
    return float(np.clip(np.dot(first, second) / denominator, -1, 1))


def _region(frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame must be a BGR image")
    binary = np.asarray(mask, dtype=bool)
    if binary.shape != frame_bgr.shape[:2]:
        raise ValueError("mask and frame dimensions must match")
    return binary
