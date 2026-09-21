"""Deterministic appearance checks for CP01 marking-pen masks."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def segment_pen_candidate(frame_bgr: np.ndarray) -> np.ndarray:
    """Segment the most plausible dark, elongated pen beside the green dam."""
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr must be an HxWx3 BGR image")
    height, width = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (30, 35, 25), (100, 255, 255)) > 0
    if int(green.sum()) < max(100, int(height * width * 0.01)):
        return np.zeros((height, width), dtype=bool)
    green_count, green_labels, green_stats, _ = cv2.connectedComponentsWithStats(
        green.astype(np.uint8), connectivity=8
    )
    if green_count <= 1:
        return np.zeros((height, width), dtype=bool)
    green_areas = green_stats[1:, cv2.CC_STAT_AREA]
    main_green_label = int(np.argmax(green_areas)) + 1
    main_green = green_labels == main_green_label

    value = hsv[..., 2]
    dark = (value < 105).astype(np.uint8)
    close_size = max(3, round(min(height, width) * 0.025))
    if close_size % 2 == 0:
        close_size += 1
    connected = cv2.morphologyEx(
        dark,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size)),
    )
    connected = cv2.morphologyEx(
        connected,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    proximity_size = max(5, round(min(height, width) * 0.10))
    near_green = cv2.dilate(
        main_green.astype(np.uint8),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (proximity_size | 1, proximity_size | 1)
        ),
    ).astype(bool)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(connected, 8)
    frame_area = height * width
    diagonal = float(np.hypot(height, width))
    best_score = -1.0
    best = np.zeros((height, width), dtype=bool)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if not max(25, frame_area * 0.00015) <= area <= frame_area * 0.06:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        component_width = int(stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if x == 0 or y == 0 or x + component_width >= width or y + component_height >= height:
            continue
        component = labels == label
        if not np.any(component & near_green):
            continue
        contours, _ = cv2.findContours(
            component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        box_width, box_height = cv2.minAreaRect(max(contours, key=cv2.contourArea))[1]
        long_side = max(box_width, box_height)
        short_side = min(box_width, box_height)
        elongation = long_side / max(short_side, 1.0)
        if elongation < 2.2 or long_side < diagonal * 0.10:
            continue
        if short_side < max(4.0, diagonal * 0.008) or short_side > diagonal * 0.10:
            continue
        original_dark_ratio = float(np.mean(value[component] < 105))
        if original_dark_ratio < 0.55:
            continue
        score = elongation * np.sqrt(area) * original_dark_ratio
        if score > best_score:
            best_score = score
            best = component
    return best


@dataclass(frozen=True)
class PenCandidateMeasurement:
    observed: bool
    accepted: bool
    reason: str
    area_px: int
    dark_pixel_ratio: float
    dominant_component_ratio: float
    elongation: float
    selected_mask: np.ndarray


def measure_pen_candidate(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    minimum_area_px: int = 25,
    minimum_dark_pixel_ratio: float = 0.25,
    minimum_dominant_component_ratio: float = 0.70,
    minimum_elongation: float = 1.5,
) -> PenCandidateMeasurement:
    """Validate that a SAM candidate visibly resembles a dark pen body."""
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr must be an HxWx3 BGR image")
    binary = np.asarray(mask, dtype=bool)
    if binary.shape != frame_bgr.shape[:2]:
        raise ValueError("pen mask shape must match the frame")
    if minimum_area_px <= 0:
        raise ValueError("minimum_area_px must be positive")
    if not 0 <= minimum_dark_pixel_ratio <= 1:
        raise ValueError("minimum_dark_pixel_ratio must be between zero and one")
    if not 0 < minimum_dominant_component_ratio <= 1:
        raise ValueError(
            "minimum_dominant_component_ratio must be between zero and one"
        )
    if minimum_elongation < 1:
        raise ValueError("minimum_elongation must be at least one")

    area = int(binary.sum())
    empty = np.zeros(binary.shape, dtype=bool)
    if area == 0:
        return PenCandidateMeasurement(
            False, False, "pen_not_observed", 0, 0.0, 0.0, 0.0, empty
        )

    _count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    component_areas = stats[1:, cv2.CC_STAT_AREA]
    dominant_label = int(np.argmax(component_areas)) + 1
    dominant_area = int(component_areas[dominant_label - 1])
    selected = labels == dominant_label
    dominant_ratio = dominant_area / area

    contours, _ = cv2.findContours(
        selected.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    rectangle = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    width, height = rectangle[1]
    elongation = max(width, height) / max(min(width, height), 1.0)

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    dark_ratio = float(np.mean(hsv[..., 2][selected] < 105))

    if dominant_area < minimum_area_px:
        reason = "pen_mask_too_small"
    elif dominant_ratio < minimum_dominant_component_ratio:
        reason = "fragmented_pen_mask"
    elif dark_ratio < minimum_dark_pixel_ratio:
        reason = "pen_appearance_mismatch"
    elif elongation < minimum_elongation:
        reason = "pen_shape_mismatch"
    else:
        reason = "accepted"
    return PenCandidateMeasurement(
        observed=True,
        accepted=reason == "accepted",
        reason=reason,
        area_px=area,
        dark_pixel_ratio=dark_ratio,
        dominant_component_ratio=dominant_ratio,
        elongation=float(elongation),
        selected_mask=selected,
    )
