"""Deterministic appearance checks for CP01 marking-pen masks."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


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
