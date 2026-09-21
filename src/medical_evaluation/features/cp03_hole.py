"""Result-only CP03 hole geometry, pending real original-resolution ROI validation."""

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class HoleAdhesionAnalysis:
    """Deterministic observation of one internal opening in a dam mask."""

    reliable: bool
    hole_observed: bool
    adhesion_detected: bool | None
    center_xy: tuple[float, float] | None
    radius_px: float | None
    adhesion_area_ratio: float
    hole_mask: np.ndarray
    adhesion_mask: np.ndarray


def analyze_hole_adhesion(
    image_bgr: np.ndarray,
    dam_mask: np.ndarray,
    *,
    min_hole_area_px: int = 20,
    min_adhesion_area_ratio: float = 0.03,
) -> HoleAdhesionAnalysis:
    """Find a closed opening and detect dam material connected into its hull.

    Shape measurements locate a usable opening only.  A positive adhesion result
    requires pixels from the dominant dam component to intrude into the convex
    hull of that opening; disconnected specks inside the opening are ignored.
    """
    image = np.asarray(image_bgr)
    binary = np.asarray(dam_mask, dtype=bool)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected a BGR image")
    if binary.ndim != 2 or image.shape[:2] != binary.shape:
        raise ValueError("dam mask must match the image height and width")
    if min_hole_area_px < 9:
        raise ValueError("min_hole_area_px must be at least 9")
    if not 0 < min_adhesion_area_ratio < 1:
        raise ValueError("min_adhesion_area_ratio must be between zero and one")

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    empty = np.zeros(binary.shape, dtype=bool)
    if count <= 1:
        return HoleAdhesionAnalysis(
            reliable=False,
            hole_observed=False,
            adhesion_detected=None,
            center_xy=None,
            radius_px=None,
            adhesion_area_ratio=0.0,
            hole_mask=empty,
            adhesion_mask=empty.copy(),
        )
    main_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    main_dam = labels == main_label
    reliable = bool(main_dam.sum() >= binary.size * 0.05)
    contours, _ = cv2.findContours(
        main_dam.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not reliable or not contours:
        return HoleAdhesionAnalysis(
            reliable=False,
            hole_observed=False,
            adhesion_detected=None,
            center_xy=None,
            radius_px=None,
            adhesion_area_ratio=0.0,
            hole_mask=empty,
            adhesion_mask=empty.copy(),
        )

    outer = max(contours, key=cv2.contourArea)
    outer_fill = np.zeros(binary.shape, np.uint8)
    cv2.drawContours(outer_fill, [outer], -1, 1, thickness=-1)
    openings = outer_fill.astype(bool) & ~main_dam
    candidate_count, candidate_labels, candidate_stats, _ = (
        cv2.connectedComponentsWithStats(openings.astype(np.uint8), connectivity=8)
    )
    candidates: list[tuple[int, int]] = []
    height, width = binary.shape
    for label in range(1, candidate_count):
        x, y, w, h, area = candidate_stats[label]
        touches_edge = x == 0 or y == 0 or x + w >= width or y + h >= height
        if area >= min_hole_area_px and not touches_edge:
            candidates.append((int(area), label))
    if not candidates:
        return HoleAdhesionAnalysis(
            reliable=reliable,
            hole_observed=False,
            adhesion_detected=None,
            center_xy=None,
            radius_px=None,
            adhesion_area_ratio=0.0,
            hole_mask=empty,
            adhesion_mask=empty.copy(),
        )

    _, selected_label = max(candidates)
    hole_mask = candidate_labels == selected_label
    hole_contours, _ = cv2.findContours(
        hole_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    contour = max(hole_contours, key=cv2.contourArea)
    hull_mask = np.zeros(binary.shape, np.uint8)
    cv2.drawContours(hull_mask, [cv2.convexHull(contour)], -1, 1, thickness=-1)
    hull = hull_mask.astype(bool)
    adhesion_mask = hull & main_dam
    hull_area = int(hull.sum())
    adhesion_ratio = float(adhesion_mask.sum() / hull_area) if hull_area else 0.0
    moments = cv2.moments(hole_mask.astype(np.uint8), binaryImage=True)
    center = (
        float(moments["m10"] / moments["m00"]),
        float(moments["m01"] / moments["m00"]),
    )
    radius = math.sqrt(float(hole_mask.sum()) / math.pi)
    return HoleAdhesionAnalysis(
        reliable=True,
        hole_observed=True,
        adhesion_detected=adhesion_ratio >= min_adhesion_area_ratio,
        center_xy=center,
        radius_px=radius,
        adhesion_area_ratio=adhesion_ratio,
        hole_mask=hole_mask,
        adhesion_mask=adhesion_mask,
    )


def measure_hole(hole_mask: np.ndarray) -> dict[str, float | int | bool] | None:
    """Measure a candidate opening mask, not a dam-mask gap or a bright spot.

    Caller must confirm the ROI is unobscured and flat/rectified and the mask
    represents a through-hole. Solidity alone does not prove absence of flaps.
    """
    binary = np.asarray(hole_mask, dtype=bool)
    if binary.ndim != 2 or min(binary.shape) < 2:
        raise ValueError("expected a two-dimensional hole ROI mask")
    if binary.sum() < 9:
        return None
    contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    if area <= 0 or perimeter <= 0 or hull_area <= 0 or len(contour) < 5:
        return None
    _, axes, _ = cv2.fitEllipse(contour)
    return {
        "area_px": int(binary.sum()), "component_count": len(contours),
        "circularity": float(4 * math.pi * area / perimeter ** 2),
        "solidity": area / hull_area,
        "axis_ratio": float(min(axes) / max(axes)),
        "touches_roi_edge": bool(binary[0].any() or binary[-1].any()
                                 or binary[:, 0].any() or binary[:, -1].any()),
    }
