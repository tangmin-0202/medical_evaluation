"""Result-only CP03 hole geometry, pending real original-resolution ROI validation."""

import math

import cv2
import numpy as np


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
