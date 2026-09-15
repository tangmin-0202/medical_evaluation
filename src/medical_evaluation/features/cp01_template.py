"""CP01 template-coordinate features; no annotation points are consumed."""

import math
from collections.abc import Sequence

import cv2
import numpy as np

Point = tuple[float, float]


def select_target_dot(
    points: Sequence[Point], *, orientation_reliable: bool, layout_complete: bool,
) -> Point | None:
    """Input coordinates must be rectified, upright and centered on printed axes.

    Callers must establish layout completeness separately; a mere count of three
    detected dots does not establish that the first two template dots were found.
    """
    if not orientation_reliable or not layout_complete:
        return None
    if any(not all(math.isfinite(v) and 0 <= v <= 1 for v in p) for p in points):
        raise ValueError("template points must be finite normalized coordinates")
    candidates = sorted((p for p in points if p[0] > 0.5 and p[1] > 0.5), key=lambda p: p[1])
    return candidates[2] if len(candidates) >= 3 else None


def detect_template_dots(rectified_bgr: np.ndarray) -> list[Point]:
    """Find small round dark candidates, not their anatomical identities.

    Use an upright rectified board ROI, never an entire scene or dam ROI.
    Text/layout validation is required before select_target_dot is called.
    """
    if rectified_bgr.ndim != 3 or rectified_bgr.shape[2] != 3:
        raise ValueError("expected BGR board image")
    gray = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2GRAY)
    binary = (gray < 100).astype(np.uint8)
    count, _, stats, centers = cv2.connectedComponentsWithStats(binary, 8)
    height, width = gray.shape
    points = []
    for index in range(1, count):
        _, _, w, h, area = stats[index]
        if not 3 <= area <= height * width * 0.002:
            continue
        if max(w, h) / max(1, min(w, h)) > 1.8 or area / (w * h) < 0.45:
            continue
        x, y = centers[index]
        points.append((float(x / width), float(y / height)))
    return points


def map_target(point: Point, template_to_frame: np.ndarray) -> Point:
    matrix = np.asarray(template_to_frame, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("expected finite 3x3 registration matrix")
    if abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError("singular registration matrix")
    value = matrix @ np.array([*point, 1.0])
    if abs(value[2]) < 1e-12 or not np.isfinite(value).all():
        raise ValueError("target cannot be mapped reliably")
    return float(value[0] / value[2]), float(value[1] / value[2])


def new_ink_mask(
    before_bgr: np.ndarray, after_bgr: np.ndarray, dam_mask: np.ndarray,
) -> np.ndarray:
    """Darkening candidates in registered images; not proof of ink by themselves.

    Caller must reject moving dam, shadows, glove/pen occlusion and unreliable
    illumination, and confirm persistence or a lifted-dam view before grading.
    """
    if before_bgr.shape != after_bgr.shape or before_bgr.shape[:2] != dam_mask.shape:
        raise ValueError("registered images and dam mask must have matching dimensions")
    before = cv2.cvtColor(before_bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)
    after = cv2.cvtColor(after_bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)
    return np.asarray(dam_mask, bool) & (after < 100) & ((before - after) >= 35)


def nearest_mark_distance(
    marks: Sequence[Point], target: Point, *, neighbor_spacing: float,
) -> float | None:
    if not math.isfinite(neighbor_spacing) or neighbor_spacing <= 0:
        raise ValueError("neighbor spacing must be finite and positive")
    if any(not all(math.isfinite(v) for v in p) for p in [target, *marks]):
        raise ValueError("mark coordinates must be finite")
    return min((math.dist(mark, target) / neighbor_spacing for mark in marks), default=None)
