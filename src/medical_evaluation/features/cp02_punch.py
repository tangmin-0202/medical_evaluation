"""CP02 local punch-disk features; requires rectified, validated object ROIs."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Hole:
    x: float
    y: float
    radius: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in (self.x, self.y, self.radius)) or self.radius <= 0:
            raise ValueError("hole geometry must be finite with positive radius")


def second_largest_hole(
    holes: Sequence[Hole], *, layout_reliable: bool, min_radius_gap: float,
) -> int | None:
    """Radii must come from a common metric rectified disk plane, not screen ellipses."""
    if not math.isfinite(min_radius_gap) or min_radius_gap <= 0:
        raise ValueError("positive calibrated size separation required")
    if not layout_reliable or len(holes) < 2:
        return None
    order = sorted(range(len(holes)), key=lambda i: holes[i].radius, reverse=True)
    if holes[order[0]].radius - holes[order[1]].radius < min_radius_gap:
        return None
    if len(order) > 2 and holes[order[1]].radius - holes[order[2]].radius < min_radius_gap:
        return None
    return order[1]


def aligned_hole(
    holes: Sequence[Hole], tip: tuple[float, float], *, max_distance_in_radii: float,
) -> int | None:
    if not all(math.isfinite(v) for v in tip) or max_distance_in_radii <= 0:
        raise ValueError("invalid tip or alignment tolerance")
    matches = [i for i, h in enumerate(holes)
               if math.dist((h.x, h.y), tip) / h.radius <= max_distance_in_radii]
    return matches[0] if len(matches) == 1 else None


def dam_color_ratio(
    frame_bgr: np.ndarray, hole_interior: np.ndarray, dam_reference: np.ndarray,
    *, max_lab_distance: float,
) -> float | None:
    """Same-frame color candidate ratio; not a residue label without visibility validation."""
    if frame_bgr.shape[:2] != hole_interior.shape or hole_interior.shape != dam_reference.shape:
        raise ValueError("frame and mask dimensions must match")
    if not math.isfinite(max_lab_distance) or max_lab_distance <= 0:
        raise ValueError("positive color tolerance required")
    interior = np.asarray(hole_interior, bool)
    reference = np.asarray(dam_reference, bool)
    if not interior.any() or not reference.any():
        return None
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(float)
    center = np.median(lab[reference], axis=0)
    distance = np.linalg.norm(lab[interior] - center, axis=1)
    return float(np.mean(distance <= max_lab_distance))
