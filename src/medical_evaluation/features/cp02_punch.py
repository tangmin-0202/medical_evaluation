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


@dataclass(frozen=True)
class MovingDisk:
    frame_position: int
    x: float
    y: float
    radius: float
    hole_count: int
    motion_ratio: float

    def normalized_box(
        self, width: int, height: int, *, padding_radii: float = 1.35,
    ) -> list[float]:
        if width <= 0 or height <= 0 or padding_radii <= 1:
            raise ValueError("valid image size and disk padding are required")
        extent = self.radius * padding_radii
        return [
            max(0.0, (self.x - extent) / width),
            max(0.0, (self.y - extent) / height),
            min(1.0, (self.x + extent) / width),
            min(1.0, (self.y + extent) / height),
        ]


def locate_moving_multihole_disk(
    frames_bgr: Sequence[np.ndarray], *, min_holes: int = 3,
    min_motion_ratio: float = 0.15,
) -> MovingDisk | None:
    """Locate a moving multi-hole wheel without a manually supplied point or box."""
    if len(frames_bgr) < 3 or min_holes < 2 or not 0 < min_motion_ratio < 1:
        raise ValueError("at least three frames and valid disk thresholds are required")
    frames = [np.asarray(frame, dtype=np.uint8) for frame in frames_bgr]
    shape = frames[0].shape
    if len(shape) != 3 or shape[2] != 3 or any(frame.shape != shape for frame in frames):
        raise ValueError("all frames must be same-size BGR images")
    grays = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
    background = np.median(np.stack(grays), axis=0).astype(np.uint8)
    height, width = shape[:2]
    min_radius = max(8, round(width * 0.01))
    max_radius = max(min_radius + 2, round(width * 0.05))
    candidates: list[MovingDisk] = []
    yy, xx = np.ogrid[:height, :width]
    for frame_position, gray in enumerate(grays):
        circles = cv2.HoughCircles(
            cv2.medianBlur(gray, 7), cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=max(20, min_radius), param1=120, param2=24,
            minRadius=min_radius, maxRadius=max_radius,
        )
        if circles is None:
            continue
        for x, y, radius in np.round(circles[0]).astype(int):
            inner = (xx - x) ** 2 + (yy - y) ** 2 < (0.8 * radius) ** 2
            dark = inner & (gray < 110)
            count, _, stats, centers = cv2.connectedComponentsWithStats(
                dark.astype(np.uint8),
            )
            min_area = max(3, round(radius * radius * 0.002))
            max_area = max(min_area + 1, round(radius * radius * 0.20))
            hole_count = 0
            hole_centers = []
            for index in range(1, count):
                component_width = int(stats[index, cv2.CC_STAT_WIDTH])
                component_height = int(stats[index, cv2.CC_STAT_HEIGHT])
                component_area = int(stats[index, cv2.CC_STAT_AREA])
                aspect_ratio = max(component_width, component_height) / max(
                    1, min(component_width, component_height),
                )
                fill_ratio = component_area / (component_width * component_height)
                if (
                    min_area <= component_area <= max_area
                    and aspect_ratio <= 1.8
                    and fill_ratio >= 0.45
                ):
                    hole_count += 1
                    hole_centers.append(centers[index])
            if hole_count < min_holes:
                continue
            refined_x, refined_y = np.mean(hole_centers, axis=0)
            center_distances = np.linalg.norm(
                np.asarray(hole_centers) - np.array([refined_x, refined_y]), axis=1,
            )
            refined_radius = max(float(radius) * 0.55, float(center_distances.max()) * 1.5)
            refined_inner = (
                (xx - refined_x) ** 2 + (yy - refined_y) ** 2 < refined_radius ** 2
            )
            motion_ratio = float(
                np.mean(cv2.absdiff(gray, background)[refined_inner] > 20),
            )
            if hole_count >= min_holes and motion_ratio >= min_motion_ratio:
                candidates.append(
                    MovingDisk(
                        frame_position=frame_position,
                        x=float(refined_x),
                        y=float(refined_y),
                        radius=refined_radius,
                        hole_count=hole_count,
                        motion_ratio=motion_ratio,
                    )
                )
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item.hole_count, item.motion_ratio, item.radius))


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
