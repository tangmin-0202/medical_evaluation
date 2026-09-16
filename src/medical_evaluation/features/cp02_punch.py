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


@dataclass(frozen=True)
class HeldPunchBox:
    frame_position: int
    x1: int
    y1: int
    x2: int
    y2: int
    elongation: float
    motion_ratio: float

    def __post_init__(self) -> None:
        values = (self.elongation, self.motion_ratio)
        if self.frame_position < 0 or self.x1 < 0 or self.y1 < 0:
            raise ValueError("held punch box coordinates must be non-negative")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("held punch box corners must be ordered")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("held punch box measurements must be finite")

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def normalized(self, width: int, height: int) -> list[float]:
        if width <= 0 or height <= 0:
            raise ValueError("positive image dimensions are required")
        return [
            self.x1 / width,
            self.y1 / height,
            self.x2 / width,
            self.y2 / height,
        ]


@dataclass(frozen=True)
class HeldPunchMaskMeasurement:
    accepted: bool
    reason: str
    anchor_distance_radii: float
    inside_box_ratio: float
    green_ratio: float
    elongation: float


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


def locate_held_punch_box(
    frames_bgr: Sequence[np.ndarray], disk: MovingDisk | None,
) -> HeldPunchBox | None:
    """Locate the moving elongated metal body connected to an automatic disk anchor."""
    if disk is None:
        return None
    frames = [np.asarray(frame, dtype=np.uint8) for frame in frames_bgr]
    if not frames or not 0 <= disk.frame_position < len(frames):
        raise ValueError("disk frame position must refer to the supplied frames")
    shape = frames[0].shape
    if len(shape) != 3 or shape[2] != 3 or any(frame.shape != shape for frame in frames):
        raise ValueError("all frames must be same-size BGR images")

    seed = frames[disk.frame_position]
    height, width = shape[:2]
    background = np.median(np.stack(frames), axis=0).astype(np.uint8)
    gray_seed = cv2.cvtColor(seed, cv2.COLOR_BGR2GRAY)
    gray_background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    motion = cv2.absdiff(gray_seed, gray_background) > 18
    radius = max(4, round(disk.radius))
    motion_support = cv2.dilate(
        motion.astype(np.uint8),
        np.ones((max(3, radius // 4), max(3, radius // 4)), np.uint8),
    ).astype(bool)
    edges = cv2.Canny(gray_seed, 60, 160)
    moving_edges = (edges.astype(bool) & motion_support).astype(np.uint8) * 255
    lines = cv2.HoughLinesP(
        moving_edges,
        1,
        np.pi / 180,
        threshold=max(20, round(radius * 0.55)),
        minLineLength=max(30, round(radius * 1.8)),
        maxLineGap=max(10, round(radius * 0.5)),
    )
    line_candidates = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            distance = _point_segment_distance(
                disk.x, disk.y, float(x1), float(y1), float(x2), float(y2),
            )
            endpoint_distances = (
                math.dist((disk.x, disk.y), (float(x1), float(y1))),
                math.dist((disk.x, disk.y), (float(x2), float(y2))),
            )
            far_index = int(endpoint_distances[1] > endpoint_distances[0])
            far_distance = endpoint_distances[far_index]
            if distance <= 1.8 * disk.radius and far_distance >= 2.0 * disk.radius:
                far_point = ((x1, y1), (x2, y2))[far_index]
                line_candidates.append((far_distance, distance, far_point))
    if line_candidates:
        far_distance, _, (far_x, far_y) = max(
            line_candidates, key=lambda item: (item[0], -item[1]),
        )
        padding = max(4, round(disk.radius * 0.9))
        x1 = max(0, round(min(disk.x, far_x)) - padding)
        y1 = max(0, round(min(disk.y, far_y)) - padding)
        x2 = min(width, round(max(disk.x, far_x)) + padding + 1)
        y2 = min(height, round(max(disk.y, far_y)) + padding + 1)
        return HeldPunchBox(
            frame_position=disk.frame_position,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            elongation=float(far_distance / (2 * disk.radius)),
            motion_ratio=disk.motion_ratio,
        )

    hsv = cv2.cvtColor(seed, cv2.COLOR_BGR2HSV)
    low_saturation_bright = (hsv[..., 1] < 95) & (hsv[..., 2] > 75)
    candidate = (motion & low_saturation_bright).astype(np.uint8)

    close_size = max(3, round(radius * 0.45)) | 1
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        np.ones((close_size, close_size), np.uint8),
    )
    candidate = cv2.dilate(
        candidate,
        np.ones((max(3, radius // 4), max(3, radius // 4)), np.uint8),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate)
    yy, xx = np.ogrid[:height, :width]
    anchor = (xx - disk.x) ** 2 + (yy - disk.y) ** 2 <= (1.25 * disk.radius) ** 2
    matches = []
    for index in range(1, count):
        component = labels == index
        if not np.any(component & anchor):
            continue
        points = np.column_stack(np.nonzero(component))[:, ::-1].astype(np.float32)
        (_, _), (rect_width, rect_height), _ = cv2.minAreaRect(points)
        elongation = max(rect_width, rect_height) / max(1.0, min(rect_width, rect_height))
        area = int(stats[index, cv2.CC_STAT_AREA])
        matches.append((elongation, area, component))
    if not matches:
        return None
    elongation, _, component = max(matches, key=lambda item: (item[0], item[1]))
    if elongation < 1.35:
        return None

    ys, xs = np.nonzero(component)
    padding = max(4, round(disk.radius * 0.45))
    x1 = max(0, int(xs.min()) - padding)
    y1 = max(0, int(ys.min()) - padding)
    x2 = min(width, int(xs.max()) + padding + 1)
    y2 = min(height, int(ys.max()) + padding + 1)
    return HeldPunchBox(
        frame_position=disk.frame_position,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        elongation=float(elongation),
        motion_ratio=disk.motion_ratio,
    )


def _point_segment_distance(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float,
) -> float:
    dx = x2 - x1
    dy = y2 - y1
    denominator = dx * dx + dy * dy
    if denominator == 0:
        return math.dist((px, py), (x1, y1))
    position = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / denominator))
    return math.dist((px, py), (x1 + position * dx, y1 + position * dy))


def measure_held_punch_mask(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    disk: MovingDisk,
    box: HeldPunchBox,
) -> HeldPunchMaskMeasurement:
    """Measure one SAM3 mask and reject obvious instance or semantic swaps."""
    frame = np.asarray(frame_bgr, dtype=np.uint8)
    binary = np.asarray(mask, dtype=bool)
    if frame.ndim != 3 or frame.shape[2] != 3 or binary.shape != frame.shape[:2]:
        raise ValueError("frame and held punch mask dimensions must match")
    if not binary.any():
        return HeldPunchMaskMeasurement(False, "empty_mask", math.inf, 0.0, 0.0, 0.0)

    ys, xs = np.nonzero(binary)
    anchor_distance = float(
        np.sqrt((xs - disk.x) ** 2 + (ys - disk.y) ** 2).min() / disk.radius
    )
    inside = (
        (xs >= box.x1) & (xs < box.x2) & (ys >= box.y1) & (ys < box.y2)
    )
    inside_ratio = float(np.mean(inside))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0][binary]
    saturation = hsv[..., 1][binary]
    value = hsv[..., 2][binary]
    green_ratio = float(np.mean(
        (hue >= 30) & (hue <= 100) & (saturation >= 55) & (value >= 45)
    ))
    points = np.column_stack((xs, ys)).astype(np.float32)
    (_, _), (rect_width, rect_height), _ = cv2.minAreaRect(points)
    elongation = float(
        max(rect_width, rect_height) / max(1.0, min(rect_width, rect_height))
    )

    if anchor_distance > 1.25:
        reason = "anchor_missed"
    elif inside_ratio < 0.55:
        reason = "outside_tool_box"
    elif green_ratio > 0.35:
        reason = "green_sheet_mask"
    elif elongation < 1.35:
        reason = "not_elongated"
    else:
        reason = "criteria_satisfied"
    return HeldPunchMaskMeasurement(
        accepted=reason == "criteria_satisfied",
        reason=reason,
        anchor_distance_radii=anchor_distance,
        inside_box_ratio=inside_ratio,
        green_ratio=green_ratio,
        elongation=elongation,
    )


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
