"""CP02 local punch-disk features; requires rectified, validated object ROIs."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from medical_evaluation.domain import TimeRange


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
    surface_contrast: float = 0.0

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

    def normalized(
        self, width: int, height: int, *, padding_px: int = 0,
    ) -> list[float]:
        if width <= 0 or height <= 0 or padding_px < 0:
            raise ValueError("positive image dimensions and non-negative padding required")
        return [
            max(0, self.x1 - padding_px) / width,
            max(0, self.y1 - padding_px) / height,
            min(width, self.x2 + padding_px) / width,
            min(height, self.y2 + padding_px) / height,
        ]


@dataclass(frozen=True)
class HeldPunchMaskMeasurement:
    accepted: bool
    reason: str
    anchor_distance_radii: float
    inside_box_ratio: float
    green_ratio: float
    elongation: float
    visible_hole_count: int = 0
    visible_disk_center: tuple[float, float] | None = None


@dataclass(frozen=True)
class DiskHoleLayout:
    holes: tuple[Hole, ...]
    reliable: bool
    reason: str
    second_largest_index: int | None
    ranking_confidence: float


@dataclass(frozen=True)
class PrePunchObservation:
    time_sec: float
    aligned_hole_index: int | None
    disk_to_handle_angle_deg: float | None
    reliable: bool

    def __post_init__(self) -> None:
        if not math.isfinite(self.time_sec) or self.time_sec < 0:
            raise ValueError("observation time must be finite and non-negative")
        if self.aligned_hole_index is not None and self.aligned_hole_index < 0:
            raise ValueError("hole index must be non-negative")
        if (
            self.disk_to_handle_angle_deg is not None
            and not math.isfinite(self.disk_to_handle_angle_deg)
        ):
            raise ValueError("relative wheel angle must be finite")


def prepunch_scan_range(
    cp02_range: TimeRange,
    cp03_range: TimeRange,
    *,
    max_interstage_gap_sec: float = 8.0,
) -> TimeRange:
    """Include a short interstage gap where the last wheel adjustment may occur."""
    if max_interstage_gap_sec <= 0:
        raise ValueError("positive interstage gap limit required")
    gap = cp03_range.start_sec - cp02_range.end_sec
    if 0 <= gap <= max_interstage_gap_sec:
        return TimeRange(
            start_sec=cp02_range.start_sec,
            end_sec=cp03_range.start_sec,
        )
    return cp02_range


def final_prepunch_hole(
    observations: Sequence[PrePunchObservation],
    *,
    contact_time_sec: float,
    max_gap_sec: float = 1.0,
    max_angle_delta_deg: float = 3.0,
) -> int | None:
    """Require the last two pre-contact frames to show one stable aligned hole.

    An earlier correct alignment cannot override a later disk adjustment. The
    angle is relative to the punch handle, so camera/tool motion is not treated
    as disk rotation. Missing late frames remain unknown, never pass.
    """
    if not math.isfinite(contact_time_sec) or contact_time_sec < 0:
        raise ValueError("contact time must be finite and non-negative")
    if max_gap_sec <= 0 or max_angle_delta_deg <= 0:
        raise ValueError("positive temporal and angular tolerances are required")
    before = sorted(
        (item for item in observations if item.time_sec < contact_time_sec),
        key=lambda item: item.time_sec,
    )
    if len(before) < 2:
        return None
    earlier, last = before[-2:]
    if (
        not earlier.reliable
        or not last.reliable
        or earlier.aligned_hole_index is None
        or last.aligned_hole_index is None
        or earlier.aligned_hole_index != last.aligned_hole_index
        or earlier.disk_to_handle_angle_deg is None
        or last.disk_to_handle_angle_deg is None
        or last.time_sec - earlier.time_sec > max_gap_sec
        or contact_time_sec - last.time_sec > max_gap_sec
    ):
        return None
    angle_delta = (
        last.disk_to_handle_angle_deg - earlier.disk_to_handle_angle_deg + 180
    ) % 360 - 180
    if abs(angle_delta) > max_angle_delta_deg:
        return None
    return last.aligned_hole_index


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
            if not _disk_fully_visible(
                refined_x, refined_y, refined_radius, width, height,
            ):
                continue
            refined_inner = (
                (xx - refined_x) ** 2 + (yy - refined_y) ** 2 < refined_radius ** 2
            )
            motion_ratio = float(
                np.mean(cv2.absdiff(gray, background)[refined_inner] > 20),
            )
            if hole_count >= min_holes and motion_ratio >= min_motion_ratio:
                surface_contrast = _disk_surface_contrast(
                    frames[frame_position], refined_x, refined_y, refined_radius,
                )
                if surface_contrast < 35.0:
                    continue
                candidates.append(
                    MovingDisk(
                        frame_position=frame_position,
                        x=float(refined_x),
                        y=float(refined_y),
                        radius=refined_radius,
                        hole_count=hole_count,
                        motion_ratio=motion_ratio,
                        surface_contrast=surface_contrast,
                    )
                )
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (item.surface_contrast, item.motion_ratio, item.hole_count),
    )


def locate_last_moving_multihole_disk(
    frames_bgr: Sequence[np.ndarray], *, window_frames: int = 3,
) -> MovingDisk | None:
    """Find the latest locally visible moving wheel, independent of SAM3 masks.

    A tool leaving the view must not erase an earlier clear pre-contact wheel.
    This is only a disk observation, not a selected-hole judgement.
    """
    if window_frames < 3:
        raise ValueError("at least three frames are required for motion evidence")
    frames = list(frames_bgr)
    for end in range(len(frames), window_frames - 1, -1):
        start = end - window_frames
        candidate = locate_moving_multihole_disk(frames[start:end])
        if candidate is not None:
            return MovingDisk(
                frame_position=start + candidate.frame_position,
                x=candidate.x,
                y=candidate.y,
                radius=candidate.radius,
                hole_count=candidate.hole_count,
                motion_ratio=candidate.motion_ratio,
                surface_contrast=candidate.surface_contrast,
            )
    return None


def measure_disk_holes(
    frame_bgr: np.ndarray,
    disk: MovingDisk,
    *,
    min_holes: int = 3,
    expected_hole_count: int = 5,
) -> DiskHoleLayout:
    """Measure the punch's five-hole layout inside an automatically located disk.

    ``MovingDisk.hole_count`` belongs to the coarse motion locator and can
    include dark punch-mechanism details.  Layout reliability therefore uses
    the known wheel layout rather than requiring both detectors to repeat the
    same counting error.
    """
    frame = np.asarray(frame_bgr, dtype=np.uint8)
    if (
        frame.ndim != 3
        or frame.shape[2] != 3
        or min_holes < 3
        or expected_hole_count < min_holes
    ):
        raise ValueError("BGR frame and at least three holes are required")
    height, width = frame.shape[:2]
    yy, xx = np.ogrid[:height, :width]
    search = (xx - disk.x) ** 2 + (yy - disk.y) ** 2 <= (1.05 * disk.radius) ** 2
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    metal = (
        search
        & (hsv[..., 1] < 105)
        & (hsv[..., 2] > 90)
    ).astype(np.uint8)
    metal = cv2.morphologyEx(
        metal, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8),
    )
    contours, _ = cv2.findContours(metal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = (float(disk.x), float(disk.y))
    containing = [c for c in contours if cv2.pointPolygonTest(c, center, False) >= 0]
    face_contour = max(containing or contours, key=cv2.contourArea) if contours else None
    if (
        face_contour is not None
        and cv2.contourArea(face_contour) >= 0.35 * math.pi * disk.radius**2
    ):
        face = np.zeros((height, width), np.uint8)
        cv2.drawContours(face, [cv2.convexHull(face_contour)], -1, 1, -1)
        face = face.astype(bool) & search
    else:
        # Warm procedure lighting makes the metal wheel highly saturated. The
        # motion/multi-hole locator has already established the disk geometry,
        # so use its conservative inner circle rather than rejecting the face.
        face = (xx - disk.x) ** 2 + (yy - disk.y) ** 2 <= (0.82 * disk.radius) ** 2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_values = gray[face]
    dark_limit = min(125.0, float(np.median(face_values)) - 25.0)
    dark = (face & (gray <= dark_limit)).astype(np.uint8)
    count, _labels, stats, centers = cv2.connectedComponentsWithStats(dark)
    holes: list[Hole] = []
    angles: list[float] = []
    min_area = max(3, round(disk.radius * disk.radius * 0.002))
    max_area = max(min_area + 1, round(disk.radius * disk.radius * 0.08))
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        component_width = int(stats[index, cv2.CC_STAT_WIDTH])
        component_height = int(stats[index, cv2.CC_STAT_HEIGHT])
        aspect = max(component_width, component_height) / max(
            1, min(component_width, component_height),
        )
        fill = area / max(1, component_width * component_height)
        x, y = map(float, centers[index])
        radial = math.dist((x, y), center) / disk.radius
        if not (
            min_area <= area <= max_area
            and aspect <= 1.8
            and fill >= 0.45
            and 0.12 <= radial <= 0.78
        ):
            continue
        holes.append(Hole(x=x, y=y, radius=math.sqrt(area / math.pi)))
        angles.append(math.degrees(math.atan2(y - disk.y, x - disk.x)) % 360)

    if len(holes) < min_holes:
        return DiskHoleLayout(
            tuple(holes), False, "insufficient_spatially_distinct_holes", None, 0.0,
        )
    if len(holes) != expected_hole_count:
        return DiskHoleLayout(
            tuple(holes), False, "unexpected_hole_count", None, 0.0,
        )
    ordered_angles = sorted(angles)
    gaps = [
        (ordered_angles[(i + 1) % len(ordered_angles)] - ordered_angles[i]) % 360
        for i in range(len(ordered_angles))
    ]
    angular_coverage = 360.0 - max(gaps)
    # This punch exposes its five sizes along a partial arc; requiring holes
    # around most of a full circle rejects the real, unobstructed wheel.
    if angular_coverage < 100.0:
        return DiskHoleLayout(
            tuple(holes), False, "insufficient_spatially_distinct_holes", None, 0.0,
        )
    radii = sorted((hole.radius for hole in holes), reverse=True)
    min_gap = max(0.35, 0.05 * radii[0])
    selected = second_largest_hole(
        holes, layout_reliable=True, min_radius_gap=min_gap,
    )
    if selected is None:
        return DiskHoleLayout(tuple(holes), False, "ambiguous_hole_ranking", None, 0.0)
    confidence = min(radii[0] - radii[1], radii[1] - radii[2]) / radii[0]
    return DiskHoleLayout(
        tuple(holes), True, "criteria_satisfied", selected, float(confidence),
    )


def locate_disk_layout_near(
    frame_bgr: np.ndarray,
    reference: MovingDisk,
    *,
    max_center_shift_radii: float = 3.0,
    expected_hole_count: int = 5,
) -> tuple[MovingDisk, DiskHoleLayout] | None:
    """Relocate a previously automatic disk anchor in one adjacent frame."""
    frame = np.asarray(frame_bgr, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3 or max_center_shift_radii <= 0:
        raise ValueError("BGR frame and positive tracking tolerance are required")
    height, width = frame.shape[:2]
    extent = round(reference.radius * (max_center_shift_radii + 1.8))
    x1 = max(0, round(reference.x) - extent)
    x2 = min(width, round(reference.x) + extent + 1)
    y1 = max(0, round(reference.y) - extent)
    y2 = min(height, round(reference.y) + extent + 1)
    crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(
        cv2.medianBlur(crop, 7),
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(16, round(reference.radius * 0.5)),
        param1=120,
        param2=24,
        minRadius=max(8, round(reference.radius * 0.5)),
        maxRadius=max(12, round(reference.radius * 1.7)),
    )
    if circles is None:
        return None
    candidates: list[tuple[float, MovingDisk, DiskHoleLayout]] = []
    for local_x, local_y, radius in np.round(circles[0]).astype(int):
        x = float(local_x + x1)
        y = float(local_y + y1)
        center_shift = math.dist((x, y), (reference.x, reference.y)) / reference.radius
        radius_ratio = float(radius) / reference.radius
        if center_shift > max_center_shift_radii or not 0.5 <= radius_ratio <= 1.7:
            continue
        disk = MovingDisk(
            frame_position=0,
            x=x,
            y=y,
            radius=float(radius),
            hole_count=expected_hole_count,
            motion_ratio=reference.motion_ratio,
            surface_contrast=_disk_surface_contrast(frame, x, y, float(radius)),
        )
        layout = measure_disk_holes(
            frame, disk, expected_hole_count=expected_hole_count,
        )
        if not layout.reliable:
            continue
        score = (
            center_shift
            + abs(math.log(radius_ratio))
            - 0.25 * layout.ranking_confidence
        )
        candidates.append((score, disk, layout))
    if not candidates:
        return None
    _, disk, layout = min(candidates, key=lambda item: item[0])
    return disk, layout


def _disk_fully_visible(x: float, y: float, radius: float, width: int, height: int) -> bool:
    return x - radius >= 0 and y - radius >= 0 and x + radius < width and y + radius < height


def _disk_surface_contrast(frame_bgr: np.ndarray, x: float, y: float, radius: float) -> float:
    """Compare a candidate disk face with its immediate outer surroundings."""
    height, width = frame_bgr.shape[:2]
    extent = round(1.8 * radius) + 1
    x1, x2 = max(0, round(x) - extent), min(width, round(x) + extent + 1)
    y1, y2 = max(0, round(y) - extent), min(height, round(y) + extent + 1)
    crop = frame_bgr[y1:y2, x1:x2]
    yy, xx = np.ogrid[y1:y2, x1:x2]
    distance_sq = (xx - x) ** 2 + (yy - y) ** 2
    inner = distance_sq < radius ** 2
    outer = (distance_sq >= (1.3 * radius) ** 2) & (
        distance_sq < (1.8 * radius) ** 2
    )
    if not inner.any() or not outer.any():
        return 0.0
    return float(np.linalg.norm(
        np.median(crop[inner], axis=0) - np.median(crop[outer], axis=0),
    ))


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
    """Accept a mask when a multi-hole disk is visible in its current-frame region."""
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

    visible_hole_count, visible_disk_center = _visible_disk_in_mask(
        frame, binary, disk.radius,
    )
    if green_ratio > 0.35:
        reason = "green_sheet_mask"
    elif visible_hole_count < 3:
        reason = "disk_holes_not_visible"
    else:
        reason = "criteria_satisfied"
    return HeldPunchMaskMeasurement(
        accepted=reason == "criteria_satisfied",
        reason=reason,
        anchor_distance_radii=anchor_distance,
        inside_box_ratio=inside_ratio,
        green_ratio=green_ratio,
        elongation=elongation,
        visible_hole_count=visible_hole_count,
        visible_disk_center=visible_disk_center,
    )


def _visible_disk_in_mask(
    frame_bgr: np.ndarray, mask: np.ndarray, seed_radius: float,
) -> tuple[int, tuple[float, float] | None]:
    """Find a multi-hole circle inside the current mask, without a fixed screen anchor."""
    ys, xs = np.nonzero(mask)
    padding = max(12, round(seed_radius))
    y1 = max(0, int(ys.min()) - padding)
    y2 = min(mask.shape[0], int(ys.max()) + padding + 1)
    x1 = max(0, int(xs.min()) - padding)
    x2 = min(mask.shape[1], int(xs.max()) + padding + 1)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    crop = gray[y1:y2, x1:x2]
    circles = cv2.HoughCircles(
        cv2.medianBlur(crop, 7), cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=max(12, round(seed_radius * 0.5)), param1=120, param2=24,
        minRadius=max(8, round(seed_radius * 0.45)),
        maxRadius=max(12, round(seed_radius * 2.5)),
    )
    if circles is None:
        return 0, None
    best_count = 0
    best_center = None
    best_rank = -math.inf
    for local_x, local_y, radius in np.round(circles[0]).astype(int):
        x, y = local_x + x1, local_y + y1
        if not (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and mask[y, x]):
            continue
        # In this fixed overhead view the wheel is at the upper end of the
        # hand-held punch; dark highlights in the lower press body are not holes.
        mask_height = max(1, int(ys.max()) - int(ys.min()))
        if (
            mask_height >= 3.0 * seed_radius
            and (y - int(ys.min())) / mask_height > 0.42
        ):
            continue
        top, bottom = max(0, y - radius), min(mask.shape[0], y + radius + 1)
        left, right = max(0, x - radius), min(mask.shape[1], x + radius + 1)
        yy, xx = np.ogrid[top:bottom, left:right]
        dark = ((xx - x) ** 2 + (yy - y) ** 2 < (0.8 * radius) ** 2) & (
            gray[top:bottom, left:right] < 110
        )
        count, _, stats, _ = cv2.connectedComponentsWithStats(dark.astype(np.uint8))
        holes = 0
        for index in range(1, count):
            width = int(stats[index, cv2.CC_STAT_WIDTH])
            height = int(stats[index, cv2.CC_STAT_HEIGHT])
            area = int(stats[index, cv2.CC_STAT_AREA])
            if (
                max(3, round(radius * radius * 0.002))
                <= area <= max(4, round(radius * radius * 0.20))
                and max(width, height) / max(1, min(width, height)) <= 1.8
                and area / (width * height) >= 0.45
            ):
                holes += 1
        contrast = _disk_surface_contrast(frame_bgr, x, y, radius)
        # A genuine wheel has both several openings and a distinct metal face.
        # Neither raw spot count nor border contrast alone is reliable when
        # SAM3 includes the glove or the lower press mechanism in its mask.
        rank = contrast + 15.0 * min(holes, 8)
        if holes >= 3 and rank > best_rank:
            best_rank = rank
            best_count = holes
            best_center = (float(x), float(y))
    return best_count, best_center


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


def aligned_hole_from_plunger(
    frame_bgr: np.ndarray,
    disk: MovingDisk,
    holes: Sequence[Hole],
    *,
    sector_half_angle_deg: float = 12.0,
    minimum_edge_support: float = 0.025,
    minimum_score_ratio: float = 1.2,
) -> int | None:
    """Select the hole whose radial axis continues into the opposite plunger.

    The Ainsworth die plate rotates but the tapered plunger is fixed to the
    opposite jaw.  Its metal outline therefore creates a supported radial
    continuation outside exactly one hole.  Ambiguous external structures are
    rejected instead of being resolved from screen direction.
    """
    frame = np.asarray(frame_bgr, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3 or not holes:
        raise ValueError("BGR frame and at least one hole are required")
    if (
        sector_half_angle_deg <= 0
        or minimum_edge_support <= 0
        or minimum_score_ratio <= 1
    ):
        raise ValueError("positive plunger support thresholds are required")
    height, width = frame.shape[:2]
    yy, xx = np.ogrid[:height, :width]
    dx = xx - disk.x
    dy = yy - disk.y
    distance = np.sqrt(dx * dx + dy * dy)
    angles = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
    annulus = (distance >= 0.78 * disk.radius) & (distance <= 2.2 * disk.radius)
    edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 60, 160) > 0
    scores: list[float] = []
    for hole in holes:
        angle = math.degrees(math.atan2(hole.y - disk.y, hole.x - disk.x)) % 360.0
        angle_delta = np.abs((angles - angle + 180.0) % 360.0 - 180.0)
        support = annulus & (angle_delta <= sector_half_angle_deg)
        scores.append(float(np.mean(edges[support])) if support.any() else 0.0)
    order = np.argsort(scores)[::-1]
    best = int(order[0])
    second_score = scores[int(order[1])] if len(order) > 1 else 0.0
    if (
        scores[best] < minimum_edge_support
        or scores[best] < minimum_score_ratio * max(second_score, 1e-6)
    ):
        return None
    return best


def aligned_hole_opposite_handle(
    frame_bgr: np.ndarray,
    disk: MovingDisk,
    holes: Sequence[Hole],
    *,
    maximum_alignment_angle_deg: float = 45.0,
    minimum_unique_gap_deg: float = 8.0,
) -> int | None:
    """Infer the pin-facing hole opposite the broad handle/jaw edge cluster.

    The five-hole die plate rotates while the handles and opposing jaw remain
    fixed.  The handle side produces a broad group of exterior metal edges;
    the selected hole is on the opposite radial side.  A non-unique angular
    match is deliberately left unresolved.
    """
    frame = np.asarray(frame_bgr, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3 or len(holes) < 2:
        raise ValueError("BGR frame and at least two holes are required")
    if maximum_alignment_angle_deg <= 0 or minimum_unique_gap_deg <= 0:
        raise ValueError("positive alignment tolerances are required")
    height, width = frame.shape[:2]
    yy, xx = np.ogrid[:height, :width]
    dx = xx - disk.x
    dy = yy - disk.y
    distance = np.sqrt(dx * dx + dy * dy)
    angles = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
    exterior = (distance >= 1.02 * disk.radius) & (distance <= 2.2 * disk.radius)
    edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 60, 160) > 0
    bin_centers = np.arange(0.0, 360.0, 10.0)
    support = []
    for center in bin_centers:
        delta = np.abs((angles - center + 180.0) % 360.0 - 180.0)
        sector = exterior & (delta <= 5.0)
        support.append(float(np.mean(edges[sector])) if sector.any() else 0.0)
    support_array = np.asarray(support)
    window_bins = 9
    window_scores = np.array([
        sum(support[(start + offset) % len(support)] for offset in range(window_bins))
        for start in range(len(support))
    ])
    best_start = int(np.argmax(window_scores))
    if window_scores[best_start] / window_bins < 0.015:
        return None
    selected_bins = [(best_start + offset) % len(support) for offset in range(window_bins)]
    weights = support_array[selected_bins]
    radians = np.deg2rad(bin_centers[selected_bins])
    body_angle = math.degrees(math.atan2(
        float(np.sum(weights * np.sin(radians))),
        float(np.sum(weights * np.cos(radians))),
    )) % 360.0
    pin_angle = (body_angle + 180.0) % 360.0
    distances = []
    for index, hole in enumerate(holes):
        hole_angle = math.degrees(
            math.atan2(hole.y - disk.y, hole.x - disk.x),
        ) % 360.0
        delta = abs((hole_angle - pin_angle + 180.0) % 360.0 - 180.0)
        distances.append((delta, index))
    distances.sort()
    if (
        distances[0][0] > maximum_alignment_angle_deg
        or distances[1][0] - distances[0][0] < minimum_unique_gap_deg
    ):
        return None
    return distances[0][1]


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
