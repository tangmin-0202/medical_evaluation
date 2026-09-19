"""Deterministic CP08 geometry and same-frame colour measurements.

The helpers deliberately report measurements and reliability separately.  They
do not decide the checkpoint result; that remains the CP08 Judge's job.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

import cv2
import numpy as np


@dataclass(frozen=True)
class InstrumentShapeMeasurement:
    observed: bool
    reliable: bool
    matches_proxy: bool | None
    reason: str
    input_component_count: int
    selected_component_area_px: int
    elongation: float | None
    handle_to_shaft_width_ratio: float | None
    curvature_deg: float | None
    component_mask: np.ndarray | None


@dataclass(frozen=True)
class ClampWingSplit:
    reliable: bool
    reason: str
    left_wing: np.ndarray | None
    right_wing: np.ndarray | None
    axis_xy: tuple[float, float] | None
    tooth_center_xy: tuple[float, float] | None
    left_complete: bool = False
    right_complete: bool = False
    left_completeness_score: float = 0.0
    right_completeness_score: float = 0.0


@dataclass(frozen=True)
class WingHoleColorMeasurement:
    hole_detected: bool
    reliable: bool
    reason: str
    dam_color_ratio: float | None
    non_dam_color_ratio: float | None
    hole_area_px: int
    sample_area_px: int
    hole_mask: np.ndarray | None
    sample_mask: np.ndarray | None


def _binary(mask: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(mask)
    if value.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional mask")
    return value.astype(bool).astype(np.uint8)


def _readonly(mask: np.ndarray) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    result.setflags(write=False)
    return result


def _components(binary: np.ndarray) -> list[tuple[int, np.ndarray]]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    result = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= 9:
            result.append((area, (labels == label).astype(np.uint8)))
    return sorted(result, key=lambda item: item[0], reverse=True)


def _component_elongation(component: np.ndarray) -> float:
    points = cv2.findNonZero(component)
    if points is None or len(points) < 5:
        return 1.0
    _, (width, height), _ = cv2.minAreaRect(points)
    return float(max(width, height) / max(min(width, height), 1.0))


def _axis_profile(component: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    y, x = np.nonzero(component)
    points = np.column_stack([x, y]).astype(np.float64)
    center = points.mean(axis=0)
    covariance = np.cov(points - center, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))]
    if axis[0] < 0:
        axis = -axis
    normal = np.array([-axis[1], axis[0]])
    longitudinal = (points - center) @ axis
    transverse = (points - center) @ normal
    edges = np.linspace(longitudinal.min(), longitudinal.max(), 17)
    centers: list[np.ndarray] = []
    widths: list[float] = []
    for index in range(len(edges) - 1):
        included = (longitudinal >= edges[index]) & (longitudinal < edges[index + 1])
        if included.sum() < 3:
            continue
        local_t = float(np.median(longitudinal[included]))
        local_u = float(np.median(transverse[included]))
        centers.append(center + local_t * axis + local_u * normal)
        widths.append(float(np.percentile(transverse[included], 95)
                            - np.percentile(transverse[included], 5) + 1.0))
    if len(centers) < 5:
        return np.asarray(centers), np.asarray(widths), 0.0
    centerline = np.asarray(centers)
    early = centerline[min(2, len(centerline) - 1)] - centerline[0]
    late = centerline[-1] - centerline[max(0, len(centerline) - 3)]
    denominator = max(float(np.linalg.norm(early) * np.linalg.norm(late)), 1e-6)
    angle = math.degrees(math.acos(float(np.clip(np.dot(early, late) / denominator, -1, 1))))
    return centerline, np.asarray(widths), angle


def _has_large_side_branch(component: np.ndarray, centerline: np.ndarray) -> bool:
    """Reject a compact clamp merged onto an otherwise long instrument.

    A valid handle-to-shaft silhouette follows one centreline.  A merged clamp
    produces a sizeable connected lobe far from every centreline segment.
    """
    if len(centerline) < 2:
        return False
    y, x = np.nonzero(component)
    points = np.column_stack([x, y]).astype(np.float32)
    line = centerline.astype(np.float32)
    distances = np.full(len(points), np.inf, dtype=np.float32)
    for start, end in pairwise(line):
        delta = end - start
        denom = float(np.dot(delta, delta))
        if denom <= 0:
            continue
        fraction = np.clip(((points - start) @ delta) / denom, 0, 1)
        projection = start + fraction[:, None] * delta
        distances = np.minimum(distances, np.linalg.norm(points - projection, axis=1))
    distance_map = cv2.distanceTransform(component, cv2.DIST_L2, 5)
    normal_radius = float(np.percentile(distance_map[component.astype(bool)], 85))
    return bool(np.mean(distances > max(2.8 * normal_radius, 12.0)) > 0.055)


def measure_instrument_shape(mask: np.ndarray) -> InstrumentShapeMeasurement:
    """Measure the agreed proxy: long handle, thin shaft, and curved end.

    Disconnected compact objects (notably the U-shaped rubber-dam clamp) are
    excluded before measurement.  Two plausible long objects or a sizeable
    merged side branch make the result unreliable rather than a forced match.
    """
    binary = _binary(mask, "instrument mask")
    parts = _components(binary)
    if not parts:
        return InstrumentShapeMeasurement(False, False, None, "instrument_not_observed",
                                          0, 0, None, None, None, None)
    elongated = [(area, part, _component_elongation(part)) for area, part in parts
                 if _component_elongation(part) >= 3.0]
    if len(elongated) >= 2 and elongated[1][0] >= 0.08 * elongated[0][0]:
        reason = ("ambiguous_instrument_components"
                  if elongated[1][0] >= 0.35 * elongated[0][0]
                  else "fragmented_instrument_component")
        return InstrumentShapeMeasurement(True, False, None, reason, len(parts),
                                          elongated[0][0], elongated[0][2], None, None,
                                          None)

    if not elongated:
        area, part = parts[0]
        centerline, widths, _ = _axis_profile(part)
        if (_component_elongation(part) >= 2.0 and len(widths) >= 5
                and float(widths.max()) > 2.0 * float(np.median(widths))):
            return InstrumentShapeMeasurement(
                True, False, None, "branched_or_merged_component", len(parts), area,
                _component_elongation(part), None, None, _readonly(part),
            )
        return InstrumentShapeMeasurement(True, True, False,
                                          "component_not_long_enough", len(parts), area,
                                          _component_elongation(part), 1.0, 0.0,
                                          _readonly(part))

    area, selected, elongation = elongated[0]
    centerline, widths, curvature = _axis_profile(selected)
    if len(widths) < 5:
        return InstrumentShapeMeasurement(True, False, None,
                                          "insufficient_component_geometry", len(parts),
                                          area, elongation, None, None, _readonly(selected))
    width_spike = float(widths.max()) > 2.2 * float(np.median(widths))
    if width_spike or _has_large_side_branch(selected, centerline):
        return InstrumentShapeMeasurement(True, False, None,
                                          "branched_or_merged_component", len(parts), area,
                                          elongation, None, None, _readonly(selected))
    narrow = float(np.percentile(widths, 25))
    broad = float(np.percentile(widths, 75))
    width_ratio = broad / max(narrow, 1.0)
    # The user-confirmed real instrument has a long straight handle and only a
    # short curved/tapered working end.  Requiring large whole-silhouette
    # curvature rejects that valid perspective.  Keep the strong elongation
    # gate and require either measurable taper or measurable end curvature;
    # a plain straight rod has neither.
    matches = bool(
        elongation >= 4.0
        and (width_ratio >= 1.25 or curvature >= 5.0)
    )
    reason = "shape_proxy_matched" if matches else "shape_proxy_not_matched"
    return InstrumentShapeMeasurement(True, True, matches, reason, len(parts), area,
                                      elongation, width_ratio, curvature,
                                      _readonly(selected))


def _centroid(binary: np.ndarray) -> np.ndarray | None:
    moments = cv2.moments(binary)
    if moments["m00"] <= 0:
        return None
    return np.array([moments["m10"] / moments["m00"],
                     moments["m01"] / moments["m00"]], dtype=np.float64)


def split_clamp_wings(tooth_mask: np.ndarray, clamp_mask: np.ndarray) -> ClampWingSplit:
    """Split clamp pixels about the tooth along the clamp's principal axis.

    The tooth-projected central band is deliberately removed so clamp body
    pixels cannot inflate either wing's completeness measurement.
    """
    tooth = _binary(tooth_mask, "tooth mask")
    clamp = _binary(clamp_mask, "clamp mask")
    if tooth.shape != clamp.shape:
        raise ValueError("tooth and clamp masks must have matching spatial dimensions")
    center = _centroid(tooth)
    y, x = np.nonzero(clamp)
    if center is None or len(x) < 20:
        return ClampWingSplit(False, "missing_tooth_or_clamp_reference", None, None,
                              None, None if center is None else tuple(center))
    points = np.column_stack([x, y]).astype(np.float64)
    values, vectors = np.linalg.eigh(np.cov(points - points.mean(axis=0), rowvar=False))
    axis = vectors[:, int(np.argmax(values))]
    if axis[0] < 0:
        axis = -axis
    tooth_y, tooth_x = np.nonzero(tooth)
    tooth_points = np.column_stack([tooth_x, tooth_y]).astype(np.float64)
    tooth_projection = (tooth_points - center) @ axis
    exclusion = max(float(np.percentile(np.abs(tooth_projection), 85)) * 0.82, 3.0)
    projection = (points - center) @ axis

    tooth_normal = np.array([-axis[1], axis[0]])
    tooth_normal_projection = (tooth_points - center) @ tooth_normal
    tooth_parallel_span = max(float(np.ptp(tooth_projection)), 1.0)
    tooth_normal_span = max(float(np.ptp(tooth_normal_projection)), 1.0)

    def side_mask(
        select: np.ndarray,
        side_projection: np.ndarray,
    ) -> tuple[np.ndarray | None, float, bool]:
        result = np.zeros_like(clamp)
        selected = points[select].astype(int)
        outward_extent = (float(np.max(np.abs(side_projection[select]))) - exclusion
                          if select.any() else 0.0)
        if len(selected) < 9 or outward_extent < max(7.0, 0.75 * exclusion):
            return None, 0.0, False
        result[selected[:, 1], selected[:, 0]] = 1
        parts = _components(result)
        if not parts:
            return None, 0.0, False
        wing = parts[0][1]
        wing_y, wing_x = np.nonzero(wing)
        wing_points = np.column_stack([wing_x, wing_y]).astype(np.float64)
        transverse_span = float(np.ptp((wing_points - center) @ tooth_normal))
        length_ratio = outward_extent / tooth_parallel_span
        breadth_ratio = transverse_span / tooth_normal_span
        score = float(np.clip(min(length_ratio, breadth_ratio), 0.0, 1.0))
        return _readonly(wing), score, bool(score >= 0.55)

    left, left_score, left_complete = side_mask(projection < -exclusion, projection)
    right, right_score, right_complete = side_mask(projection > exclusion, projection)
    return ClampWingSplit(True, "wings_split", left, right,
                          (float(axis[0]), float(axis[1])),
                          (float(center[0]), float(center[1])),
                          left_complete, right_complete, left_score, right_score)


def measure_wing_hole_color(
    frame_bgr: np.ndarray,
    wing_mask: np.ndarray,
    dam_mask: np.ndarray,
) -> WingHoleColorMeasurement:
    """Locate one enclosed wing hole and compare its inset to same-frame dam."""
    frame = np.asarray(frame_bgr)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be a BGR image")
    wing = _binary(wing_mask, "wing mask")
    dam = _binary(dam_mask, "dam mask")
    if frame.shape[:2] != wing.shape or wing.shape != dam.shape:
        raise ValueError("frame and masks must have matching spatial dimensions")
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    reference_hsv = hsv[dam.astype(bool)]
    green_reference = ((reference_hsv[:, 0] >= 30.0)
                       & (reference_hsv[:, 0] <= 100.0)
                       & (reference_hsv[:, 1] >= 35.0))
    if int(green_reference.sum()) < max(20, int(0.25 * len(reference_hsv))):
        return WingHoleColorMeasurement(False, False, "dam_green_reference_unreliable",
                                        None, None, 0, 0, None, None)
    green_hsv = reference_hsv[green_reference]
    hue_center = float(np.median(green_hsv[:, 0]))
    hue_spread = np.abs(green_hsv[:, 0] - hue_center)
    hue_tolerance = min(18.0, max(6.0, float(np.percentile(hue_spread, 90)) * 2.0 + 3.0))
    hue_distance = np.abs(hsv[:, :, 0] - hue_center)
    matching_image = ((hue_distance <= hue_tolerance)
                      & (hsv[:, :, 1] >= 25.0))
    contours, _ = cv2.findContours(wing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return WingHoleColorMeasurement(False, False, "wing_not_observed", None, None,
                                        0, 0, None, None)
    contour = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(wing)
    cv2.drawContours(filled, [contour], -1, 1, -1)
    holes = cv2.bitwise_and(filled, cv2.bitwise_not(wing))
    raw_candidates = [(area, item) for area, item in _components(holes) if area >= 20]
    filled_area = max(int(filled.sum()), 1)
    candidates: list[tuple[int, np.ndarray]] = []
    for area, item in raw_candidates:
        item_contours, _ = cv2.findContours(item, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)
        item_contour = max(item_contours, key=cv2.contourArea)
        contour_area = float(cv2.contourArea(item_contour))
        perimeter = float(cv2.arcLength(item_contour, True))
        _, (box_width, box_height), _ = cv2.minAreaRect(item_contour)
        _, enclosing_radius = cv2.minEnclosingCircle(item_contour)
        circularity = (4.0 * math.pi * contour_area / max(perimeter ** 2, 1.0))
        axis_ratio = min(box_width, box_height) / max(max(box_width, box_height), 1.0)
        enclosing_fill = contour_area / max(math.pi * enclosing_radius ** 2, 1.0)
        relative_area = area / filled_area
        if (circularity >= 0.65 and axis_ratio >= 0.55 and enclosing_fill >= 0.78
                and 0.005 <= relative_area <= 0.35):
            candidates.append((area, item))
    if not candidates:
        # SAM semantic masks commonly fill the tiny wing aperture.  In that
        # case the wing mask is only an automatic ROI: find the compact green
        # spot directly in the original frame instead of requiring a
        # topological hole in the binary mask.
        inner = cv2.erode(wing, np.ones((5, 5), np.uint8))
        if inner.any():
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dark_cutoff = float(np.percentile(gray[inner.astype(bool)], 25))
            proposals = (
                (gray < dark_cutoff) & inner.astype(bool)
            ).astype(np.uint8)
            appearance_candidates: list[tuple[float, int, np.ndarray]] = []
            wing_area = max(int(wing.sum()), 1)
            for area, item in _components(proposals):
                item_contours, _ = cv2.findContours(
                    item, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )
                item_contour = max(item_contours, key=cv2.contourArea)
                _, (box_width, box_height), _ = cv2.minAreaRect(item_contour)
                axis_ratio = min(box_width, box_height) / max(
                    max(box_width, box_height), 1.0
                )
                relative_area = area / wing_area
                green_ratio = float(matching_image[item.astype(bool)].mean())
                if (
                    area >= 12
                    and axis_ratio >= 0.45
                    and relative_area <= 0.08
                    and green_ratio >= 0.55
                ):
                    appearance_candidates.append(
                        (green_ratio * axis_ratio * math.sqrt(area), area, item)
                    )
            if appearance_candidates:
                _score, area, item = max(
                    appearance_candidates, key=lambda value: value[0]
                )
                candidates.append((area, item))
    if len(candidates) != 1:
        if raw_candidates and not candidates:
            reason = "hole_geometry_invalid"
        else:
            reason = "hole_not_detected" if not raw_candidates else "ambiguous_hole_candidates"
        return WingHoleColorMeasurement(False, False, reason, None, None, 0, 0,
                                        None, None)
    hole_area, hole = candidates[0]
    distance = cv2.distanceTransform(hole, cv2.DIST_L2, 5)
    maximum = float(distance.max())
    sample = (distance >= max(2.0, 0.38 * maximum)).astype(np.uint8)
    sample_area = int(sample.sum())
    reference_pixels = frame[dam.astype(bool)]
    if sample_area < 9 or len(reference_pixels) < 20:
        return WingHoleColorMeasurement(True, False, "insufficient_color_reference",
                                        None, None, hole_area, sample_area,
                                        _readonly(hole), _readonly(sample))
    sample_hsv = hsv[sample.astype(bool)]
    hue_distance = np.abs(sample_hsv[:, 0] - hue_center)
    matching = ((hue_distance <= hue_tolerance)
                & (sample_hsv[:, 1] >= 25.0))
    dam_ratio = float(matching.mean())
    return WingHoleColorMeasurement(True, True, "hole_color_measured", dam_ratio,
                                    1.0 - dam_ratio, hole_area, sample_area,
                                    _readonly(hole), _readonly(sample))
