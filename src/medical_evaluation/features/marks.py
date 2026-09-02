from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

import cv2
import numpy as np


@dataclass(frozen=True)
class MarkCandidate:
    """One stable dark-mark track expressed in rubber-dam local coordinates."""

    u: float
    v: float
    first_sec: float
    last_sec: float
    observed_frame_count: int
    first_frame_index: int = 0
    last_frame_index: int = 0
    median_darkness: float = 0.0
    darkness_delta: float = 0.0


@dataclass(frozen=True)
class MarkObservation:
    u: float
    v: float
    frame_index: int
    time_sec: float
    local_darkness: float = 0.0


@dataclass(frozen=True)
class PunchSelection:
    status: Literal["selected", "missing", "ambiguous"]
    punch: MarkCandidate | None
    reason: str
    ranked_candidates: tuple[MarkCandidate, ...] = ()


def select_punch_candidate(
    candidates: Sequence[MarkCandidate],
    *,
    corner_margin: float,
) -> PunchSelection:
    """Select a punch only after all stable CP01 candidates are collected.

    Candidate timestamps are deliberately ignored.  A learner may place the
    corner helper point before or after the real punch-location point.
    """

    if not 0 < corner_margin < 0.5:
        raise ValueError("corner_margin must be between 0 and 0.5")
    if not candidates:
        return PunchSelection("missing", None, "no_stable_candidate")
    if len(candidates) == 1:
        return PunchSelection("selected", candidates[0], "single_stable_candidate")
    if len(candidates) == 2:
        corner_flags = [_is_corner(item, corner_margin) for item in candidates]
        if sum(corner_flags) == 1:
            punch = candidates[corner_flags.index(False)]
            return PunchSelection("selected", punch, "corner_auxiliary_removed")
    return PunchSelection("ambiguous", None, "ambiguous_stable_candidates")


def cluster_stable_marks(
    observations: Sequence[MarkObservation],
    *,
    min_observed_frames: int,
    max_local_distance: float,
) -> list[MarkCandidate]:
    """Cluster all observations after the stage has finished.

    The function intentionally has no streaming decision output: a point seen
    early cannot be declared the punch until later observations are available.
    """

    if min_observed_frames <= 0:
        raise ValueError("min_observed_frames must be positive")
    if not 0 < max_local_distance < 1:
        raise ValueError("max_local_distance must be between zero and one")

    clusters: list[list[MarkObservation]] = []
    for observation in sorted(observations, key=lambda item: (item.time_sec, item.u, item.v)):
        eligible: list[tuple[float, list[MarkObservation]]] = []
        for cluster in clusters:
            if any(item.frame_index == observation.frame_index for item in cluster):
                continue
            center_u = float(np.mean([item.u for item in cluster]))
            center_v = float(np.mean([item.v for item in cluster]))
            distance = float(np.hypot(observation.u - center_u, observation.v - center_v))
            if distance <= max_local_distance:
                eligible.append((distance, cluster))
        if eligible:
            min(eligible, key=lambda item: item[0])[1].append(observation)
        else:
            clusters.append([observation])

    stable: list[MarkCandidate] = []
    for cluster in clusters:
        frame_count = len({item.frame_index for item in cluster})
        if frame_count < min_observed_frames:
            continue
        stable.append(
            MarkCandidate(
                u=float(np.median([item.u for item in cluster])),
                v=float(np.median([item.v for item in cluster])),
                first_sec=min(item.time_sec for item in cluster),
                last_sec=max(item.time_sec for item in cluster),
                observed_frame_count=frame_count,
                first_frame_index=min(item.frame_index for item in cluster),
                last_frame_index=max(item.frame_index for item in cluster),
                median_darkness=float(
                    np.median([item.local_darkness for item in cluster])
                ),
            )
        )
    return sorted(stable, key=lambda item: (item.u, item.v))


def build_temporal_mark_tracks(
    *,
    pre_contact: Sequence[MarkObservation],
    post_contact: Sequence[MarkObservation],
    minimum_observed_frames: int,
    maximum_local_distance: float,
    minimum_darkness_delta: float,
) -> list[MarkCandidate]:
    preexisting = cluster_stable_marks(
        pre_contact,
        min_observed_frames=minimum_observed_frames,
        max_local_distance=maximum_local_distance,
    )
    post_tracks = cluster_stable_marks(
        post_contact,
        min_observed_frames=minimum_observed_frames,
        max_local_distance=maximum_local_distance,
    )
    candidates: list[MarkCandidate] = []
    for track in post_tracks:
        nearby = [
            (
                float(np.hypot(track.u - background.u, track.v - background.v)),
                background,
            )
            for background in preexisting
            if np.hypot(track.u - background.u, track.v - background.v)
            <= maximum_local_distance
        ]
        if not nearby:
            candidates.append(replace(track, darkness_delta=track.median_darkness))
            continue
        background = min(nearby, key=lambda item: item[0])[1]
        darkness_delta = track.median_darkness - background.median_darkness
        if darkness_delta >= minimum_darkness_delta:
            candidates.append(replace(track, darkness_delta=darkness_delta))
    return sorted(candidates, key=lambda item: (-item.median_darkness, item.u, item.v))


def select_darkest_nearest_reference(
    candidates: Sequence[MarkCandidate],
    *,
    reference_u: float,
    reference_v: float,
    maximum_candidates: int = 2,
) -> PunchSelection:
    if not 0 <= reference_u <= 1 or not 0 <= reference_v <= 1:
        raise ValueError("reference coordinates must be normalized")
    if maximum_candidates <= 0:
        raise ValueError("maximum_candidates must be positive")
    ranked = tuple(
        sorted(candidates, key=lambda item: (-item.median_darkness, item.u, item.v))[
            :maximum_candidates
        ]
    )
    if not ranked:
        return PunchSelection("missing", None, "no_new_or_darkened_mark")
    punch = min(
        ranked,
        key=lambda item: np.hypot(item.u - reference_u, item.v - reference_v),
    )
    return PunchSelection(
        "selected",
        punch,
        "nearest_reference_among_darkest",
        ranked,
    )


def detect_dark_mark_observations(
    frame_bgr: np.ndarray,
    rubber_dam_mask: np.ndarray,
    *,
    frame_index: int,
    time_sec: float,
    min_area_ratio: float,
    max_area_ratio: float,
    maximum_black_value: int = 55,
    maximum_faint_value: int = 160,
    maximum_faint_saturation: int = 120,
    max_mark_aspect_ratio: float = 2.0,
    min_mark_circularity: float = 0.35,
    local_darkness_ring_radius: int = 5,
) -> list[MarkObservation]:
    """Find compact, near-neutral dark blobs inside the segmented dam."""

    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr must be an HxWx3 image")
    dam = np.asarray(rubber_dam_mask, dtype=bool)
    if dam.shape != frame_bgr.shape[:2]:
        raise ValueError("rubber_dam_mask shape must match the frame")
    if not 0 < min_area_ratio < max_area_ratio < 1:
        raise ValueError("mark area ratios must be ordered between zero and one")
    if not 0 <= maximum_black_value <= maximum_faint_value <= 255:
        raise ValueError("mark value thresholds must be ordered between zero and 255")
    if not 0 <= maximum_faint_saturation <= 255:
        raise ValueError("maximum_faint_saturation must be between zero and 255")
    if max_mark_aspect_ratio < 1:
        raise ValueError("max_mark_aspect_ratio must be at least one")
    if not 0 <= min_mark_circularity <= 1:
        raise ValueError("min_mark_circularity must be between zero and one")
    if local_darkness_ring_radius <= 0:
        raise ValueError("local_darkness_ring_radius must be positive")
    y_values, x_values = np.where(dam)
    if not len(x_values):
        return []

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    dark_neutral = dam & (
        (value <= maximum_black_value)
        | (
            (value <= maximum_faint_value)
            & (saturation <= maximum_faint_saturation)
        )
    )
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        dark_neutral.astype(np.uint8),
        connectivity=8,
    )
    dam_area = int(dam.sum())
    x_min, x_max = int(x_values.min()), int(x_values.max()) + 1
    y_min, y_max = int(y_values.min()), int(y_values.max()) + 1
    width = max(x_max - x_min, 1)
    height = max(y_max - y_min, 1)
    results: list[MarkObservation] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        area_ratio = area / dam_area
        component_width = int(stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        aspect = max(component_width, component_height) / max(
            min(component_width, component_height),
            1,
        )
        if (
            not min_area_ratio <= area_ratio <= max_area_ratio
            or aspect > max_mark_aspect_ratio
        ):
            continue
        component = (labels == label).astype(np.uint8)
        contours, _ = cv2.findContours(
            component,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        perimeter = cv2.arcLength(max(contours, key=cv2.contourArea), True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter else 0.0
        if circularity < min_mark_circularity:
            continue
        component_mask = labels == label
        kernel_size = 2 * local_darkness_ring_radius + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        ring = (
            cv2.dilate(component_mask.astype(np.uint8), kernel).astype(bool)
            & dam
            & ~component_mask
        )
        local_background = float(np.median(value[ring])) if ring.any() else 0.0
        component_value = float(np.median(value[component_mask]))
        center_x, center_y = centroids[label]
        results.append(
            MarkObservation(
                u=float((center_x - x_min) / width),
                v=float((center_y - y_min) / height),
                frame_index=frame_index,
                time_sec=time_sec,
                local_darkness=max(local_background - component_value, 0.0),
            )
        )
    return sorted(results, key=lambda item: (item.u, item.v))


def _is_corner(candidate: MarkCandidate, margin: float) -> bool:
    horizontal_edge = candidate.u <= margin or candidate.u >= 1 - margin
    vertical_edge = candidate.v <= margin or candidate.v >= 1 - margin
    return horizontal_edge and vertical_edge
