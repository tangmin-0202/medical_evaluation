from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
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


@dataclass(frozen=True)
class MarkObservation:
    u: float
    v: float
    frame_index: int
    time_sec: float


@dataclass(frozen=True)
class PunchSelection:
    status: Literal["selected", "missing", "ambiguous"]
    punch: MarkCandidate | None
    reason: str


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
            )
        )
    return sorted(stable, key=lambda item: (item.u, item.v))


def detect_dark_mark_observations(
    frame_bgr: np.ndarray,
    rubber_dam_mask: np.ndarray,
    *,
    frame_index: int,
    time_sec: float,
    min_area_ratio: float,
    max_area_ratio: float,
) -> list[MarkObservation]:
    """Find compact, near-neutral dark blobs inside the segmented dam."""

    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr must be an HxWx3 image")
    dam = np.asarray(rubber_dam_mask, dtype=bool)
    if dam.shape != frame_bgr.shape[:2]:
        raise ValueError("rubber_dam_mask shape must match the frame")
    if not 0 < min_area_ratio < max_area_ratio < 1:
        raise ValueError("mark area ratios must be ordered between zero and one")
    y_values, x_values = np.where(dam)
    if not len(x_values):
        return []

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    dark_neutral = dam & (value <= 160)
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
        if not min_area_ratio <= area_ratio <= max_area_ratio or aspect > 2.0:
            continue
        component = (labels == label).astype(np.uint8)
        contours, _ = cv2.findContours(
            component,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        perimeter = cv2.arcLength(max(contours, key=cv2.contourArea), True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter else 0.0
        if circularity < 0.35:
            continue
        center_x, center_y = centroids[label]
        results.append(
            MarkObservation(
                u=float((center_x - x_min) / width),
                v=float((center_y - y_min) / height),
                frame_index=frame_index,
                time_sec=time_sec,
            )
        )
    return sorted(results, key=lambda item: (item.u, item.v))


def _is_corner(candidate: MarkCandidate, margin: float) -> bool:
    horizontal_edge = candidate.u <= margin or candidate.u >= 1 - margin
    vertical_edge = candidate.v <= margin or candidate.v >= 1 - margin
    return horizontal_edge and vertical_edge
