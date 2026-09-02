from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MaskOverlap:
    frame_index: int
    time_sec: float
    ratio: float


@dataclass(frozen=True)
class ContactEvent:
    detected: bool
    first_frame_index: int | None
    first_time_sec: float | None
    contact_frame_count: int
    maximum_ratio: float


def mask_overlap_ratio(subject: np.ndarray, target: np.ndarray) -> float:
    subject_mask = np.asarray(subject, dtype=bool)
    target_mask = np.asarray(target, dtype=bool)
    if subject_mask.shape != target_mask.shape:
        raise ValueError("overlap masks must have equal shapes")
    subject_area = int(subject_mask.sum())
    if subject_area == 0:
        return 0.0
    return float((subject_mask & target_mask).sum() / subject_area)


def stable_contact_event(
    observations: Sequence[MaskOverlap],
    *,
    minimum_ratio: float,
    minimum_consecutive_frames: int,
) -> ContactEvent:
    if not 0 <= minimum_ratio <= 1:
        raise ValueError("minimum_ratio must be between zero and one")
    if minimum_consecutive_frames <= 0:
        raise ValueError("minimum_consecutive_frames must be positive")

    ordered = sorted(observations, key=lambda item: (item.time_sec, item.frame_index))
    qualifying_count = sum(item.ratio >= minimum_ratio for item in ordered)
    maximum_ratio = max((item.ratio for item in ordered), default=0.0)
    run: list[MaskOverlap] = []
    first: MaskOverlap | None = None
    for item in ordered:
        if item.ratio < minimum_ratio:
            run.clear()
            continue
        run.append(item)
        if len(run) >= minimum_consecutive_frames:
            first = run[0]
            break

    return ContactEvent(
        detected=first is not None,
        first_frame_index=first.frame_index if first else None,
        first_time_sec=first.time_sec if first else None,
        contact_frame_count=qualifying_count,
        maximum_ratio=maximum_ratio,
    )
