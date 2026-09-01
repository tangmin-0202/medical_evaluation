from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class MarkCandidate:
    """One stable dark-mark track expressed in rubber-dam local coordinates."""

    u: float
    v: float
    first_sec: float
    last_sec: float
    observed_frame_count: int


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


def _is_corner(candidate: MarkCandidate, margin: float) -> bool:
    horizontal_edge = candidate.u <= margin or candidate.u >= 1 - margin
    vertical_edge = candidate.v <= margin or candidate.v >= 1 - margin
    return horizontal_edge and vertical_edge
