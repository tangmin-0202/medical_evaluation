from __future__ import annotations

import cv2
import numpy as np
import pytest

from medical_evaluation.features.marks import (
    MarkCandidate,
    MarkObservation,
    cluster_stable_marks,
    detect_dark_mark_observations,
    select_punch_candidate,
)


def _candidate(u: float, v: float, *, first_sec: float) -> MarkCandidate:
    return MarkCandidate(
        u=u,
        v=v,
        first_sec=first_sec,
        last_sec=9.0,
        observed_frame_count=4,
    )


def test_one_stable_corner_point_is_selected_only_after_stage_collection() -> None:
    corner = _candidate(0.05, 0.08, first_sec=1.0)

    result = select_punch_candidate([corner], corner_margin=0.15)

    assert result.status == "selected"
    assert result.punch == corner
    assert result.reason == "single_stable_candidate"


@pytest.mark.parametrize(
    "ordered_candidates",
    [
        [_candidate(0.04, 0.06, first_sec=1.0), _candidate(0.48, 0.57, first_sec=8.0)],
        [_candidate(0.48, 0.57, first_sec=1.0), _candidate(0.04, 0.06, first_sec=8.0)],
    ],
)
def test_two_points_select_non_corner_regardless_of_time_order(
    ordered_candidates: list[MarkCandidate],
) -> None:
    result = select_punch_candidate(ordered_candidates, corner_margin=0.15)

    assert result.status == "selected"
    assert result.punch is not None
    assert result.punch.u == pytest.approx(0.48)
    assert result.punch.v == pytest.approx(0.57)
    assert result.reason == "corner_auxiliary_removed"


@pytest.mark.parametrize(
    "candidates",
    [
        [_candidate(0.04, 0.06, first_sec=1.0), _candidate(0.95, 0.92, first_sec=8.0)],
        [_candidate(0.35, 0.40, first_sec=1.0), _candidate(0.65, 0.60, first_sec=8.0)],
        [
            _candidate(0.04, 0.06, first_sec=1.0),
            _candidate(0.48, 0.57, first_sec=5.0),
            _candidate(0.70, 0.30, first_sec=8.0),
        ],
    ],
)
def test_ambiguous_final_candidate_sets_require_review(
    candidates: list[MarkCandidate],
) -> None:
    result = select_punch_candidate(candidates, corner_margin=0.15)

    assert result.status == "ambiguous"
    assert result.punch is None


def test_no_stable_candidate_is_reported_as_missing() -> None:
    result = select_punch_candidate([], corner_margin=0.15)

    assert result.status == "missing"
    assert result.punch is None


def test_stage_collection_keeps_early_corner_and_later_punch_until_end() -> None:
    observations = [
        MarkObservation(u=0.05, v=0.06, frame_index=index, time_sec=float(index))
        for index in range(1, 10)
    ]
    observations.extend(
        MarkObservation(u=0.48, v=0.57, frame_index=index, time_sec=float(index))
        for index in range(6, 10)
    )
    observations.append(
        MarkObservation(u=0.75, v=0.50, frame_index=4, time_sec=4.0)
    )

    candidates = cluster_stable_marks(
        observations,
        min_observed_frames=3,
        max_local_distance=0.04,
    )
    result = select_punch_candidate(candidates, corner_margin=0.15)

    assert len(candidates) == 2
    assert result.status == "selected"
    assert result.punch is not None
    assert result.punch.u == pytest.approx(0.48)
    assert result.reason == "corner_auxiliary_removed"


def test_dark_mark_detector_uses_dam_mask_and_rejects_large_shadow() -> None:
    frame = np.full((100, 120, 3), (30, 170, 90), dtype=np.uint8)
    dam_mask = np.zeros((100, 120), dtype=np.uint8)
    dam_mask[10:90, 10:110] = 1
    cv2.circle(frame, (25, 25), 3, (12, 12, 12), -1)
    cv2.circle(frame, (62, 57), 4, (8, 8, 8), -1)
    cv2.rectangle(frame, (80, 20), (105, 55), (25, 55, 35), -1)

    observations = detect_dark_mark_observations(
        frame,
        dam_mask,
        frame_index=7,
        time_sec=3.5,
        min_area_ratio=0.0005,
        max_area_ratio=0.01,
    )

    assert len(observations) == 2
    assert np.allclose(
        sorted((item.u, item.v) for item in observations),
        [(0.15, 0.19), (0.52, 0.59)],
        atol=0.03,
    )
