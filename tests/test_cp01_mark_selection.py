from __future__ import annotations

import cv2
import numpy as np
import pytest

from medical_evaluation.features.marks import (
    MarkCandidate,
    MarkObservation,
    build_temporal_mark_tracks,
    cluster_stable_marks,
    detect_dark_mark_observations,
    select_darkest_nearest_reference,
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


def test_dark_mark_detector_keeps_tiny_black_and_faint_gray_marks() -> None:
    frame = np.full((120, 120, 3), (30, 170, 90), dtype=np.uint8)
    dam_mask = np.ones((120, 120), dtype=np.uint8)
    cv2.circle(frame, (105, 12), 1, (20, 20, 20), -1)
    cv2.circle(frame, (78, 82), 2, (145, 145, 145), -1)
    cv2.circle(frame, (35, 70), 3, (20, 80, 40), -1)

    observations = detect_dark_mark_observations(
        frame,
        dam_mask,
        frame_index=9,
        time_sec=4.5,
        min_area_ratio=0.0001,
        max_area_ratio=0.01,
    )

    assert len(observations) == 2
    assert np.allclose(
        sorted((item.u, item.v) for item in observations),
        [(0.65, 0.68), (0.88, 0.10)],
        atol=0.03,
    )


def _observation(
    u: float,
    v: float,
    *,
    darkness: float,
    frame: int,
) -> MarkObservation:
    return MarkObservation(
        u=u,
        v=v,
        frame_index=frame,
        time_sec=frame / 2,
        local_darkness=darkness,
    )


def _dark_candidate(u: float, v: float, *, darkness: float) -> MarkCandidate:
    return MarkCandidate(
        u=u,
        v=v,
        first_sec=1.0,
        last_sec=3.0,
        observed_frame_count=3,
        first_frame_index=2,
        last_frame_index=6,
        median_darkness=darkness,
        darkness_delta=darkness,
    )


def test_pre_contact_template_marks_are_excluded() -> None:
    tracks = build_temporal_mark_tracks(
        pre_contact=[_observation(0.30, 0.30, darkness=35, frame=1)],
        post_contact=[_observation(0.30, 0.30, darkness=36, frame=5)],
        minimum_observed_frames=1,
        maximum_local_distance=0.04,
        minimum_darkness_delta=15,
    )

    assert tracks == []


def test_existing_mark_that_darkens_after_contact_becomes_candidate() -> None:
    tracks = build_temporal_mark_tracks(
        pre_contact=[_observation(0.60, 0.62, darkness=20, frame=1)],
        post_contact=[_observation(0.60, 0.62, darkness=55, frame=5)],
        minimum_observed_frames=1,
        maximum_local_distance=0.04,
        minimum_darkness_delta=15,
    )

    assert len(tracks) == 1
    assert tracks[0].darkness_delta == pytest.approx(35)


def test_new_post_contact_mark_becomes_candidate() -> None:
    tracks = build_temporal_mark_tracks(
        pre_contact=[_observation(0.20, 0.20, darkness=40, frame=1)],
        post_contact=[_observation(0.65, 0.65, darkness=70, frame=5)],
        minimum_observed_frames=1,
        maximum_local_distance=0.04,
        minimum_darkness_delta=15,
    )

    assert len(tracks) == 1
    assert tracks[0].u == pytest.approx(0.65)


def test_track_reconnects_after_temporary_occlusion() -> None:
    tracks = build_temporal_mark_tracks(
        pre_contact=[],
        post_contact=[
            _observation(0.60, 0.62, darkness=55, frame=5),
            _observation(0.61, 0.61, darkness=58, frame=8),
        ],
        minimum_observed_frames=2,
        maximum_local_distance=0.04,
        minimum_darkness_delta=15,
    )

    assert len(tracks) == 1
    assert tracks[0].observed_frame_count == 2


def test_darkest_two_choose_candidate_nearest_reference() -> None:
    selected = select_darkest_nearest_reference(
        [
            _dark_candidate(0.92, 0.06, darkness=90),
            _dark_candidate(0.62, 0.64, darkness=80),
            _dark_candidate(0.20, 0.30, darkness=30),
        ],
        reference_u=0.65,
        reference_v=0.65,
        maximum_candidates=2,
    )

    assert selected.punch is not None
    assert selected.punch.u == pytest.approx(0.62)
    assert len(selected.ranked_candidates) == 2


def test_detector_records_local_darkness_against_surrounding_dam() -> None:
    frame = np.full((80, 80, 3), (80, 180, 120), dtype=np.uint8)
    dam_mask = np.ones((80, 80), dtype=np.uint8)
    cv2.circle(frame, (40, 40), 3, (20, 20, 20), -1)

    observations = detect_dark_mark_observations(
        frame,
        dam_mask,
        frame_index=4,
        time_sec=2.0,
        min_area_ratio=0.0001,
        max_area_ratio=0.01,
        local_darkness_ring_radius=5,
    )

    assert len(observations) == 1
    assert observations[0].local_darkness > 80
