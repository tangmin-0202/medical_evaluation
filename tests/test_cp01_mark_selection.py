from __future__ import annotations

import pytest

from medical_evaluation.features.marks import MarkCandidate, select_punch_candidate


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
