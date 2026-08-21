from __future__ import annotations

import numpy as np

from medical_evaluation.step_localizer import (
    align_ordered_steps,
    cosine_cost,
    event_anchor,
    localized_segment,
    refine_with_anchors,
)


def test_alignment_stretches_steps_without_reordering() -> None:
    prototypes = np.eye(3, dtype=np.float32)
    sequence = np.stack(
        [prototypes[0], prototypes[0], prototypes[1], prototypes[2], prototypes[2]]
    )

    result = align_ordered_steps(
        sequence,
        prototypes,
        frame_times=np.arange(5.0),
        skip_penalty=1.5,
    )

    assert [item.step_index for item in result.segments] == [0, 1, 2]
    assert not any(item.missing for item in result.segments)
    assert result.segments[0].end_sec <= result.segments[1].start_sec
    assert result.state_path == [0, 0, 1, 2, 2]


def test_alignment_can_mark_a_missing_step() -> None:
    prototypes = np.eye(3, dtype=np.float32)
    sequence = np.stack([prototypes[0], prototypes[0], prototypes[2], prototypes[2]])

    result = align_ordered_steps(
        sequence,
        prototypes,
        frame_times=np.arange(4.0),
        skip_penalty=0.2,
    )

    assert result.segments[1].missing is True
    assert result.state_path == [0, 0, 2, 2]


def test_alignment_never_backtracks() -> None:
    prototypes = np.eye(3, dtype=np.float32)
    sequence = np.stack([prototypes[0], prototypes[2], prototypes[1], prototypes[2]])

    result = align_ordered_steps(sequence, prototypes, np.arange(4.0), skip_penalty=1.5)

    assert result.state_path == sorted(result.state_path)


def test_event_anchor_refines_boundary_without_changing_raw_boundary() -> None:
    segment = localized_segment(start_sec=10.0, end_sec=20.0)

    refined = refine_with_anchors(
        segment,
        [event_anchor("floss_appears", 12.5, confidence=0.9)],
    )

    assert refined.raw_start_sec == 10.0
    assert refined.start_sec == 12.5
    assert refined.anchor_evidence[0].name == "floss_appears"
    assert refined.anchor_evidence[0].accepted is True


def test_low_confidence_or_out_of_range_anchors_are_recorded_but_rejected() -> None:
    segment = localized_segment(start_sec=10.0, end_sec=20.0)

    refined = refine_with_anchors(
        segment,
        [
            event_anchor("mouth_entry", 12.0, confidence=0.2),
            event_anchor("frame_expansion", 30.0, confidence=0.9),
        ],
    )

    assert refined.start_sec == 10.0
    assert [item.accepted for item in refined.anchor_evidence] == [False, False]


def test_cosine_cost_is_zero_for_matching_unit_vectors() -> None:
    vectors = np.eye(2, dtype=np.float32)

    costs = cosine_cost(vectors, vectors)

    np.testing.assert_allclose(np.diag(costs), 0, atol=1e-6)
