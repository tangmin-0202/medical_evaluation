from __future__ import annotations

import numpy as np
import pytest

from medical_evaluation.features.contact import (
    MaskOverlap,
    mask_overlap_ratio,
    stable_contact_event,
)


def test_contact_requires_consecutive_sampled_frames() -> None:
    result = stable_contact_event(
        [
            MaskOverlap(frame_index=10, time_sec=0.5, ratio=0.03),
            MaskOverlap(frame_index=25, time_sec=1.0, ratio=0.00),
            MaskOverlap(frame_index=40, time_sec=1.5, ratio=0.04),
            MaskOverlap(frame_index=55, time_sec=2.0, ratio=0.05),
        ],
        minimum_ratio=0.02,
        minimum_consecutive_frames=2,
    )

    assert result.detected is True
    assert result.first_frame_index == 40
    assert result.first_time_sec == pytest.approx(1.5)
    assert result.contact_frame_count == 3
    assert result.maximum_ratio == pytest.approx(0.05)


def test_single_overlap_frame_does_not_form_contact() -> None:
    result = stable_contact_event(
        [
            MaskOverlap(frame_index=10, time_sec=0.5, ratio=0.03),
            MaskOverlap(frame_index=25, time_sec=1.0, ratio=0.00),
        ],
        minimum_ratio=0.02,
        minimum_consecutive_frames=2,
    )

    assert result.detected is False
    assert result.first_frame_index is None
    assert result.first_time_sec is None


def test_pen_overlap_is_normalized_by_pen_area() -> None:
    dam = np.zeros((20, 20), dtype=bool)
    pen = np.zeros((20, 20), dtype=bool)
    pen[5:15, 5:15] = True
    dam[10:15, 5:15] = True

    assert mask_overlap_ratio(pen, dam) == pytest.approx(0.5)


def test_empty_pen_mask_has_zero_overlap() -> None:
    empty = np.zeros((10, 10), dtype=bool)

    assert mask_overlap_ratio(empty, np.ones((10, 10), dtype=bool)) == 0.0
