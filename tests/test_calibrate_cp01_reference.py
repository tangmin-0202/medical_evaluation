from __future__ import annotations

import pytest

from medical_evaluation.annotations import BoxPrompt, PointPrompt, VideoAnnotations
from medical_evaluation.domain import TimeRange
from scripts.calibrate_cp01_reference import calibrate_cp01_reference


def test_calibration_saves_reference_in_rubber_dam_local_coordinates() -> None:
    annotations = VideoAnnotations(
        video_id="success",
        prompts=[
            BoxPrompt(
                video_id="success",
                frame_time_sec=5.0,
                object_id="rubber_dam",
                x1=0.10,
                y1=0.20,
                x2=0.90,
                y2=0.80,
            ),
            PointPrompt(
                video_id="success",
                frame_time_sec=5.0,
                object_id="cp01_reference",
                x=0.42,
                y=0.56,
            ),
        ],
    )

    result = calibrate_cp01_reference(
        annotations,
        TimeRange(start_sec=0.0, end_sec=10.0),
    )

    assert result == {
        "reference_u": pytest.approx(0.40),
        "reference_v": pytest.approx(0.60),
    }
