import pytest

from medical_evaluation.annotations import BoxPrompt, PointPrompt, VideoAnnotations
from medical_evaluation.domain import TimeRange
from medical_evaluation.segmentation.prompts import prompts_for_object


def test_selects_object_prompts_inside_interval() -> None:
    annotations = VideoAnnotations(
        video_id="success",
        prompts=[
            PointPrompt(
                video_id="success",
                frame_time_sec=193.0,
                object_id="rubber_dam_frame",
                x=0.2,
                y=0.3,
            ),
            BoxPrompt(
                video_id="success",
                frame_time_sec=194.0,
                object_id="rubber_dam_frame",
                x1=0.1,
                y1=0.2,
                x2=0.8,
                y2=0.9,
            ),
            PointPrompt(
                video_id="success",
                frame_time_sec=100.0,
                object_id="rubber_dam_frame",
                x=0.4,
                y=0.5,
            ),
            PointPrompt(
                video_id="success",
                frame_time_sec=193.0,
                object_id="clamp",
                x=0.4,
                y=0.5,
            ),
        ],
    )

    result = prompts_for_object(
        annotations,
        "rubber_dam_frame",
        TimeRange(start_sec=175, end_sec=195),
    )

    assert [(item.kind, item.coordinates) for item in result] == [
        ("point", [0.2, 0.3]),
        ("box", [0.1, 0.2, 0.8, 0.9]),
    ]


def test_missing_required_prompt_is_actionable() -> None:
    with pytest.raises(ValueError, match="success.*cp_09.*rubber_dam_frame"):
        prompts_for_object(
            VideoAnnotations(video_id="success"),
            "rubber_dam_frame",
            TimeRange(start_sec=175, end_sec=195),
            checkpoint_id="cp_09",
        )
