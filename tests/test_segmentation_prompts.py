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


def test_default_prompt_selection_remains_strict_at_interval_boundaries() -> None:
    annotations = VideoAnnotations(
        video_id="strict",
        prompts=[
            PointPrompt(
                video_id="strict",
                frame_time_sec=174.999,
                object_id="oral_region",
                x=0.5,
                y=0.5,
            )
        ],
    )

    with pytest.raises(ValueError, match=r"175\.000-195\.000s"):
        prompts_for_object(
            annotations,
            "oral_region",
            TimeRange(start_sec=175.0, end_sec=195.0),
        )


def test_accepts_prompt_just_outside_interval_with_explicit_tolerance() -> None:
    annotations = VideoAnnotations(
        video_id="success",
        prompts=[
            BoxPrompt(
                video_id="success",
                frame_time_sec=174.686926,
                object_id="oral_region",
                x1=0.45,
                y1=0.39,
                x2=0.59,
                y2=0.76,
            )
        ],
    )
    result = prompts_for_object(
        annotations,
        "oral_region",
        TimeRange(start_sec=175.0, end_sec=195.0),
        checkpoint_id="cp_09",
        boundary_tolerance_sec=0.5,
    )

    assert len(result) == 1
    assert result[0].frame_time_sec == pytest.approx(174.686926)


def test_rejects_negative_boundary_tolerance() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        prompts_for_object(
            VideoAnnotations(video_id="invalid"),
            "oral_region",
            TimeRange(start_sec=175.0, end_sec=195.0),
            boundary_tolerance_sec=-0.1,
        )


def test_missing_prompt_error_includes_expanded_tolerant_range() -> None:
    with pytest.raises(ValueError, match=r"174\.500-195\.500s"):
        prompts_for_object(
            VideoAnnotations(video_id="missing"),
            "oral_region",
            TimeRange(start_sec=175.0, end_sec=195.0),
            boundary_tolerance_sec=0.5,
        )

@pytest.mark.parametrize("tolerance", [float("nan"), float("inf"), float("-inf")])
def test_rejects_non_finite_boundary_tolerance(tolerance: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        prompts_for_object(
            VideoAnnotations(video_id="invalid"),
            "oral_region",
            TimeRange(start_sec=175.0, end_sec=195.0),
            boundary_tolerance_sec=tolerance,
        )


def test_accepts_prompts_at_tolerance_boundaries() -> None:
    annotations = VideoAnnotations(
        video_id="edges",
        prompts=[
            PointPrompt(video_id="edges", frame_time_sec=174.5, object_id="oral_region", x=0.5, y=0.5),
            PointPrompt(video_id="edges", frame_time_sec=195.5, object_id="oral_region", x=0.5, y=0.5),
        ],
    )

    result = prompts_for_object(
        annotations,
        "oral_region",
        TimeRange(start_sec=175.0, end_sec=195.0),
        boundary_tolerance_sec=0.5,
    )

    assert [item.frame_time_sec for item in result] == [174.5, 195.5]


@pytest.mark.parametrize("frame_time_sec", [174.499999, 195.500001])
def test_rejects_prompts_just_outside_tolerance_boundaries(frame_time_sec: float) -> None:
    annotations = VideoAnnotations(
        video_id="outside",
        prompts=[
            PointPrompt(
                video_id="outside",
                frame_time_sec=frame_time_sec,
                object_id="oral_region",
                x=0.5,
                y=0.5,
            )
        ],
    )

    with pytest.raises(ValueError, match="174\\.500-195\\.500s"):
        prompts_for_object(
            annotations,
            "oral_region",
            TimeRange(start_sec=175.0, end_sec=195.0),
            boundary_tolerance_sec=0.5,
        )
