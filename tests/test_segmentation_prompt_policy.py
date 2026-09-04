from medical_evaluation.annotations import PointPrompt, VideoAnnotations
from medical_evaluation.domain import TimeRange
from medical_evaluation.segmentation.prompt_policy import (
    AnnotationPromptPolicy,
    TextPromptPolicy,
)


def test_text_policy_ignores_annotation_object_prompts() -> None:
    annotations = VideoAnnotations(
        video_id="success",
        prompts=[
            PointPrompt(
                video_id="success",
                frame_time_sec=9,
                object_id="rubber_dam_frame",
                x=0.1,
                y=0.2,
            )
        ],
    )

    frame = TextPromptPolicy().frame_prompts(
        annotations,
        TimeRange(start_sec=10, end_sec=20),
        checkpoint_id="cp_09",
        boundary_tolerance_sec=0.5,
    )
    dam = TextPromptPolicy().dam_prompts(
        annotations,
        TimeRange(start_sec=30, end_sec=33),
        checkpoint_id="cp_11",
    )

    assert [(p.kind, p.text, p.frame_time_sec) for p in frame] == [
        ("text", "white U-shaped dental frame", 10)
    ]
    assert [(p.kind, p.text, p.frame_time_sec) for p in dam] == [
        ("text", "green dental rubber dam", 30)
    ]


def test_annotation_policy_preserves_existing_prompt_rules() -> None:
    annotations = VideoAnnotations(
        video_id="success",
        prompts=[
            PointPrompt(
                video_id="success",
                frame_time_sec=10,
                object_id="rubber_dam_frame",
                x=0.5,
                y=0.5,
            )
        ],
    )

    result = AnnotationPromptPolicy().frame_prompts(
        annotations,
        TimeRange(start_sec=10, end_sec=20),
        checkpoint_id="cp_09",
        boundary_tolerance_sec=0.5,
    )

    assert result[0].kind == "point"
