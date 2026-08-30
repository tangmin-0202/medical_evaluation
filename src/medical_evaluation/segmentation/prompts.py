from medical_evaluation.annotations import VideoAnnotations
from medical_evaluation.domain import TimeRange
from medical_evaluation.segmentation.base import SegmentationPrompt


def prompts_for_object(
    annotations: VideoAnnotations,
    object_id: str,
    time_range: TimeRange,
    *,
    checkpoint_id: str = "cp_09",
    boundary_tolerance_sec: float = 0.0,
) -> list[SegmentationPrompt]:
    if boundary_tolerance_sec < 0:
        raise ValueError("boundary_tolerance_sec must be non-negative")
    allowed_start = time_range.start_sec - boundary_tolerance_sec
    allowed_end = time_range.end_sec + boundary_tolerance_sec

    converted: list[SegmentationPrompt] = []
    for prompt in annotations.prompts:
        if prompt.object_id != object_id:
            continue
        if not allowed_start <= prompt.frame_time_sec <= allowed_end:
            continue
        if prompt.kind == "point":
            converted.append(
                SegmentationPrompt(
                    object_id=object_id,
                    kind="point",
                    frame_time_sec=prompt.frame_time_sec,
                    coordinates=[prompt.x, prompt.y],
                    positive=prompt.positive,
                )
            )
        elif prompt.kind == "box":
            converted.append(
                SegmentationPrompt(
                    object_id=object_id,
                    kind="box",
                    frame_time_sec=prompt.frame_time_sec,
                    coordinates=[prompt.x1, prompt.y1, prompt.x2, prompt.y2],
                )
            )
    if not converted:
        raise ValueError(
            f"{annotations.video_id} {checkpoint_id} has no {object_id} prompt inside "
            f"{allowed_start:.3f}-{allowed_end:.3f}s"
        )
    return sorted(converted, key=lambda item: item.frame_time_sec)
