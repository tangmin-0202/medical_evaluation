from medical_evaluation.annotations import VideoAnnotations
from medical_evaluation.domain import TimeRange
from medical_evaluation.segmentation.base import SegmentationPrompt


def prompts_for_object(
    annotations: VideoAnnotations,
    object_id: str,
    time_range: TimeRange,
    *,
    checkpoint_id: str = "cp_09",
) -> list[SegmentationPrompt]:
    converted: list[SegmentationPrompt] = []
    for prompt in annotations.prompts:
        if prompt.object_id != object_id:
            continue
        if not time_range.start_sec <= prompt.frame_time_sec <= time_range.end_sec:
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
            f"{time_range.start_sec:.3f}-{time_range.end_sec:.3f}s"
        )
    return sorted(converted, key=lambda item: item.frame_time_sec)
