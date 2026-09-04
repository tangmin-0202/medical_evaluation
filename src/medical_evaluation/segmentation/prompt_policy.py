from __future__ import annotations

from typing import Protocol

from medical_evaluation.annotations import VideoAnnotations
from medical_evaluation.domain import TimeRange
from medical_evaluation.segmentation.base import SegmentationPrompt
from medical_evaluation.segmentation.prompts import prompts_for_object

FRAME_TEXT_PROMPT = "thin white U-shaped plastic frame around the mouth"
DAM_TEXT_PROMPT = "green dental rubber dam"


class Cp09Cp11PromptPolicy(Protocol):
    def frame_prompts(
        self,
        annotations: VideoAnnotations,
        time_range: TimeRange,
        *,
        checkpoint_id: str,
        boundary_tolerance_sec: float = 0.0,
    ) -> list[SegmentationPrompt]: ...

    def dam_prompts(
        self,
        annotations: VideoAnnotations,
        time_range: TimeRange,
        *,
        checkpoint_id: str,
    ) -> list[SegmentationPrompt]: ...


class AnnotationPromptPolicy:
    def frame_prompts(
        self,
        annotations: VideoAnnotations,
        time_range: TimeRange,
        *,
        checkpoint_id: str,
        boundary_tolerance_sec: float = 0.0,
    ) -> list[SegmentationPrompt]:
        return prompts_for_object(
            annotations,
            "rubber_dam_frame",
            time_range,
            checkpoint_id=checkpoint_id,
            boundary_tolerance_sec=boundary_tolerance_sec,
        )

    def dam_prompts(
        self,
        annotations: VideoAnnotations,
        time_range: TimeRange,
        *,
        checkpoint_id: str,
    ) -> list[SegmentationPrompt]:
        prompts = prompts_for_object(
            annotations,
            "rubber_dam",
            time_range,
            checkpoint_id=checkpoint_id,
        )
        if not 3 <= len(prompts) <= 5 or any(
            prompt.kind != "point" or not prompt.positive for prompt in prompts
        ):
            raise ValueError("cp_11 requires 3-5 positive rubber_dam points")
        return prompts


class TextPromptPolicy:
    def frame_prompts(
        self,
        _annotations: VideoAnnotations,
        time_range: TimeRange,
        *,
        checkpoint_id: str,
        boundary_tolerance_sec: float = 0.0,
    ) -> list[SegmentationPrompt]:
        return [
            SegmentationPrompt(
                object_id="rubber_dam_frame",
                kind="text",
                frame_time_sec=time_range.start_sec,
                text=FRAME_TEXT_PROMPT,
            )
        ]

    def dam_prompts(
        self,
        _annotations: VideoAnnotations,
        time_range: TimeRange,
        *,
        checkpoint_id: str,
    ) -> list[SegmentationPrompt]:
        return [
            SegmentationPrompt(
                object_id="rubber_dam",
                kind="text",
                frame_time_sec=time_range.start_sec,
                text=DAM_TEXT_PROMPT,
            )
        ]
