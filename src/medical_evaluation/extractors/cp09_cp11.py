from __future__ import annotations

from pathlib import Path
from typing import Protocol

from medical_evaluation.annotations import BoxPrompt, VideoAnnotations
from medical_evaluation.domain import TimeRange
from medical_evaluation.pipeline import EvaluationInputMissing, ExtractedEvidence


class CheckpointExtractor(Protocol):
    @property
    def model_version(self) -> str: ...

    def extract(
        self,
        video_path: Path,
        checkpoint_id: str,
        time_range: TimeRange,
        *,
        dense_fps: float,
        analysis_width: int,
    ) -> ExtractedEvidence: ...


class Cp09Cp11FeatureExtractor:
    prompt_boundary_tolerance_sec = 0.5

    def __init__(
        self,
        *,
        cp09: CheckpointExtractor,
        cp11: CheckpointExtractor,
        annotations: VideoAnnotations,
    ) -> None:
        self.cp09 = cp09
        self.cp11 = cp11
        self.annotations = annotations

    @property
    def model_version(self) -> str:
        return f"cp09={self.cp09.model_version};cp11={self.cp11.model_version}"

    def extract(
        self,
        video_path: Path,
        checkpoint_id: str,
        time_range: TimeRange,
        *,
        dense_fps: float,
        analysis_width: int,
    ) -> ExtractedEvidence:
        if checkpoint_id == "cp_09":
            self._validate_cp09_inputs(time_range)
            delegate = self.cp09
        elif checkpoint_id == "cp_11":
            delegate = self.cp11
        else:
            raise ValueError(f"unsupported checkpoint: {checkpoint_id}")
        return delegate.extract(
            video_path,
            checkpoint_id,
            time_range,
            dense_fps=dense_fps,
            analysis_width=analysis_width,
        )

    def _validate_cp09_inputs(self, time_range: TimeRange) -> None:
        start = time_range.start_sec - self.prompt_boundary_tolerance_sec
        end = time_range.end_sec + self.prompt_boundary_tolerance_sec
        frame_prompts = [
            prompt
            for prompt in self.annotations.prompts
            if prompt.object_id == "rubber_dam_frame"
            and start <= prompt.frame_time_sec <= end
        ]
        oral_boxes = [
            prompt
            for prompt in self.annotations.prompts
            if prompt.object_id == "oral_region"
            and isinstance(prompt, BoxPrompt)
            and start <= prompt.frame_time_sec <= end
        ]
        if not frame_prompts:
            raise EvaluationInputMissing("cp_09 requires a rubber_dam_frame prompt")
        if len(oral_boxes) != 1:
            raise EvaluationInputMissing("cp_09 requires exactly one oral_region box")
