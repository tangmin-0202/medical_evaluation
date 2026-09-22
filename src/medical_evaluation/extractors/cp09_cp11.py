from __future__ import annotations

from pathlib import Path
from typing import Protocol

from medical_evaluation.annotations import BoxPrompt, VideoAnnotations
from medical_evaluation.domain import TimeRange
from medical_evaluation.pipeline import EvaluationInputMissing, ExtractedEvidence
from medical_evaluation.segmentation.prompt_policy import (
    AnnotationPromptPolicy,
    Cp09Cp11PromptPolicy,
    TextPromptPolicy,
)


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
        cp02: CheckpointExtractor | None = None,
        cp03: CheckpointExtractor | None = None,
        cp04: CheckpointExtractor | None = None,
        cp08: CheckpointExtractor | None = None,
        cp09: CheckpointExtractor,
        cp10: CheckpointExtractor | None = None,
        cp11: CheckpointExtractor,
        annotations: VideoAnnotations,
        prompt_policy: Cp09Cp11PromptPolicy | None = None,
    ) -> None:
        self.cp02 = cp02
        self.cp03 = cp03
        self.cp04 = cp04
        self.cp08 = cp08
        self.cp09 = cp09
        self.cp10 = cp10
        self.cp11 = cp11
        self.annotations = annotations
        self.prompt_policy = prompt_policy or AnnotationPromptPolicy()

    @property
    def model_version(self) -> str:
        cp02_version = self.cp02.model_version if self.cp02 is not None else "disabled"
        cp03_version = self.cp03.model_version if self.cp03 is not None else "disabled"
        cp04_version = self.cp04.model_version if self.cp04 is not None else "disabled"
        cp08_version = self.cp08.model_version if self.cp08 is not None else "disabled"
        cp10_version = self.cp10.model_version if self.cp10 is not None else "disabled"
        cp08_part = f";cp08={cp08_version}" if self.cp08 is not None else ""
        return (
            f"cp02={cp02_version};cp03={cp03_version};cp04={cp04_version}{cp08_part};"
            f"cp09={self.cp09.model_version};"
            f"cp10={cp10_version};cp11={self.cp11.model_version}"
        )

    def extract(
        self,
        video_path: Path,
        checkpoint_id: str,
        time_range: TimeRange,
        *,
        dense_fps: float,
        analysis_width: int,
    ) -> ExtractedEvidence:
        if checkpoint_id == "cp_02" and self.cp02 is not None:
            delegate = self.cp02
        elif checkpoint_id == "cp_03" and self.cp03 is not None:
            delegate = self.cp03
        elif checkpoint_id == "cp_04" and self.cp04 is not None:
            delegate = self.cp04
        elif checkpoint_id == "cp_08" and self.cp08 is not None:
            delegate = self.cp08
        elif checkpoint_id == "cp_09":
            self._validate_cp09_inputs(time_range)
            delegate = self.cp09
        elif checkpoint_id == "cp_10" and self.cp10 is not None:
            delegate = self.cp10
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
        if isinstance(self.prompt_policy, TextPromptPolicy):
            return
        start = time_range.start_sec - self.prompt_boundary_tolerance_sec
        end = time_range.end_sec + self.prompt_boundary_tolerance_sec
        try:
            self.prompt_policy.frame_prompts(
                self.annotations,
                time_range,
                checkpoint_id="cp_09",
                boundary_tolerance_sec=self.prompt_boundary_tolerance_sec,
            )
        except ValueError as exc:
            raise EvaluationInputMissing(str(exc)) from exc
        oral_boxes = [
            prompt
            for prompt in self.annotations.prompts
            if prompt.object_id == "oral_region"
            and isinstance(prompt, BoxPrompt)
            and start <= prompt.frame_time_sec <= end
        ]
        if len(oral_boxes) != 1:
            raise EvaluationInputMissing("cp_09 requires exactly one oral_region box")
