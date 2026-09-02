from pathlib import Path

import pytest

from medical_evaluation.annotations import BoxPrompt, PointPrompt, VideoAnnotations
from medical_evaluation.domain import TimeRange
from medical_evaluation.extractors.cp09_cp11 import Cp09Cp11FeatureExtractor
from medical_evaluation.pipeline import EvaluationInputMissing, ExtractedEvidence


class RecordingExtractor:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def extract(
        self,
        _video_path: Path,
        checkpoint_id: str,
        _time_range: TimeRange,
        *,
        dense_fps: float,
        analysis_width: int,
    ) -> ExtractedEvidence:
        self.calls.append(checkpoint_id)
        return ExtractedEvidence(features={"dense_fps": dense_fps, "width": float(analysis_width)})


def annotations(*, include_oral_box: bool = True) -> VideoAnnotations:
    prompts = [
        PointPrompt(
            video_id="success",
            frame_time_sec=5,
            object_id="rubber_dam_frame",
            x=0.5,
            y=0.5,
        )
    ]
    if include_oral_box:
        prompts.append(
            BoxPrompt(
                video_id="success",
                frame_time_sec=5,
                object_id="oral_region",
                x1=0.2,
                y1=0.2,
                x2=0.8,
                y2=0.8,
            )
        )
    return VideoAnnotations(video_id="success", prompts=prompts)


def test_dispatches_only_to_matching_cp09_and_cp11_extractors() -> None:
    cp09 = RecordingExtractor()
    cp11 = RecordingExtractor()
    extractor = Cp09Cp11FeatureExtractor(cp09=cp09, cp11=cp11, annotations=annotations())

    for checkpoint_id in ("cp_09", "cp_11"):
        extractor.extract(
            Path("video.mp4"),
            checkpoint_id,
            TimeRange(start_sec=0, end_sec=10),
            dense_fps=2,
            analysis_width=1280,
        )

    assert cp09.calls == ["cp_09"]
    assert cp11.calls == ["cp_11"]


def test_missing_cp09_oral_box_is_reported_before_cp09_delegate_runs() -> None:
    cp09 = RecordingExtractor()
    extractor = Cp09Cp11FeatureExtractor(
        cp09=cp09,
        cp11=RecordingExtractor(),
        annotations=annotations(include_oral_box=False),
    )

    with pytest.raises(EvaluationInputMissing, match="oral_region"):
        extractor.extract(
            Path("video.mp4"),
            "cp_09",
            TimeRange(start_sec=0, end_sec=10),
            dense_fps=2,
            analysis_width=1280,
        )

    assert cp09.calls == []
