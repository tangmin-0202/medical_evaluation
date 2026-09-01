from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest

from medical_evaluation.annotations import BoxPrompt, VideoAnnotations
from medical_evaluation.domain import TimeRange
from medical_evaluation.extractors.cp01 import Cp01FeatureExtractor
from medical_evaluation.segmentation.base import FrameMasks, SegmentationPrompt


class FakeSegmenter:
    def __init__(self, frames: list[FrameMasks]) -> None:
        self.frames = frames
        self.prompts: list[SegmentationPrompt] = []

    @property
    def model_version(self) -> str:
        return "fake-sam2"

    def track(
        self,
        _video_path: Path,
        _time_range: TimeRange,
        prompts: list[SegmentationPrompt],
        sample_fps: float,
    ) -> Iterator[FrameMasks]:
        assert sample_fps == 2
        self.prompts = prompts
        yield from self.frames


def _video(path: Path) -> Path:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10,
        (120, 100),
    )
    assert writer.isOpened()
    for index in range(10):
        frame = np.full((100, 120, 3), (30, 170, 90), dtype=np.uint8)
        if index >= 1:
            cv2.circle(frame, (20, 20), 3, (0, 0, 0), -1)
        if index >= 7:
            cv2.circle(frame, (62, 58), 4, (0, 0, 0), -1)
        writer.write(frame)
    writer.release()
    return path


def _annotations() -> VideoAnnotations:
    return VideoAnnotations(
        video_id="failure",
        prompts=[
            BoxPrompt(
                video_id="failure",
                frame_time_sec=0.1,
                object_id="rubber_dam",
                x1=0.08,
                y1=0.10,
                x2=0.92,
                y2=0.90,
            )
        ],
    )


def _tracked_frames() -> list[FrameMasks]:
    mask = np.zeros((100, 120), dtype=bool)
    mask[10:90, 10:110] = True
    return [
        FrameMasks(
            frame_index=index,
            frame_time_sec=index / 10,
            masks={"rubber_dam": mask},
        )
        for index in (1, 3, 5, 7, 9)
    ]


def test_extracts_punch_only_after_full_stage_candidate_collection(
    tmp_path: Path,
) -> None:
    evidence_root = tmp_path / "evidence"
    segmenter = FakeSegmenter(_tracked_frames())
    extractor = Cp01FeatureExtractor(
        segmenter=segmenter,
        annotations=_annotations(),
        evidence_root=evidence_root,
        reference_u=0.50,
        reference_v=0.60,
        min_observed_frames=2,
    )

    result = extractor.extract(
        _video(tmp_path / "video.avi"),
        "cp_01",
        TimeRange(start_sec=0.1, end_sec=1.0),
        dense_fps=2,
        analysis_width=1280,
    )

    assert {prompt.object_id for prompt in segmenter.prompts} == {"rubber_dam"}
    assert result.features["mark_candidate_count"] == 2.0
    assert result.features["mark_selection_ambiguous"] is False
    assert result.features["mark_u"] == pytest.approx(0.52, abs=0.03)
    assert result.features["mark_v"] == pytest.approx(0.60, abs=0.03)
    assert result.features["mark_reference_distance"] == pytest.approx(0.02, abs=0.03)
    assert result.features["dam_valid_frame_count"] == 5.0
    assert len(result.evidence) == 1
    assert result.evidence[0].rule == "auto_punch_relative_to_saved_reference"
    assert (evidence_root / result.evidence[0].overlay_path).is_file()
