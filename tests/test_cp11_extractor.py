from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest

from medical_evaluation.annotations import BoxPrompt, PointPrompt, VideoAnnotations
from medical_evaluation.domain import TimeRange
from medical_evaluation.extractors.cp11 import Cp11FeatureExtractor
from medical_evaluation.segmentation.base import FrameMasks, SegmentationPrompt


class FakeSegmenter:
    model_version = "fake-sam2"

    def __init__(
        self,
        dam_frames: list[FrameMasks],
        frame_frames: list[FrameMasks],
    ) -> None:
        self.dam_frames = dam_frames
        self.frame_frames = frame_frames
        self.calls: list[set[str]] = []

    def track(
        self,
        _video_path: Path,
        _time_range: TimeRange,
        prompts: list[SegmentationPrompt],
        sample_fps: float,
    ) -> Iterator[FrameMasks]:
        object_ids = {item.object_id for item in prompts}
        self.calls.append(object_ids)
        if object_ids == {"rubber_dam"}:
            assert sample_fps == 2
            yield from self.dam_frames
        else:
            assert object_ids == {"rubber_dam_frame"}
            assert sample_fps == 1
            yield from self.frame_frames


def _video(path: Path, *, green_stage: bool) -> Path:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10,
        (40, 40),
    )
    assert writer.isOpened()
    for index in range(50):
        frame = np.zeros((40, 40, 3), dtype=np.uint8)
        if green_stage and index >= 10:
            frame[:] = (30, 170, 90)
        if index == 5:
            frame[14:26, 14:26] = 240
        writer.write(frame)
    writer.release()
    return path


def _mask(object_id: str, index: int, region: tuple[slice, slice]) -> FrameMasks:
    mask = np.zeros((40, 40), dtype=bool)
    mask[region] = True
    return FrameMasks(
        frame_index=index,
        frame_time_sec=index / 10,
        masks={object_id: mask},
    )


def _attempted_annotations() -> VideoAnnotations:
    return VideoAnnotations(
        video_id="success",
        prompts=[
            *[
                PointPrompt(
                    video_id="success",
                    frame_time_sec=3.0,
                    object_id="rubber_dam",
                    x=x,
                    y=y,
                )
                for x, y in ((0.3, 0.3), (0.7, 0.3), (0.5, 0.7))
            ],
            BoxPrompt(
                video_id="success",
                frame_time_sec=3.0,
                object_id="nose_region",
                x1=0.0,
                y1=0.0,
                x2=0.2,
                y2=0.2,
            ),
            PointPrompt(
                video_id="success",
                frame_time_sec=0.5,
                object_id="rubber_dam_frame",
                x=0.5,
                y=0.5,
            ),
        ],
    )


def test_no_green_dam_returns_incomplete_features_without_sam2(tmp_path: Path) -> None:
    segmenter = FakeSegmenter([], [])
    extractor = Cp11FeatureExtractor(
        segmenter=segmenter,
        annotations=VideoAnnotations(video_id="clamp_failure"),
        evidence_root=tmp_path / "evidence",
        frame_reference_time_range=TimeRange(start_sec=0.4, end_sec=0.6),
        min_stage_dam_presence_ratio=0.05,
    )

    result = extractor.extract(
        _video(tmp_path / "missing.avi", green_stage=False),
        "cp_11",
        TimeRange(start_sec=1.0, end_sec=4.0),
        dense_fps=2,
        analysis_width=1280,
    )

    assert result.features == {
        "dam_stage_presence_ratio": 0.0,
        "dam_area_ratio": None,
        "nose_overlap": None,
        "visible_frame_area_ratio": None,
        "final_valid_frame_count": 0.0,
    }
    assert segmenter.calls == []


def test_attempted_stage_extracts_good_final_coverage(tmp_path: Path) -> None:
    dam_region = (slice(10, 30), slice(10, 30))
    frame_region = (slice(14, 26), slice(14, 26))
    segmenter = FakeSegmenter(
        [_mask("rubber_dam", index, dam_region) for index in (10, 20, 30)],
        [_mask("rubber_dam_frame", index, frame_region) for index in (5, 10, 20, 30)],
    )
    evidence_root = tmp_path / "evidence"
    extractor = Cp11FeatureExtractor(
        segmenter=segmenter,
        annotations=_attempted_annotations(),
        evidence_root=evidence_root,
        frame_reference_time_range=TimeRange(start_sec=0.4, end_sec=0.6),
        min_stage_dam_presence_ratio=0.05,
    )

    result = extractor.extract(
        _video(tmp_path / "attempted.avi", green_stage=True),
        "cp_11",
        TimeRange(start_sec=1.0, end_sec=4.0),
        dense_fps=2,
        analysis_width=1280,
    )

    assert segmenter.calls == [{"rubber_dam"}, {"rubber_dam_frame"}]
    assert result.features["dam_stage_presence_ratio"] == pytest.approx(1.0)
    assert result.features["dam_area_ratio"] == pytest.approx(0.25)
    assert result.features["nose_overlap"] == 0.0
    assert result.features["visible_frame_area_ratio"] == 0.0
    assert result.features["final_valid_frame_count"] == 3.0
    assert len(result.evidence) == 3
    assert all((evidence_root / item.overlay_path).is_file() for item in result.evidence)
