from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest

from medical_evaluation.annotations import PointPrompt, VideoAnnotations
from medical_evaluation.domain import TimeRange
from medical_evaluation.extractors.cp09 import Cp09FeatureExtractor
from medical_evaluation.segmentation.base import FrameMasks, SegmentationPrompt


class FakeSegmenter:
    def __init__(self, frames: list[FrameMasks]) -> None:
        self.frames = frames

    @property
    def model_version(self) -> str:
        return "fake-sam2"

    def track(
        self,
        _video_path: Path,
        _time_range: TimeRange,
        _prompts: list[SegmentationPrompt],
        sample_fps: float,
    ) -> Iterator[FrameMasks]:
        assert sample_fps == 2
        yield from self.frames


def _video(path: Path) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (40, 20))
    assert writer.isOpened()
    for _ in range(20):
        writer.write(np.full((20, 40, 3), 32, dtype=np.uint8))
    writer.release()
    return path


def _annotations() -> VideoAnnotations:
    return VideoAnnotations(
        video_id="success",
        prompts=[
            PointPrompt(
                video_id="success",
                frame_time_sec=1.0,
                object_id="rubber_dam_frame",
                x=0.5,
                y=0.5,
            )
        ],
    )


def _frame(frame_index: int, mask: np.ndarray) -> FrameMasks:
    return FrameMasks(
        frame_index=frame_index,
        frame_time_sec=frame_index / 10,
        masks={"rubber_dam_frame": mask},
    )


def test_extracts_center_feature_and_three_evidence_overlays(tmp_path: Path) -> None:
    centered = np.zeros((20, 40), bool)
    centered[5:15, 10:30] = True
    extractor = Cp09FeatureExtractor(
        segmenter=FakeSegmenter([_frame(5, centered), _frame(10, centered), _frame(15, centered)]),
        annotations=_annotations(),
        evidence_root=tmp_path / "evidence",
    )

    result = extractor.extract(
        _video(tmp_path / "video.avi"),
        "cp_09",
        TimeRange(start_sec=0.5, end_sec=1.6),
        dense_fps=2,
        analysis_width=1280,
    )

    assert result.features["frame_center_offset"] == pytest.approx(0.0, abs=0.03)
    assert result.features["frame_valid_count"] == 3.0
    assert len(result.evidence) == 3
    assert all((tmp_path / "evidence" / item.overlay_path).is_file() for item in result.evidence)
    assert len(list((tmp_path / "evidence" / "cp_09" / "masks").glob("*.png"))) == 3


def test_empty_masks_return_missing_feature_without_evidence(tmp_path: Path) -> None:
    empty = np.zeros((20, 40), bool)
    extractor = Cp09FeatureExtractor(
        segmenter=FakeSegmenter([_frame(5, empty), _frame(10, empty), _frame(15, empty)]),
        annotations=_annotations(),
        evidence_root=tmp_path / "evidence",
    )

    result = extractor.extract(
        _video(tmp_path / "video.avi"),
        "cp_09",
        TimeRange(start_sec=0.5, end_sec=1.6),
        dense_fps=2,
        analysis_width=1280,
    )

    assert result.features == {"frame_center_offset": None, "frame_valid_count": 0.0}
    assert result.evidence == []


def test_two_valid_masks_keep_evidence_but_require_review(tmp_path: Path) -> None:
    centered = np.zeros((20, 40), bool)
    centered[5:15, 10:30] = True
    extractor = Cp09FeatureExtractor(
        segmenter=FakeSegmenter([_frame(5, centered), _frame(10, centered)]),
        annotations=_annotations(),
        evidence_root=tmp_path / "evidence",
    )

    result = extractor.extract(
        _video(tmp_path / "video.avi"),
        "cp_09",
        TimeRange(start_sec=0.5, end_sec=1.6),
        dense_fps=2,
        analysis_width=1280,
    )

    assert result.features == {"frame_center_offset": None, "frame_valid_count": 2.0}
    assert len(result.evidence) == 2


def test_rejects_other_checkpoints(tmp_path: Path) -> None:
    extractor = Cp09FeatureExtractor(
        segmenter=FakeSegmenter([]),
        annotations=_annotations(),
        evidence_root=tmp_path,
    )

    with pytest.raises(ValueError, match="CP09-only"):
        extractor.extract(
            tmp_path / "video.avi",
            "cp_10",
            TimeRange(start_sec=0.5, end_sec=1.6),
            dense_fps=2,
            analysis_width=1280,
        )
