from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest

from medical_evaluation.annotations import BoxPrompt, PointPrompt, VideoAnnotations
from medical_evaluation.domain import TimeRange
from medical_evaluation.extractors.cp09 import Cp09FeatureExtractor
from medical_evaluation.segmentation.base import FrameMasks, SegmentationPrompt
from medical_evaluation.segmentation.prompt_policy import TextPromptPolicy


class FakeSegmenter:
    def __init__(self, frames: list[FrameMasks]) -> None:
        self.frames = frames
        self.time_range: TimeRange | None = None
        self.prompts: list[SegmentationPrompt] = []
        self.track_called = False

    @property
    def model_version(self) -> str:
        return "fake-sam2"

    def track(
        self,
        _video_path: Path,
        time_range: TimeRange,
        prompts: list[SegmentationPrompt],
        sample_fps: float,
    ) -> Iterator[FrameMasks]:
        assert sample_fps == 2
        self.track_called = True
        self.time_range = time_range
        self.prompts = prompts
        yield from self.frames


def _video(path: Path) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (40, 20))
    assert writer.isOpened()
    for _ in range(20):
        writer.write(np.full((20, 40, 3), 32, dtype=np.uint8))
    writer.release()
    return path


def _annotations(*, oral_region: str = "box") -> VideoAnnotations:
    prompts: list[PointPrompt | BoxPrompt] = [
        PointPrompt(
            video_id="success",
            frame_time_sec=1.0,
            object_id="rubber_dam_frame",
            x=0.5,
            y=0.5,
        )
    ]
    if oral_region in {"box", "duplicate"}:
        prompts.append(
            BoxPrompt(
                video_id="success",
                frame_time_sec=0.4,
                object_id="oral_region",
                x1=0.25,
                y1=0.2,
                x2=0.75,
                y2=0.8,
            )
        )
    if oral_region == "duplicate":
        prompts.append(
            BoxPrompt(
                video_id="success",
                frame_time_sec=0.45,
                object_id="oral_region",
                x1=0.2,
                y1=0.15,
                x2=0.8,
                y2=0.85,
            )
        )
    if oral_region == "point":
        prompts.append(
            PointPrompt(
                video_id="success",
                frame_time_sec=0.4,
                object_id="oral_region",
                x=0.5,
                y=0.5,
            )
        )
    return VideoAnnotations(video_id="success", prompts=prompts)


def _masks() -> tuple[np.ndarray, np.ndarray]:
    frame = np.zeros((20, 40), bool)
    frame[7:13, 15:25] = True
    oral = np.zeros((20, 40), bool)
    oral[4:16, 10:30] = True
    return frame, oral


def _frame(
    frame_index: int,
    frame_mask: np.ndarray | None,
    oral_mask: np.ndarray | None,
) -> FrameMasks:
    masks: dict[str, np.ndarray] = {}
    if frame_mask is not None:
        masks["rubber_dam_frame"] = frame_mask
    if oral_mask is not None:
        masks["oral_region"] = oral_mask
    return FrameMasks(
        frame_index=frame_index,
        frame_time_sec=frame_index / 10,
        masks=masks,
    )


def test_extracts_fixed_reference_feature_and_three_evidence_overlays(tmp_path: Path) -> None:
    frame_mask, _oral_mask = _masks()
    segmenter = FakeSegmenter(
        [
            _frame(4, frame_mask, None),
            _frame(5, frame_mask, None),
            _frame(10, frame_mask, None),
            _frame(15, frame_mask, None),
        ]
    )
    evidence_root = tmp_path / "evidence"
    extractor = Cp09FeatureExtractor(
        segmenter=segmenter,
        annotations=_annotations(),
        evidence_root=evidence_root,
    )

    result = extractor.extract(
        _video(tmp_path / "video.avi"),
        "cp_09",
        TimeRange(start_sec=0.5, end_sec=1.6),
        dense_fps=2,
        analysis_width=1280,
    )

    assert {prompt.object_id for prompt in segmenter.prompts} == {"rubber_dam_frame"}
    assert segmenter.time_range is not None
    assert segmenter.time_range.start_sec == pytest.approx(0.5)
    assert segmenter.time_range.end_sec == pytest.approx(1.6)
    assert result.features == {
        "frame_oral_center_offset": pytest.approx(0.0),
        "frame_valid_count": 3.0,
        "oral_reference_count": 1.0,
        "relative_offset_valid_count": 3.0,
    }
    assert len(result.evidence) == 3
    assert all(
        item.rule == "rubber_dam_frame_relative_to_oral_reference"
        for item in result.evidence
    )
    assert all(item.time_sec >= 0.5 for item in result.evidence)
    assert all((evidence_root / item.overlay_path).is_file() for item in result.evidence)
    assert len(list((evidence_root / "cp_09/masks/rubber_dam_frame").glob("*.png"))) == 3
    assert not (evidence_root / "cp_09/masks/oral_region").exists()


def test_text_policy_needs_no_manual_frame_prompt(tmp_path: Path) -> None:
    frame_mask, _ = _masks()
    segmenter = FakeSegmenter([_frame(index, frame_mask, None) for index in (5, 10, 15)])
    annotations = VideoAnnotations(
        video_id="success",
        prompts=[
            BoxPrompt(
                video_id="success",
                frame_time_sec=0.5,
                object_id="oral_region",
                x1=0.25,
                y1=0.2,
                x2=0.75,
                y2=0.8,
            )
        ],
    )
    extractor = Cp09FeatureExtractor(
        segmenter=segmenter,
        annotations=annotations,
        evidence_root=tmp_path / "evidence",
        prompt_policy=TextPromptPolicy(),
    )

    result = extractor.extract(
        _video(tmp_path / "video.avi"),
        "cp_09",
        TimeRange(start_sec=0.5, end_sec=1.6),
        dense_fps=2,
        analysis_width=1280,
    )

    assert result.features["frame_oral_center_offset"] == pytest.approx(0.0)
    assert [(p.kind, p.text) for p in segmenter.prompts] == [
        ("text", "white U-shaped dental frame")
    ]


def test_missing_frame_masks_produce_review_features(tmp_path: Path) -> None:
    extractor = Cp09FeatureExtractor(
        segmenter=FakeSegmenter(
            [
                _frame(5, None, None),
                _frame(10, None, None),
                _frame(15, None, None),
            ]
        ),
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

    assert result.features == {
        "frame_oral_center_offset": None,
        "frame_valid_count": 0.0,
        "oral_reference_count": 1.0,
        "relative_offset_valid_count": 0.0,
    }
    assert result.evidence == []


def test_two_frame_masks_keep_evidence_but_require_review(tmp_path: Path) -> None:
    frame_mask, _oral_mask = _masks()
    extractor = Cp09FeatureExtractor(
        segmenter=FakeSegmenter(
            [_frame(5, frame_mask, None), _frame(10, frame_mask, None)]
        ),
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

    assert result.features == {
        "frame_oral_center_offset": None,
        "frame_valid_count": 2.0,
        "oral_reference_count": 1.0,
        "relative_offset_valid_count": 2.0,
    }
    assert len(result.evidence) == 2


@pytest.mark.parametrize(
    ("oral_region", "message"),
    [
        ("missing", "exactly one oral_region box"),
        ("point", "oral_region must be a box"),
        ("duplicate", "exactly one oral_region box"),
    ],
)
def test_invalid_oral_reference_fails_before_tracking(
    tmp_path: Path,
    oral_region: str,
    message: str,
) -> None:
    segmenter = FakeSegmenter([])
    extractor = Cp09FeatureExtractor(
        segmenter=segmenter,
        annotations=_annotations(oral_region=oral_region),
        evidence_root=tmp_path / "evidence",
    )

    with pytest.raises(ValueError, match=message):
        extractor.extract(
            tmp_path / "video.avi",
            "cp_09",
            TimeRange(start_sec=0.5, end_sec=1.6),
            dense_fps=2,
            analysis_width=1280,
        )

    assert segmenter.track_called is False


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
