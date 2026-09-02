from __future__ import annotations

import json
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


def _annotations(*, include_pen: bool) -> VideoAnnotations:
    prompts = [
        BoxPrompt(
            video_id="failure",
            frame_time_sec=0.1,
            object_id="rubber_dam",
            x1=0.08,
            y1=0.10,
            x2=0.92,
            y2=0.90,
        )
    ]
    if include_pen:
        prompts.append(
            BoxPrompt(
                video_id="failure",
                frame_time_sec=0.5,
                object_id="marking_pen",
                x1=0.45,
                y1=0.45,
                x2=0.60,
                y2=0.65,
            )
        )
    return VideoAnnotations(
        video_id="failure",
        prompts=prompts,
    )


def _tracked_frames(*, include_pen: bool) -> list[FrameMasks]:
    mask = np.zeros((100, 120), dtype=bool)
    mask[10:90, 10:110] = True
    empty_pen = np.zeros((100, 120), dtype=bool)
    visible_pen = np.zeros((100, 120), dtype=bool)
    visible_pen[0:8, 50:70] = True
    return [
        FrameMasks(
            frame_index=index,
            frame_time_sec=index / 10,
            masks={
                "rubber_dam": mask,
                **(
                    {
                        "marking_pen": (
                            visible_pen if index >= 5 else empty_pen
                        )
                    }
                    if include_pen
                    else {}
                ),
            },
        )
        for index in (1, 3, 5, 7, 9)
    ]


def test_pen_presence_selects_nearest_of_two_darkest_stable_marks(
    tmp_path: Path,
) -> None:
    evidence_root = tmp_path / "evidence"
    segmenter = FakeSegmenter(_tracked_frames(include_pen=True))
    extractor = Cp01FeatureExtractor(
        segmenter=segmenter,
        annotations=_annotations(include_pen=True),
        evidence_root=evidence_root,
        reference_u=0.50,
        reference_v=0.60,
        min_pen_presence_frames=2,
        min_mark_observed_frames=2,
    )

    result = extractor.extract(
        _video(tmp_path / "video.avi"),
        "cp_01",
        TimeRange(start_sec=0.1, end_sec=1.0),
        dense_fps=2,
        analysis_width=1280,
    )

    assert {prompt.object_id for prompt in segmenter.prompts} == {
        "rubber_dam",
        "marking_pen",
    }
    assert result.features["pen_presence_detected"] is True
    assert result.features["pen_valid_frame_count"] == 3.0
    assert result.features["mark_candidate_count"] == 2.0
    assert result.features["selected_mark_u"] == pytest.approx(0.52, abs=0.03)
    assert result.features["selected_mark_v"] == pytest.approx(0.60, abs=0.03)
    assert result.features["mark_reference_distance"] == pytest.approx(0.02, abs=0.03)
    assert result.features["dam_valid_frame_count"] == 5.0
    assert 1 <= len(result.evidence) <= 3
    assert all(
        (evidence_root / item.overlay_path).is_file()
        for item in result.evidence
    )
    assert (evidence_root / "cp_01/evidence.json").is_file()
    evidence_payload = json.loads(
        (evidence_root / "cp_01/evidence.json").read_text(encoding="utf-8")
    )
    assert evidence_payload["thresholds"] == {
        "min_pen_presence_frames": 2,
        "min_mark_observed_frames": 2,
        "min_mark_area_ratio": 0.0001,
        "max_mark_area_ratio": 0.01,
        "maximum_black_value": 55,
        "maximum_faint_value": 160,
        "maximum_faint_saturation": 120,
        "max_mark_aspect_ratio": 2.0,
        "min_mark_circularity": 0.35,
        "max_local_cluster_distance": 0.04,
        "local_darkness_ring_radius": 5,
    }


def test_missing_pen_prompt_returns_no_presence_features_without_error(
    tmp_path: Path,
) -> None:
    segmenter = FakeSegmenter(_tracked_frames(include_pen=False))
    extractor = Cp01FeatureExtractor(
        segmenter=segmenter,
        annotations=_annotations(include_pen=False),
        evidence_root=tmp_path / "evidence",
        reference_u=0.50,
        reference_v=0.60,
        min_pen_presence_frames=2,
        min_mark_observed_frames=2,
    )

    result = extractor.extract(
        _video(tmp_path / "video.avi"),
        "cp_01",
        TimeRange(start_sec=0.1, end_sec=1.0),
        dense_fps=2,
        analysis_width=1280,
    )

    assert {prompt.object_id for prompt in segmenter.prompts} == {"rubber_dam"}
    assert result.features["pen_presence_detected"] is False
    assert result.features["pen_valid_frame_count"] == 0.0
    assert result.features["mark_candidate_count"] == 0.0
    assert result.features["mark_reference_distance"] is None
