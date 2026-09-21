from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from medical_evaluation.domain import TimeRange
from medical_evaluation.extractors.cp01_pen import (
    PEN_PROMPT,
    Cp01PenGateExtractor,
)
from medical_evaluation.segmentation.base import (
    FrameMasks,
    SegmentationPrompt,
)


def _video(path: Path, *, frame_count: int = 6) -> Path:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 1, (180, 120)
    )
    assert writer.isOpened()
    for _index in range(frame_count):
        frame = np.full((120, 180, 3), 220, dtype=np.uint8)
        frame[48:68, 25:155] = 25
        writer.write(frame)
    writer.release()
    return path


def _pen_mask() -> np.ndarray:
    mask = np.zeros((120, 180), dtype=bool)
    mask[48:68, 25:155] = True
    return mask


class FakeSegmenter:
    model_version = "fake-sam3"

    def __init__(self, accepted_positions: set[int]) -> None:
        self.accepted_positions = accepted_positions
        self.calls: list[tuple[TimeRange, list[SegmentationPrompt], float]] = []

    def track(
        self,
        _video_path: Path,
        time_range: TimeRange,
        prompts: list[SegmentationPrompt],
        sample_fps: float,
    ) -> Iterator[FrameMasks]:
        self.calls.append((time_range, prompts, sample_fps))
        for position in range(6):
            yield FrameMasks(
                frame_index=position,
                frame_time_sec=float(position),
                sample_position=position,
                masks={
                    "cp01_marking_pen": (
                        _pen_mask()
                        if position in self.accepted_positions
                        else np.zeros((120, 180), dtype=bool)
                    )
                },
            )


def _make_extractor(segmenter: FakeSegmenter, root: Path) -> Cp01PenGateExtractor:
    return Cp01PenGateExtractor(
        segmenter=segmenter,
        evidence_root=root,
        minimum_consecutive_frames=3,
    )


def test_three_consecutive_automatic_pen_frames_pass_gate(tmp_path: Path) -> None:
    segmenter = FakeSegmenter({1, 2, 3})
    root = tmp_path / "evidence"
    result = _make_extractor(segmenter, root).extract(
        _video(tmp_path / "video.avi"),
        "cp_01",
        TimeRange(start_sec=0, end_sec=5),
        dense_fps=5,
        analysis_width=1280,
    )

    assert len(segmenter.calls) == 1
    _time_range, prompts, sample_fps = segmenter.calls[0]
    assert sample_fps == 1.0
    assert len(prompts) == 1
    assert prompts[0].kind == "text"
    assert prompts[0].object_id == "cp01_marking_pen"
    assert prompts[0].text == PEN_PROMPT
    assert result.features == {
        "stage_scan_reliable": True,
        "pen_presence_detected": True,
        "pen_valid_frame_count": 3.0,
        "pen_max_consecutive_valid_frames": 3.0,
        "pen_first_seen_sec": 1.0,
        "pen_last_seen_sec": 3.0,
    }
    assert result.evidence
    assert all((root / item.overlay_path).is_file() for item in result.evidence)


def test_single_pen_frame_fails_gate_and_keeps_negative_evidence(
    tmp_path: Path,
) -> None:
    segmenter = FakeSegmenter({2})
    root = tmp_path / "evidence"
    result = _make_extractor(segmenter, root).extract(
        _video(tmp_path / "video.avi"),
        "cp_01",
        TimeRange(start_sec=0, end_sec=5),
        dense_fps=5,
        analysis_width=1280,
    )

    assert result.features["stage_scan_reliable"] is True
    assert result.features["pen_presence_detected"] is False
    assert result.features["pen_valid_frame_count"] == 1.0
    assert result.features["pen_max_consecutive_valid_frames"] == 1.0
    metrics_path = root / "cp_01/pen_gate/metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert len(metrics["frames"]) == 6
    assert metrics["failure_reason"] == "pen_not_observed_consecutively"
    assert any(row["rejection_reason"] == "pen_not_observed" for row in metrics["frames"])
    assert list((root / "cp_01/pen_gate/raw").glob("*.jpg"))
    assert list((root / "cp_01/pen_gate/masks/raw").glob("*.png"))
    assert list((root / "cp_01/pen_gate/overlays").glob("*.jpg"))
