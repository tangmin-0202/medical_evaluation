from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from medical_evaluation.domain import TimeRange
from medical_evaluation.extractors.cp01_pen import Cp01PenGateExtractor


def _video(path: Path, *, pen_positions: set[int], frame_count: int = 6) -> Path:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 1, (180, 120)
    )
    assert writer.isOpened()
    for index in range(frame_count):
        frame = np.full((120, 180, 3), (180, 150, 120), dtype=np.uint8)
        cv2.rectangle(frame, (20, 20), (165, 105), (45, 155, 55), -1)
        if index in pen_positions:
            cv2.rectangle(frame, (50, 55), (140, 65), (25, 25, 25), -1)
        writer.write(frame)
    writer.release()
    return path


def _make_extractor(root: Path) -> Cp01PenGateExtractor:
    return Cp01PenGateExtractor(evidence_root=root)


def test_three_consecutive_automatic_pen_frames_pass_gate(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    result = _make_extractor(root).extract(
        _video(tmp_path / "video.avi", pen_positions={1, 2, 3}),
        "cp_01",
        TimeRange(start_sec=0, end_sec=5),
        dense_fps=5,
        analysis_width=1280,
    )

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
    root = tmp_path / "evidence"
    result = _make_extractor(root).extract(
        _video(tmp_path / "video.avi", pen_positions={2}),
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
    assert len(metrics["frames"]) == 5
    assert metrics["failure_reason"] == "pen_not_observed_consecutively"
    assert any(row["rejection_reason"] == "pen_not_observed" for row in metrics["frames"])
    assert list((root / "cp_01/pen_gate/raw").glob("*.jpg"))
    assert list((root / "cp_01/pen_gate/masks/raw").glob("*.png"))
    assert list((root / "cp_01/pen_gate/overlays").glob("*.jpg"))


def test_two_consecutive_pen_frames_pass_default_gate(tmp_path: Path) -> None:
    result = _make_extractor(tmp_path / "evidence").extract(
        _video(tmp_path / "video.avi", pen_positions={2, 3}),
        "cp_01",
        TimeRange(start_sec=0, end_sec=5),
        dense_fps=5,
        analysis_width=1280,
    )

    assert result.features["pen_presence_detected"] is True
    assert result.features["pen_max_consecutive_valid_frames"] == 2.0
